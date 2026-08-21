from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from app.services.agents.query_agent import analyze_query
from app.services.agents.retrieval_agent import retrieve_documents
from app.services.agents.response_agent import generate_response
from app.services.citation_verifier import verify_answer


class RAGState(TypedDict, total=False):
    question: str
    workspace_id: int
    retrieval_mode: str

    query_result: dict[str, Any]
    retrieval_result: dict[str, Any]

    retry_used: bool

    answer: str

    citation_verified: bool
    regeneration_count: int


def query_node(
    state: RAGState
) -> dict[str, Any]:

    query_result = analyze_query(
        state["question"]
    )

    return {
        "query_result": query_result
    }


def retrieve_node(
    state: RAGState
) -> dict[str, Any]:

    query_result = state["query_result"]
    
    mode = state.get("retrieval_mode", "fast")
    strategy_mode = "hybrid_rerank" if mode == "accurate" else "hybrid"

    retrieval_result = retrieve_documents(
        question=state["question"],
        workspace_id=state["workspace_id"],
        retrieval_count=query_result[
            "retrieval_count"
        ],
        retrieval_strategy=query_result[
            "retrieval_strategy"
        ],
        retrieval_mode=strategy_mode
    )

    return {
        "retrieval_result": retrieval_result,
        "retry_used": False
    }


def is_retrieval_weak(
    retrieval_result: dict[str, Any],
    expected_count: int
) -> bool:

    retrieved_count = retrieval_result[
        "retrieved_count"
    ]

    context = retrieval_result[
        "context"
    ]

    faiss_distances = retrieval_result.get(
        "faiss_distances",
        []
    )

    if not context.strip():
        return True

    if retrieved_count < expected_count:
        return True

    if not faiss_distances:
        return True

    average_distance = (
        sum(faiss_distances)
        / len(faiss_distances)
    )

    print(
        f"Average FAISS distance: "
        f"{average_distance:.4f}"
    )

    return average_distance > 0.8


def retrieval_router(
    state: RAGState
) -> str:

    retrieval_result = state[
        "retrieval_result"
    ]

    query_result = state[
        "query_result"
    ]

    if is_retrieval_weak(
        retrieval_result,
        query_result["retrieval_count"]
    ):
        return "retry"

    return "generate"


def retry_retrieval_node(
    state: RAGState
) -> dict[str, Any]:

    print(
        "Weak retrieval detected. "
        "Retrying with broader retrieval..."
    )

    retrieval_result = retrieve_documents(
        question=state["question"],
        workspace_id=state["workspace_id"],
        retrieval_count=10,
        retrieval_strategy="broad"
    )

    return {
        "retrieval_result": retrieval_result,
        "retry_used": True
    }


def generate_node(
    state: RAGState
) -> dict[str, Any]:

    retrieval_result = state[
        "retrieval_result"
    ]

    response_result = generate_response(
        question=state["question"],
        context=retrieval_result["context"]
    )

    return {
        "answer": response_result["answer"],
        "regeneration_count": 0
    }


def verify_node(
    state: RAGState
) -> dict[str, Any]:

    retrieval_result = state[
        "retrieval_result"
    ]

    citation_verified = verify_answer(
        question=state["question"],
        answer=state["answer"],
        context=retrieval_result["context"]
    )

    return {
        "citation_verified": citation_verified
    }


def citation_router(
    state: RAGState
) -> str:

    if state["citation_verified"]:
        return "end"

    regeneration_count = state.get(
        "regeneration_count",
        0
    )

    if regeneration_count >= 1:
        return "safe_response"

    return "regenerate"


def regenerate_node(
    state: RAGState
) -> dict[str, Any]:

    print(
        "Citation verification failed. "
        "Regenerating answer once..."
    )

    retrieval_result = state[
        "retrieval_result"
    ]

    response_result = generate_response(
        question=state["question"],
        context=retrieval_result["context"]
    )

    return {
        "answer": response_result["answer"],
        "regeneration_count": (
            state.get(
                "regeneration_count",
                0
            )
            + 1
        )
    }


def safe_response_node(
    state: RAGState
) -> dict[str, Any]:

    print(
        "Citation verification failed after "
        "regeneration. Returning safe response."
    )

    return {
        "answer": (
            "I couldn't verify that answer using "
            "the uploaded documents."
        ),
        "citation_verified": False
    }


def build_rag_graph():

    graph: StateGraph = StateGraph(
        RAGState
    )

    graph.add_node(
        "query",
        query_node
    )

    graph.add_node(
        "retrieve",
        retrieve_node
    )

    graph.add_node(
        "retry_retrieval",
        retry_retrieval_node
    )

    graph.add_node(
        "generate",
        generate_node
    )

    graph.add_node(
        "verify",
        verify_node
    )

    graph.add_node(
        "regenerate",
        regenerate_node
    )

    graph.add_node(
        "safe_response",
        safe_response_node
    )

    graph.add_edge(
        START,
        "query"
    )

    graph.add_edge(
        "query",
        "retrieve"
    )

    graph.add_conditional_edges(
        "retrieve",
        retrieval_router,
        {
            "retry": "retry_retrieval",
            "generate": "generate"
        }
    )

    graph.add_edge(
        "retry_retrieval",
        "generate"
    )

    graph.add_edge(
        "generate",
        "verify"
    )

    graph.add_conditional_edges(
        "verify",
        citation_router,
        {
            "end": END,
            "regenerate": "regenerate",
            "safe_response": "safe_response"
        }
    )

    graph.add_edge(
        "regenerate",
        "verify"
    )

    graph.add_edge(
        "safe_response",
        END
    )

    return graph.compile()


rag_graph = build_rag_graph()