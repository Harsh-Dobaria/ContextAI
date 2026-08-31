from typing import Any
import time

from app.services.agents.rag_graph import rag_graph


def run_rag_pipeline(
    question: str,
    workspace_id: int,
    retrieval_mode: str = "fast"
) -> dict[str, Any]:

    pipeline_start = time.perf_counter()

    # -----------------------------
    # Run LangGraph workflow
    # -----------------------------

    final_state = rag_graph.invoke(
        {
            "question": question,
            "workspace_id": workspace_id,
            "retrieval_mode": retrieval_mode
        }
    )

    total_time = (
        time.perf_counter()
        - pipeline_start
    )

    # -----------------------------
    # Extract results
    # -----------------------------

    query_result = final_state.get(
        "query_result",
        {}
    )

    retrieval_result = final_state.get(
        "retrieval_result",
        {}
    )

    answer = final_state.get(
        "answer",
        "I couldn't generate an answer."
    )

    retry_used = final_state.get(
        "retry_used",
        False
    )

    citation_verified = final_state.get(
        "citation_verified",
        False
    )

    # -----------------------------
    # Performance logs
    # -----------------------------

    print(
        f"Retry Used: "
        f"{retry_used}"
    )

    print(
        f"Citation Verified: "
        f"{citation_verified}"
    )

    print(
        f"Total Pipeline: "
        f"{total_time:.2f}s"
    )

    # -----------------------------
    # Return response
    # -----------------------------

    return {
        "question": question,
        "answer": answer,
        "sources": retrieval_result.get(
            "chunk_ids",
            []
        ),
        "query_analysis": query_result,
        "retrieved_chunks": retrieval_result.get(
            "retrieved_count",
            0
        ),
        "retry_used": retry_used,
        "citation_verified": citation_verified,
        "response_time": round(
            total_time,
            2
        )
    }