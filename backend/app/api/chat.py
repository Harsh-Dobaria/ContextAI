from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.services.agents.orchestrator import run_rag_pipeline
from app.services.query_reformulation_service import reformulate_query

from app.database.database import get_db
from app.models.user import User
from app.models.workspace import Workspace
from app.models.conversation import Conversation
from app.models.message import Message

from app.api.dependencies import get_current_user


router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"]
)


@router.get("/")
def chat(
    query: str,
    workspace_id: int,
    conversation_id: int | None = None,
    retrieval_mode: str = "fast",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if retrieval_mode not in ["fast", "accurate"]:
        retrieval_mode = "fast"
    # -----------------------------
    # Verify workspace ownership
    # -----------------------------

    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.id == workspace_id
        )
        .first()
    )

    if (
        not workspace
        or workspace.user_id != current_user.id
    ):
        raise HTTPException(
            status_code=403,
            detail="Workspace not found or access denied"
        )

    # -----------------------------
    # Create or load conversation
    # -----------------------------

    if conversation_id is None:

        conversation = Conversation(
            workspace_id=workspace_id,
            title=query[:100]
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    else:

        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.workspace_id == workspace_id
            )
            .first()
        )

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )

    # -----------------------------
    # Load recent conversation
    # -----------------------------

    previous_messages = (
        db.query(Message)
        .filter(
            Message.conversation_id
            == conversation.id
        )
        .order_by(
            Message.id.desc()
        )
        .limit(6)
        .all()
    )

    previous_messages.reverse()

    previous_question = None
    previous_answer = None

    for message in reversed(previous_messages):

        if (
            message.role == "assistant"
            and previous_answer is None
        ):
            previous_answer = message.content

        elif (
            message.role == "user"
            and previous_question is None
        ):
            previous_question = message.content

        if previous_question and previous_answer:
            break
    # -----------------------------
    # Query reformulation
    # -----------------------------

    reformulated_query = reformulate_query(
        question=query,
        previous_question=previous_question,
        previous_answer=previous_answer
    )

    print(
        f"Original query: {query}"
    )

    print(
        f"Retrieval query: "
        f"{reformulated_query}"
    )

    # -----------------------------
    # Run RAG pipeline
    # -----------------------------

    result = run_rag_pipeline(
        question=reformulated_query,
        workspace_id=workspace_id,
        retrieval_mode=retrieval_mode
    )

    # -----------------------------
    # Save user message
    # -----------------------------

    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=query
    )

    db.add(user_message)

    # -----------------------------
    # Save assistant message
    # -----------------------------

    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=result["answer"]
    )

    db.add(assistant_message)

    db.commit()

    # -----------------------------
    # Return response
    # -----------------------------

    result["conversation_id"] = conversation.id
    result["original_question"] = query
    result["retrieval_question"] = reformulated_query
    result["retrieval_mode"] = retrieval_mode

    return result