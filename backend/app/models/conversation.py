from sqlalchemy import Column, ForeignKey, Integer, String
from app.database.database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)

    workspace_id = Column(
        Integer,
        ForeignKey("workspaces.id"),
        nullable=False
    )

    title = Column(
        String,
        nullable=True
    )