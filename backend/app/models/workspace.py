from sqlalchemy import ForeignKey, String, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    docs: Mapped[int] = mapped_column(default=0)
    updated: Mapped[str] = mapped_column(default="Just now")
    color: Mapped[str] = mapped_column(default="bg-emerald-500")
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)