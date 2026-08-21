from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.workspace import Workspace
from app.schemas.workspace import WorkspaceCreate
from app.models.user import User
from app.api.dependencies import get_current_user

router = APIRouter(
    prefix="/api/workspaces",
    tags=["Workspaces"]
)

@router.post("/")
def create_workspace(
    workspace: WorkspaceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    new_workspace = Workspace(
        name=workspace.name,
        docs=0,
        updated="Just now",
        color="bg-emerald-500",
        user_id=current_user.id
    )

    db.add(new_workspace)
    db.commit()
    db.refresh(new_workspace)

    return new_workspace


@router.get("/")
def get_workspaces(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return db.query(Workspace).filter(Workspace.user_id == current_user.id).all()