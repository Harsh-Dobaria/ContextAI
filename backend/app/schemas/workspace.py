from pydantic import BaseModel


class WorkspaceCreate(BaseModel):
    name: str