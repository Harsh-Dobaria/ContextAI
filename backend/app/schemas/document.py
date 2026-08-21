from pydantic import BaseModel


class DocumentCreate(BaseModel):
    name: str
    file_path: str
    workspace_id: int