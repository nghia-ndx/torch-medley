from pydantic import BaseModel


class BaseConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True
