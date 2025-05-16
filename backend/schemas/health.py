from pydantic import BaseModel

class PingResponse(BaseModel):
    status: str
    message: str

    # If a Config class with orm_mode exists, it will be updated.
    # Example:
    # class Config:
    #     orm_mode = True 
    # Should become:
    # class Config:
    #     from_attributes = True 