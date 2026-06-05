from pydantic import BaseModel

class PredictionRequest(BaseModel):
    city: str
    neighborhood: str
    property_type: str
    beds: int
    baths: int
    avg_size: float