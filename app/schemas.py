from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class CourierInput(BaseModel):
    orders: int = Field(..., ge=0, le=1_000_000)
    couriers: int = Field(..., ge=0, le=100_000)

    @field_validator('orders', 'couriers', mode='before')
    @classmethod
    def coerce_to_int(cls, v):
        return int(v) if isinstance(v, str) else v

    def to_tuple(self) -> Tuple[int, int]:
        return (self.orders, self.couriers)


class PredictionOutput(BaseModel):
    deficit: int = Field(..., ge=0)
    orders: Optional[int] = None
    couriers: Optional[int] = None


class BatchInput(BaseModel):
    items: List[CourierInput] = Field(..., min_length=1, max_length=100_000)


class BatchOutput(BaseModel):
    results: List[PredictionOutput]
    total: int
    elapsed_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None