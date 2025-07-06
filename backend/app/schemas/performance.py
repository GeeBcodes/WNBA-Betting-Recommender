import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel

class PerformanceReportBase(BaseModel):
    report_date: datetime
    segment_type: str
    segment_value: str
    metrics: Dict[str, Any]

class PerformanceReportCreate(PerformanceReportBase):
    pass

class PerformanceReport(PerformanceReportBase):
    id: uuid.UUID
    
    class Config:
        from_attributes = True 