import uuid
from datetime import datetime
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.schemas import performance as performance_schema
from backend.db import models as db_models


async def create_performance_report(
    db: AsyncSession, report: performance_schema.PerformanceReportCreate
) -> db_models.PerformanceReport:
    """
    Creates a new performance report entry in the database.
    """
    db_report = db_models.PerformanceReport(
        report_date=report.report_date,
        segment_type=report.segment_type,
        segment_value=report.segment_value,
        metrics=report.metrics,
    )
    db.add(db_report)
    await db.commit()
    await db.refresh(db_report)
    return db_report


async def get_performance_reports(
    db: AsyncSession, skip: int = 0, limit: int = 100
) -> List[db_models.PerformanceReport]:
    """
    Retrieves a list of performance reports.
    """
    stmt = select(db_models.PerformanceReport).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all() 