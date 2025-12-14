"""
Admin API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.utils.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stats")
async def get_stats(
    db: AsyncSession = Depends(get_db)
):
    """Get system statistics."""
    # TODO: Implement statistics
    return {
        "active_sessions": 0,
        "total_enrollments": 0,
        "total_alerts": 0
    }


@router.get("/sessions")
async def list_sessions(
    db: AsyncSession = Depends(get_db)
):
    """List active sessions."""
    # TODO: Implement session listing
    return []

