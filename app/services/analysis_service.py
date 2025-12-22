"""
Analysis service: performs on-demand and final session analysis using cached frames
in Redis and the ExamSessionAnalyzer. Persists final analysis to DB and dumps frames.
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.prediction.session_analyzer import ExamSessionAnalyzer
from app.services.session_service import SessionService
from app.utils.frame_buffer import FrameBuffer
from app.models import ExamAnalysis

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service to run analysis on exam sessions and persist results."""

    def __init__(self, model_path: str = "app/prediction/models", model_type: str = "lstm"):
        self.model_path = model_path
        self.model_type = model_type
        # Analyzer is lightweight to create; create on demand

    async def analyze_session(self, exam_session_id: str, db: Optional[AsyncSession] = None,
                              aggregation_method: str = 'max') -> Dict[str, Any]:
        """
        Perform analysis for a session. If session is active, analysis is done on
        frames in Redis. If not active and a DB record exists, return persisted analysis.
        """
        session_service = SessionService()
        frame_buffer = FrameBuffer()

        try:
            is_active = await session_service.is_session_active(exam_session_id)

            if not is_active and db is not None:
                # Try to load persisted analysis
                q = select(ExamAnalysis).where(ExamAnalysis.exam_session_id == exam_session_id).order_by(ExamAnalysis.analysis_timestamp.desc()).limit(1)
                res = await db.execute(q)
                record = res.scalar_one_or_none()
                if record is not None:
                    logger.info(f"Returning persisted analysis for {exam_session_id}")
                    return record.analysis

            # Fallback to real-time frames in Redis
            metadata_frames = await frame_buffer.get_metadata_frames(exam_session_id, limit=100000)

            # metadata_frames is a list of dicts; Analyzer expects list of frame dicts
            analyzer = ExamSessionAnalyzer(model_path=self.model_path, model_type=self.model_type)

            report = analyzer.analyze_session(metadata_frames, session_id=exam_session_id,
                                              aggregation_method=aggregation_method)

            return report
        finally:
            await frame_buffer.close()

    async def finalize_session_analysis(self, exam_session_id: str, db: AsyncSession,
                                         aggregation_method: str = 'max') -> Dict[str, Any]:
        """
        Run final analysis for a session, persist to DB, dump metadata frames to file and
        clear Redis buffer for the session.
        """
        session_service = SessionService()
        frame_buffer = FrameBuffer()

        try:
            # Attempt to retrieve candidate id
            candidate_id = await session_service.get_candidate_id(exam_session_id)

            metadata_frames = await frame_buffer.get_metadata_frames(exam_session_id, limit=100000)

            analyzer = ExamSessionAnalyzer(model_path=self.model_path, model_type=self.model_type)
            report = analyzer.analyze_session(metadata_frames, session_id=exam_session_id)

            # Persist to DB
            try:
                analysis_record = ExamAnalysis(
                    exam_session_id=exam_session_id,
                    candidate_id=candidate_id,
                    analysis=report,
                    is_cheating=str(report.get('verdict', {}).get('is_cheating', False)),
                    confidence=report.get('verdict', {}).get('confidence', 0.0),
                    verdict_summary=report.get('verdict', {}).get('verdict_basis', '')[:255],
                    analysis_timestamp=datetime.utcnow()
                )
                db.add(analysis_record)
                await db.commit()
                await db.refresh(analysis_record)
                logger.info(f"Persisted analysis for {exam_session_id} id={analysis_record.id}")
            except Exception as e:
                logger.error(f"Failed to persist analysis for {exam_session_id}: {e}", exc_info=True)

            # Dump frames to file for archival
            try:
                dump_path = await frame_buffer.dump_metadata_to_file(exam_session_id)
                logger.info(f"Dumped frames to {dump_path}")
            except Exception as e:
                logger.error(f"Failed to dump metadata frames for {exam_session_id}: {e}", exc_info=True)

            # Clear redis frames
            try:
                deleted = await frame_buffer.clear_session_frames(exam_session_id)
                logger.info(f"Cleared {deleted} frames from redis for {exam_session_id}")
            except Exception as e:
                logger.error(f"Failed to clear frames for {exam_session_id}: {e}", exc_info=True)

            return report
        finally:
            await frame_buffer.close()

