"""
FastAPI application entry point for SD Proctor service.
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from app.config import settings
from app.api.v1 import enrollment, verification, session, admin, webrtc
from app.prediction.live_predictor import LiveProctoringMonitor
from app.utils.redis_client import close_redis
# from app.alerts.datachannel_dispatcher import AlertDispatcher
# Include routers

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Model device: {settings.ai_model_device}")

    # Initialize database tables
    try:
        from app.utils.db import init_db
        await init_db()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        logger.warning("Application will continue but database operations may fail")

    # Initialize AI models
    try:
        from app.inference.face_model_manager import FaceModelManager
        # blocking call to load models
        FaceModelManager.get_instance().initialize()

        # Load live proctoring model
        logger.info("AI models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {e}")
        # Depending on requirements, we might want to raise here

    
    # Initialize alert dispatcher
    # alert_dispatcher = AlertDispatcher()
    # try:
    #     await alert_dispatcher.initialize()
    #     app.state.alert_dispatcher = alert_dispatcher
    #     logger.info("Alert dispatcher initialized")
    # except Exception as e:
    #     logger.warning(f"Alert dispatcher initialization failed: {e}")
    
    yield
    
    # Shutdown
    if hasattr(app.state, 'alert_dispatcher'):
        await app.state.alert_dispatcher.close()
    await close_redis()
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered proctoring backend service",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(enrollment.router, prefix="/api/v1/enroll", tags=["enrollment"])
app.include_router(verification.router, prefix="/api/v1/verify", tags=["verification"])
app.include_router(session.router, prefix="/api/v1/session", tags=["session"])
app.include_router(webrtc.router, prefix="/api/v1/signaling", tags=["signaling"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])


@app.get("/.well-known/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/.well-known/health",
    }

@app.get("/test")
async def read_index():
    from fastapi.responses import HTMLResponse
    file_path = Path(__file__).parent / "frontend_test.html"
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Replace placeholder with actual setting
    content = content.replace("{{APP_URL}}", settings.app_url)
    
    return HTMLResponse(content=content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )

