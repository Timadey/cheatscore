"""
Alert dispatcher for sending alerts to external backends via Kafka.
"""
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.schemas import AlertEvent
from app.config import settings

logger = logging.getLogger(__name__)


class AlertDispatcher:
    """Dispatches alerts to external backends."""
    
    def __init__(self):
        """Initialize alert dispatcher."""
        self.producer: Optional[KafkaProducer] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize Kafka producer."""
        if self.initialized:
            return
        
        try:
            # Create Kafka producer (synchronous, run in thread pool)
            loop = asyncio.get_event_loop()
            self.producer = await loop.run_in_executor(
                self.executor,
                lambda: KafkaProducer(
                    bootstrap_servers=settings.kafka_servers_list,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    acks='all',  # Wait for all replicas
                    retries=settings.kafka_max_retries,
                    max_in_flight_requests_per_connection=1
                    #enable_idempotence=True,
                )
            )
            
            self.initialized = True
            logger.info(f"Alert dispatcher initialized with Kafka: {settings.kafka_alert_topic}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}", exc_info=True)
            raise
    
    async def dispatch_alert(self, alert: AlertEvent) -> bool:
        """
        Dispatch alert to Kafka.
        
        Args:
            alert: Alert event to dispatch
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized:
            await self.initialize()
        
        if self.producer is None:
            logger.error("Kafka producer not initialized")
            return False
        
        try:
            # Serialize alert to dict
            alert_dict = alert.model_dump() if hasattr(alert, 'model_dump') else alert.dict()
            
            # Convert datetime to ISO string
            if isinstance(alert_dict.get('timestamp'), datetime):
                alert_dict['timestamp'] = alert_dict['timestamp'].isoformat()
            
            # Send to Kafka (synchronous operation in thread pool)
            loop = asyncio.get_event_loop()
            future = self.producer.send(
                settings.kafka_alert_topic,
                key=alert.exam_session_id,
                value=alert_dict
            )
            
            # Wait for send to complete
            record_metadata = await loop.run_in_executor(
                self.executor,
                lambda: future.get(timeout=10)
            )
            
            logger.info(
                f"Alert dispatched: {alert.event_id} to topic {record_metadata.topic} "
                f"partition {record_metadata.partition} offset {record_metadata.offset}"
            )
            
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error dispatching alert {alert.event_id}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Error dispatching alert {alert.event_id}: {e}", exc_info=True)
            return False
    
    async def dispatch_batch(self, alerts: list[AlertEvent]) -> Dict[str, bool]:
        """
        Dispatch multiple alerts.
        
        Args:
            alerts: List of alert events
            
        Returns:
            Dictionary mapping alert_id to success status
        """
        results = {}
        
        for alert in alerts:
            success = await self.dispatch_alert(alert)
            results[alert.event_id] = success
        
        return results
    
    async def close(self):
        """Close Kafka producer."""
        if self.producer:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: self.producer.close()
            )
            self.producer = None
            self.initialized = False
            logger.info("Alert dispatcher closed")
        
        self.executor.shutdown(wait=True)

