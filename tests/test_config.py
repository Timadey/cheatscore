"""
Test configuration module
"""
import pytest
from app.config import Settings


def test_settings_initialization():
    """Test that settings can be initialized"""
    settings = Settings(
        database_url="postgresql://test:test@localhost/test",
        jwt_secret_key="test-secret-key"
    )
    assert settings.app_name == "sd-proctor"
    assert settings.app_version == "0.1.0"


def test_kafka_servers_list():
    """Test Kafka servers parsing"""
    settings = Settings(
        database_url="postgresql://test:test@localhost/test",
        jwt_secret_key="test-secret-key",
        kafka_bootstrap_servers="server1:9092,server2:9092,server3:9092"
    )
    assert len(settings.kafka_servers_list) == 3
    assert "server1:9092" in settings.kafka_servers_list


def test_gaze_range_parsing():
    """Test gaze range parsing"""
    settings = Settings(
        database_url="postgresql://test:test@localhost/test",
        jwt_secret_key="test-secret-key",
        gaze_pitch_range="-15,25",
        gaze_yaw_range="-30,30"
    )
    assert settings.gaze_pitch_range_tuple == (-15.0, 25.0)
    assert settings.gaze_yaw_range_tuple == (-30.0, 30.0)
