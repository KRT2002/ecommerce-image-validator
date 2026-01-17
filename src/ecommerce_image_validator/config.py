"""Configuration management using Pydantic settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Attributes
    ----------
    groq_api_key : str
        API key for Groq LLM service
    model_name : str
        Name of the Groq model to use
    temperature : float
        Temperature parameter for LLM sampling
    blur_threshold : float
        Minimum Laplacian variance for sharp images
    background_cleanliness_threshold : float
        Minimum score (0-1) for acceptable background quality
    min_object_confidence : float
        Minimum confidence for object detection
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    
    # LLM Configuration
    groq_api_key: str = Field(..., description="Groq API key")
    model_name: str = Field(default="llama-3.3-70b-versatile", description="Groq model name")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    
    # Feature Extraction Thresholds
    blur_threshold: float = Field(
        default=100.0, 
        ge=0.0, 
        description="Laplacian variance threshold for blur"
    )
    background_cleanliness_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Background quality threshold"
    )
    min_object_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for object detection"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()


# YOLO Model Configuration
YOLO_MODEL_NAME = "yolov8n.pt"  # Nano model for speed
YOLO_DEVICE = "cpu"  # Can be changed to 'cuda' if GPU available

# Image Processing
MAX_IMAGE_SIZE = (1024, 1024)  # Resize large images for faster processing
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}

# Output Structure
OUTPUT_DIR = "examples/outputs"