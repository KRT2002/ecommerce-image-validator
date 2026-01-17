"""Base class for feature extractors."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from pydantic import BaseModel


class ExtractionResult(BaseModel):
    """
    Standard format for extraction results.
    
    Attributes
    ----------
    feature_name : str
        Name of the extracted feature
    data : Dict[str, Any]
        Extracted feature data
    metadata : Dict[str, Any]
        Additional metadata about the extraction process
    """
    
    feature_name: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True


class BaseExtractor(ABC):
    """
    Abstract base class for all feature extractors.
    
    All feature extractors should inherit from this class and implement
    the extract() method.
    
    Attributes
    ----------
    name : str
        Human-readable name of the extractor
    
    Methods
    -------
    extract(image: np.ndarray) -> ExtractionResult
        Extract features from an image
    """
    
    def __init__(self, name: str):
        """
        Initialize the extractor.
        
        Parameters
        ----------
        name : str
            Name of the extractor
        """
        self.name = name
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> ExtractionResult:
        """
        Extract features from an image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (OpenCV standard)
        
        Returns
        -------
        ExtractionResult
            Extracted features and metadata
        
        Raises
        ------
        ValueError
            If image is invalid or extraction fails
        """
        pass
    
    def __repr__(self) -> str:
        """Return string representation of the extractor."""
        return f"{self.__class__.__name__}(name='{self.name}')"