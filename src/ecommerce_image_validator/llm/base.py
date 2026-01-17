"""Base class for LLM reasoners."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel


class ReasoningResult(BaseModel):
    """
    Standard format for LLM reasoning results.
    
    Attributes
    ----------
    quality_score : float
        Overall quality score (0-1)
    verdict : str
        Final decision ('suitable', 'not_suitable', 'uncertain')
    reasoning : str
        Explanation of the decision
    issues_detected : list[str]
        List of quality issues found
    confidence : float
        Confidence in the decision (0-1)
    feature_importance : dict[str, float]
        Relative importance of each feature in the decision
    """
    
    quality_score: float
    verdict: str
    reasoning: str
    issues_detected: list[str]
    confidence: float
    feature_importance: Dict[str, float] = {}


class BaseLLM(ABC):
    """
    Abstract base class for LLM reasoners.
    
    All LLM implementations should inherit from this class and implement
    the reason() method.
    
    Attributes
    ----------
    model_name : str
        Name of the LLM model
    temperature : float
        Sampling temperature
    
    Methods
    -------
    reason(features: Dict[str, Any]) -> ReasoningResult
        Generate reasoning based on extracted features
    """
    
    def __init__(self, model_name: str, temperature: float):
        """
        Initialize the LLM reasoner.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        temperature : float
            Sampling temperature
        """
        self.model_name = model_name
        self.temperature = temperature
    
    @abstractmethod
    def reason(self, features: Dict[str, Any]) -> ReasoningResult:
        """
        Generate reasoning based on extracted features.
        
        Parameters
        ----------
        features : Dict[str, Any]
            Dictionary of extracted features from all extractors
        
        Returns
        -------
        ReasoningResult
            Structured reasoning result
        
        Raises
        ------
        ValueError
            If features are invalid or reasoning fails
        """
        pass
    
    def __repr__(self) -> str:
        """Return string representation of the LLM."""
        return f"{self.__class__.__name__}(model='{self.model_name}')"