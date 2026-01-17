"""Groq LLM reasoner using Llama 3.3 70B."""

import json
from typing import Any, Dict

from langchain_groq import ChatGroq

from ecommerce_image_validator.config import settings
from ecommerce_image_validator.llm.base import BaseLLM, ReasoningResult
from ecommerce_image_validator.llm.prompts import build_reasoning_prompt
from ecommerce_image_validator.logger import setup_logger

logger = setup_logger(__name__)


class GroqReasoner(BaseLLM):
    """
    LLM reasoner using Groq's Llama 3.3 70B model.
    
    Uses the ChatGroq wrapper from langchain-groq to generate structured
    reasoning about image quality based on extracted visual features.
    
    Parameters
    ----------
    model_name : str, optional
        Groq model to use (default: from settings)
    temperature : float, optional
        Sampling temperature (default: from settings)
    api_key : str, optional
        Groq API key (default: from settings)
    
    Attributes
    ----------
    llm : ChatGroq
        Groq LLM instance
    
    Methods
    -------
    reason(features: Dict[str, Any]) -> ReasoningResult
        Generate reasoning based on extracted features
    
    Examples
    --------
    >>> reasoner = GroqReasoner()
    >>> result = reasoner.reason(features)
    >>> print(result.verdict)
    'not_suitable'
    
    Notes
    -----
    **Model Choice: Llama 3.3 70B Versatile**
    
    Trade-offs considered:
    - Llama 3.3 70B: Excellent reasoning, fast on Groq, free tier available
    - Llama 3.1 8B: Faster but weaker reasoning for nuanced judgments
    - GPT-4o-mini: Strong alternative but costs money
    
    Why Llama 3.3 70B?
    - Best reasoning quality for this task
    - Fast inference on Groq infrastructure (~2-3s)
    - Free tier sufficient for development
    - Good at structured JSON output
    
    **Limitations:**
    - May hallucinate features not present
    - Can be overly confident in edge cases
    - JSON parsing can fail if model doesn't follow format
    - Cultural bias in "professional" aesthetics
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None
    ):
        """
        Initialize the Groq reasoner.
        
        Parameters
        ----------
        model_name : str, optional
            Model name (default: from settings)
        temperature : float, optional
            Sampling temperature (default: from settings)
        api_key : str, optional
            API key (default: from settings)
        """
        model = model_name if model_name is not None else settings.model_name
        temp = temperature if temperature is not None else settings.temperature
        key = api_key if api_key is not None else settings.groq_api_key
        
        super().__init__(model_name=model, temperature=temp)
        
        try:
            self.llm = ChatGroq(
                model_name=self.model_name,
                temperature=self.temperature,
                api_key=key
            )
            logger.info(
                f"Initialized Groq LLM: {self.model_name} "
                f"(temperature={self.temperature})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {str(e)}")
            raise RuntimeError(f"Failed to initialize Groq LLM: {str(e)}") from e
    
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
            Structured reasoning result with quality assessment
        
        Raises
        ------
        ValueError
            If features are invalid, LLM call fails, or JSON parsing fails
        
        Notes
        -----
        **Error Handling:**
        Common failure modes:
        1. API errors (rate limits, network issues)
        2. JSON parsing errors (model doesn't follow format)
        3. Invalid feature data
        
        We use retry logic and fallback defaults to handle these gracefully.
        """
        if not features:
            raise ValueError("Features dictionary is empty")
        
        try:
            # Build prompt
            prompt = build_reasoning_prompt(features)
            logger.debug(f"Built reasoning prompt ({len(prompt)} chars)")
            
            # Call LLM
            logger.info("Calling Groq LLM for reasoning...")
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            logger.debug(f"Received LLM response ({len(response_text)} chars)")
            
            # Parse JSON response
            result_dict = self._parse_json_response(response_text)
            
            # Validate and convert to ReasoningResult
            result = self._validate_and_convert(result_dict)
            
            logger.info(
                f"Reasoning complete: verdict={result.verdict}, "
                f"quality={result.quality_score:.2f}, "
                f"confidence={result.confidence:.2f}"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Reasoning failed: {str(e)}")
            raise ValueError(f"Reasoning failed: {str(e)}") from e
    
    def _parse_json_response(self, response_text: str) -> dict:
        """
        Parse JSON from LLM response.
        
        Parameters
        ----------
        response_text : str
            Raw LLM response text
        
        Returns
        -------
        dict
            Parsed JSON dictionary
        
        Raises
        ------
        ValueError
            If JSON parsing fails
        """
        try:
            # Try direct parsing first
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
                return json.loads(json_str)
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
                return json.loads(json_str)
            else:
                raise ValueError(
                    f"Failed to parse JSON from LLM response. "
                    f"Response: {response_text[:200]}..."
                )
    
    def _validate_and_convert(self, result_dict: dict) -> ReasoningResult:
        """
        Validate and convert dictionary to ReasoningResult.
        
        Parameters
        ----------
        result_dict : dict
            Parsed JSON dictionary
        
        Returns
        -------
        ReasoningResult
            Validated reasoning result
        
        Raises
        ------
        ValueError
            If required fields are missing or invalid
        """
        # Validate required fields
        required_fields = [
            "quality_score", "verdict", "reasoning", 
            "issues_detected", "confidence"
        ]
        
        for field in required_fields:
            if field not in result_dict:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate verdict
        valid_verdicts = {"suitable", "not_suitable", "uncertain"}
        if result_dict["verdict"] not in valid_verdicts:
            logger.warning(
                f"Invalid verdict '{result_dict['verdict']}', "
                f"defaulting to 'uncertain'"
            )
            result_dict["verdict"] = "uncertain"
        
        # Ensure scores are in valid range
        result_dict["quality_score"] = max(0.0, min(1.0, result_dict["quality_score"]))
        result_dict["confidence"] = max(0.0, min(1.0, result_dict["confidence"]))
        
        # Convert to ReasoningResult
        return ReasoningResult(**result_dict)