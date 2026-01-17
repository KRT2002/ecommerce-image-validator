"""Gemini LLM reasoner using Google Generative AI."""

import json
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI

from ecommerce_image_validator.config import settings
from ecommerce_image_validator.llm.base import BaseLLM, ReasoningResult
from ecommerce_image_validator.llm.prompts import build_reasoning_prompt
from ecommerce_image_validator.logger import setup_logger

logger = setup_logger(__name__)


class GeminiReasoner(BaseLLM):
    """
    LLM reasoner using Google's Gemini models.
    
    Uses the ChatGoogleGenerativeAI wrapper from langchain to generate
    structured reasoning about image quality based on extracted visual features.
    
    Parameters
    ----------
    model_id : str, optional
        Gemini model to use (default: from settings)
    temperature : float, optional
        Sampling temperature (default: from settings)
    api_key : str, optional
        Google API key (default: from settings)
    
    Attributes
    ----------
    llm : ChatGoogleGenerativeAI
        Gemini LLM instance
    
    Methods
    -------
    reason(features: Dict[str, Any]) -> ReasoningResult
        Generate reasoning based on extracted features
    
    Examples
    --------
    >>> reasoner = GeminiReasoner()
    >>> result = reasoner.reason(features)
    >>> print(result.verdict)
    'suitable'
    
    Notes
    -----
    **Model Choice: Gemini 2.0 Flash**
    
    Trade-offs considered:
    - Gemini 2.0 Flash: Very fast, good reasoning, free tier available
    - Gemini 1.5 Pro: Better reasoning but slower and more expensive
    - Gemini 1.5 Flash: Fast but older generation
    
    Why Gemini 2.0 Flash?
    - Excellent speed (~1-2s response time)
    - Strong reasoning for structured tasks
    - Free tier generous (1500 requests/day)
    - Good at JSON output format
    - Competitive with GPT-4o and Claude Sonnet
    
    **Limitations:**
    - May occasionally produce verbose responses
    - JSON parsing can fail if model doesn't follow format exactly
    - May have different aesthetic biases than Llama/Claude
    - Requires Google API key
    """
    
    def __init__(
        self,
        model_id: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None
    ):
        """
        Initialize the Gemini reasoner.
        
        Parameters
        ----------
        model_id : str, optional
            Model ID (default: from settings)
        temperature : float, optional
            Sampling temperature (default: from settings)
        api_key : str, optional
            Google API key (default: from settings)
        """
        model = model_id if model_id is not None else settings.gemini_model_id
        temp = temperature if temperature is not None else settings.temperature
        key = api_key if api_key is not None else settings.google_api_key
        
        super().__init__(model_name=model, temperature=temp)
        
        if not key:
            raise ValueError(
                "Missing Google API key. Please set GOOGLE_API_KEY in .env file"
            )
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=key
            )
            logger.info(
                f"Initialized Gemini LLM: {self.model_name} "
                f"(temperature={self.temperature})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
            raise RuntimeError(f"Failed to initialize Gemini LLM: {str(e)}") from e
    
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
        1. API errors (rate limits, quota exceeded, network issues)
        2. JSON parsing errors (model doesn't follow format)
        3. Invalid API key
        
        We use retry logic and fallback defaults to handle these gracefully.
        """
        if not features:
            raise ValueError("Features dictionary is empty")
        
        try:
            # Build prompt
            prompt = build_reasoning_prompt(features)
            logger.debug(f"Built reasoning prompt ({len(prompt)} chars)")
            
            # Call Gemini
            logger.info("Calling Google Gemini for reasoning...")
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            logger.debug(f"Received Gemini response ({len(response_text)} chars)")
            
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
        Parse JSON from Gemini response.
        
        Parameters
        ----------
        response_text : str
            Raw Gemini response text
        
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
                    f"Failed to parse JSON from Gemini response. "
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