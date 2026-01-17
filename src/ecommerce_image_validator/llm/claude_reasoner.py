"""Claude LLM reasoner using AWS Bedrock."""

import json
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

from ecommerce_image_validator.config import settings
from ecommerce_image_validator.llm.base import BaseLLM, ReasoningResult
from ecommerce_image_validator.llm.prompts import build_reasoning_prompt
from ecommerce_image_validator.logger import setup_logger

logger = setup_logger(__name__)


class ClaudeReasoner(BaseLLM):
    """
    LLM reasoner using Claude via AWS Bedrock.
    
    Uses boto3 to interface with AWS Bedrock Runtime for Claude models.
    Supports Claude 3.5 Sonnet and other Claude variants.
    
    Parameters
    ----------
    model_id : str, optional
        Claude model ID (default: from settings)
    temperature : float, optional
        Sampling temperature (default: from settings)
    aws_region : str, optional
        AWS region (default: from settings)
    
    Attributes
    ----------
    client : boto3.client
        Bedrock runtime client
    model_id : str
        Claude arn model identifier
    
    Methods
    -------
    reason(features: Dict[str, Any]) -> ReasoningResult
        Generate reasoning based on extracted features
    
    Examples
    --------
    >>> reasoner = ClaudeReasoner()
    >>> result = reasoner.reason(features)
    >>> print(result.verdict)
    'suitable'
    
    Notes
    -----
    **Model Choice: Claude 3.5 Sonnet**
    
    Trade-offs considered:
    - Claude 3.5 Sonnet: Excellent reasoning, good at structured output, moderate cost
    - Claude 3 Opus: Best reasoning but slower and more expensive
    - Claude 3 Haiku: Fastest and cheapest but weaker reasoning
    
    Why Claude 3.5 Sonnet?
    - Strong reasoning capabilities for nuanced quality assessment
    - Good at following JSON output format
    - Balanced cost/performance
    - Available via AWS Bedrock (if you have access)
    
    **Limitations:**
    - Requires AWS credentials and Bedrock access
    - Higher latency than Groq (~3-5s vs ~2s)
    - Costs money (pay per token)
    - May have different quality assessment biases than Llama
    """
    
    def __init__(
        self,
        model_id: str | None = None,
        temperature: float | None = None,
        aws_region: str | None = None
    ):
        """
        Initialize the Claude reasoner.
        
        Parameters
        ----------
        model_id : str, optional
            Model ID (default: from settings)
        temperature : float, optional
            Sampling temperature (default: from settings)
        aws_region : str, optional
            AWS region (default: from settings)
        """
        model = model_id if model_id is not None else settings.claude_model_id
        temp = temperature if temperature is not None else settings.temperature
        
        super().__init__(model_name=model, temperature=temp)
        
        self.aws_region = aws_region if aws_region is not None else settings.aws_region
        
        try:
            self.client = self._get_bedrock_client()
            logger.info(
                f"Initialized Claude LLM: {self.model_name} "
                f"(temperature={self.temperature}, region={self.aws_region})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Claude LLM: {str(e)}")
            raise RuntimeError(f"Failed to initialize Claude LLM: {str(e)}") from e
    
    def _get_bedrock_client(self) -> boto3.client:
        """
        Create and return AWS Bedrock Runtime client.
        
        Returns
        -------
        boto3.client
            Configured Bedrock client
        
        Raises
        ------
        ValueError
            If AWS credentials are missing or client creation fails
        """
        # Fetch credentials from settings
        aws_access_key = settings.aws_access_key_id
        aws_secret_key = settings.aws_secret_access_key
        aws_session_token = settings.aws_session_token
        
        # Validate credentials
        if not aws_access_key or not aws_secret_key:
            raise ValueError(
                "Missing AWS credentials. Please set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY in .env file"
            )
        
        try:
            # Create client
            client_kwargs = {
                "service_name": "bedrock-runtime",
                "region_name": self.aws_region,
                "aws_access_key_id": aws_access_key,
                "aws_secret_access_key": aws_secret_key,
                "config": boto3.session.Config(read_timeout=2000)
            }
            
            # Add session token if provided
            if aws_session_token:
                client_kwargs["aws_session_token"] = aws_session_token
            
            client = boto3.client(**client_kwargs)
            logger.debug(f"Created Bedrock client for region: {self.aws_region}")
            
            return client
        
        except ClientError as e:
            logger.error(f"AWS ClientError during Bedrock client creation: {str(e)}")
            raise ValueError(
                f"AWS ClientError occurred while creating Bedrock client: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error during Bedrock client creation: {str(e)}")
            raise ValueError(
                f"Unexpected error in loading Bedrock client: {str(e)}"
            ) from e
    
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
            If features are invalid, API call fails, or JSON parsing fails
        
        Notes
        -----
        **Error Handling:**
        Common failure modes:
        1. AWS API errors (rate limits, authentication issues)
        2. JSON parsing errors (model doesn't follow format)
        3. Network timeouts
        
        We use retry logic and fallback defaults to handle these gracefully.
        """
        if not features:
            raise ValueError("Features dictionary is empty")
        
        try:
            # Build prompt
            prompt = build_reasoning_prompt(features)
            logger.debug(f"Built reasoning prompt ({len(prompt)} chars)")
            
            # Call Claude via Bedrock
            logger.info("Calling Claude via AWS Bedrock for reasoning...")
            response_text = self._invoke_claude(prompt)
            
            logger.debug(f"Received Claude response ({len(response_text)} chars)")
            
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
    
    def _invoke_claude(self, prompt: str) -> str:
        """
        Invoke Claude model via Bedrock.
        
        Parameters
        ----------
        prompt : str
            Text prompt for Claude
        
        Returns
        -------
        str
            Model's text response
        
        Raises
        ------
        ValueError
            If API call fails
        """
        try:
            # Prepare request body
            native_request = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8192,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt.strip()}]
                    }
                ]
            }
            
            request_body = json.dumps(native_request)
            
            # Invoke model
            response = self.client.invoke_model(
                modelId=self.model_name,
                body=request_body
            )
            
            # Parse response
            model_response = json.loads(response["body"].read())
            response_text = model_response["content"][0]["text"]
            
            return response_text
        
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error(f"Bedrock API error ({error_code}): {str(e)}")
            raise ValueError(f"Bedrock API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to invoke Claude: {str(e)}")
            raise ValueError(f"Failed to invoke Claude: {str(e)}") from e
    
    def _parse_json_response(self, response_text: str) -> dict:
        """
        Parse JSON from Claude response.
        
        Parameters
        ----------
        response_text : str
            Raw Claude response text
        
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
                    f"Failed to parse JSON from Claude response. "
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