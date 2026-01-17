"""Main pipeline orchestrating feature extraction and LLM reasoning."""

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from pydantic import BaseModel

from ecommerce_image_validator.extractors.background_analyzer import BackgroundAnalyzer
from ecommerce_image_validator.extractors.blur_detector import BlurDetector
from ecommerce_image_validator.extractors.object_detector import ObjectDetector
from ecommerce_image_validator.llm.groq_reasoner import GroqReasoner
from ecommerce_image_validator.logger import setup_logger
from ecommerce_image_validator.utils import calculate_image_stats, load_image, preprocess_image

logger = setup_logger(__name__)


class ValidationResult(BaseModel):
    """
    Complete validation result for an image.
    
    Attributes
    ----------
    image_path : str
        Path to the validated image
    quality_score : float
        Overall quality score (0-1)
    verdict : str
        Final verdict ('suitable', 'not_suitable', 'uncertain')
    reasoning : str
        LLM's explanation
    issues_detected : list[str]
        List of quality issues
    confidence : float
        Confidence in the assessment (0-1)
    extracted_features : dict
        All extracted features
    feature_importance : dict
        Importance weights assigned by LLM
    metadata : dict
        Additional metadata (timing, image stats, etc.)
    """
    
    image_path: str
    quality_score: float
    verdict: str
    reasoning: str
    issues_detected: list[str]
    confidence: float
    extracted_features: Dict[str, Any]
    feature_importance: Dict[str, float]
    metadata: Dict[str, Any]
    
    class Config:
        arbitrary_types_allowed = True


class ImageValidationPipeline:
    """
    Main pipeline for validating e-commerce product images.
    
    Orchestrates:
    1. Image loading and preprocessing
    2. Feature extraction (blur, objects, background)
    3. LLM reasoning over features
    4. Result aggregation and formatting
    
    Attributes
    ----------
    blur_detector : BlurDetector
        Blur/sharpness detection extractor
    object_detector : ObjectDetector
        Object detection extractor
    background_analyzer : BackgroundAnalyzer
        Background quality extractor
    llm_reasoner : GroqReasoner
        LLM for reasoning over features
    
    Methods
    -------
    validate(image_path: str | Path) -> ValidationResult
        Run full validation pipeline on an image
    validate_from_array(image: np.ndarray, image_name: str) -> ValidationResult
        Run validation on a pre-loaded image array
    
    Examples
    --------
    >>> pipeline = ImageValidationPipeline()
    >>> result = pipeline.validate("product.jpg")
    >>> print(result.verdict)
    'suitable'
    
    Notes
    -----
    **Design Decisions:**
    
    1. **Sequential Execution:**
       Features are extracted sequentially rather than in parallel.
       - Pro: Simpler error handling, easier debugging
       - Con: Slower total time (~2-3s vs potential ~1s with parallelization)
       - Justification: For this task, simplicity > speed
    
    2. **Error Handling Strategy:**
       - Individual extractor failures don't crash the pipeline
       - LLM can still reason over partial features
       - Failures are logged and included in metadata
       - Justification: Robustness is critical for production use
    
    3. **Feature Aggregation:**
       - All features stored in flat dictionary for LLM
       - No pre-filtering or feature selection
       - LLM decides feature importance
       - Justification: Lets LLM use all available information
    
    **Scalability Considerations:**
    
    For production at scale, we would:
    - Add batch processing support
    - Implement async/parallel feature extraction
    - Add result caching (hash-based)
    - Use model quantization for faster inference
    - Add request queuing and rate limiting
    """
    
    def __init__(self):
        """Initialize the validation pipeline with all components."""
        logger.info("Initializing Image Validation Pipeline...")
        
        try:
            # Initialize extractors
            self.blur_detector = BlurDetector()
            self.object_detector = ObjectDetector()
            self.background_analyzer = BackgroundAnalyzer()
            
            # Initialize LLM reasoner
            self.llm_reasoner = GroqReasoner()
            
            logger.info("Pipeline initialization complete")
        
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {str(e)}")
            raise RuntimeError(f"Pipeline initialization failed: {str(e)}") from e
    
    def validate(self, image_path: str | Path) -> ValidationResult:
        """
        Run full validation pipeline on an image file.
        
        Parameters
        ----------
        image_path : str or Path
            Path to the image file to validate
        
        Returns
        -------
        ValidationResult
            Complete validation result with all features and reasoning
        
        Raises
        ------
        FileNotFoundError
            If image file doesn't exist
        ValueError
            If image is invalid or processing fails
        RuntimeError
            If critical pipeline components fail
        
        Examples
        --------
        >>> pipeline = ImageValidationPipeline()
        >>> result = pipeline.validate("examples/images/product.jpg")
        >>> if result.verdict == "suitable":
        ...     print("Image is good for e-commerce!")
        """
        start_time = time.time()
        image_path = Path(image_path)
        
        logger.info(f"Starting validation for: {image_path.name}")
        
        # Load image
        image = load_image(image_path)
        
        # Run validation on loaded image
        result = self.validate_from_array(image, image_name=image_path.name)
        
        total_time = time.time() - start_time
        result.metadata["total_time_seconds"] = round(total_time, 2)
        
        logger.info(
            f"Validation complete for {image_path.name}: "
            f"verdict={result.verdict}, time={total_time:.2f}s"
        )
        
        return result
    
    def validate_from_array(
        self, 
        image: np.ndarray, 
        image_name: str = "unknown"
    ) -> ValidationResult:
        """
        Run validation pipeline on a pre-loaded image array.
        
        Useful for Streamlit where image is already loaded from upload.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format
        image_name : str, optional
            Name/identifier for the image (default: "unknown")
        
        Returns
        -------
        ValidationResult
            Complete validation result
        
        Raises
        ------
        ValueError
            If image is invalid or processing fails
        """
        start_time = time.time()

        # preprocess image
        image, was_resized = preprocess_image(image)
        image_stats = calculate_image_stats(image)
        
        # Extract features
        features = self._extract_features(image)
        
        # LLM reasoning
        reasoning_result = self._run_llm_reasoning(features)
        
        # Aggregate results
        total_time = time.time() - start_time
        
        result = ValidationResult(
            image_path=image_name,
            quality_score=reasoning_result.quality_score,
            verdict=reasoning_result.verdict,
            reasoning=reasoning_result.reasoning,
            issues_detected=reasoning_result.issues_detected,
            confidence=reasoning_result.confidence,
            extracted_features=features,
            feature_importance=reasoning_result.feature_importance,
            metadata={
                "processing_time_seconds": round(total_time, 2),
                "extractor_count": len(features),
                "llm_model": self.llm_reasoner.model_name,
                "image_stats": image_stats,
                "was_resized": was_resized
            }
        )
        
        return result
    
    def _extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract all features from the image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format
        
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping feature names to feature data
        
        Notes
        -----
        Failures in individual extractors are caught and logged,
        but don't crash the entire pipeline. This allows LLM to
        reason over partial features if some extractors fail.
        """
        features = {}
        extractors = [
            ("blur_detection", self.blur_detector),
            ("object_detection", self.object_detector),
            ("background_analysis", self.background_analyzer)
        ]
        
        for feature_name, extractor in extractors:
            try:
                logger.debug(f"Running {extractor.name}...")
                extraction_start = time.time()
                
                result = extractor.extract(image)
                
                extraction_time = time.time() - extraction_start
                logger.debug(
                    f"{extractor.name} completed in {extraction_time:.3f}s"
                )
                
                # Store feature data
                features[feature_name] = result.data
                
                # Add timing to metadata
                if "extraction_times" not in features:
                    features["extraction_times"] = {}
                features["extraction_times"][feature_name] = round(extraction_time, 3)
            
            except Exception as e:
                logger.error(f"{extractor.name} failed: {str(e)}")
                # Store error info but continue with other extractors
                features[feature_name] = {
                    "error": str(e),
                    "extractor_failed": True
                }
        
        return features
    
    def _run_llm_reasoning(self, features: Dict[str, Any]) -> Any:
        """
        Run LLM reasoning over extracted features.
        
        Parameters
        ----------
        features : Dict[str, Any]
            Extracted features
        
        Returns
        -------
        ReasoningResult
            LLM reasoning result
        
        Raises
        ------
        ValueError
            If LLM reasoning fails critically
        """
        try:
            logger.debug("Running LLM reasoning...")
            llm_start = time.time()
            
            result = self.llm_reasoner.reason(features)
            
            llm_time = time.time() - llm_start
            logger.debug(f"LLM reasoning completed in {llm_time:.2f}s")
            
            # Add LLM timing to features
            features["llm_reasoning_time"] = round(llm_time, 2)
            
            return result
        
        except Exception as e:
            logger.error(f"LLM reasoning failed: {str(e)}")
            raise ValueError(f"LLM reasoning failed: {str(e)}") from e