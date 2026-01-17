"""Blur/sharpness detection using Laplacian variance method."""

import cv2
import numpy as np

from ecommerce_image_validator.config import settings
from ecommerce_image_validator.extractors.base import BaseExtractor, ExtractionResult
from ecommerce_image_validator.logger import setup_logger
from ecommerce_image_validator.utils import normalize_score

logger = setup_logger(__name__)


class BlurDetector(BaseExtractor):
    """
    Detect image blur using Laplacian variance method.
    
    The Laplacian operator measures the second derivative of an image,
    highlighting regions of rapid intensity change (edges). A blurry image
    will have low variance in the Laplacian, while a sharp image will have
    high variance.
    
    Parameters
    ----------
    threshold : float, optional
        Variance threshold below which image is considered blurry
        (default: from settings.blur_threshold)
    
    Attributes
    ----------
    threshold : float
        Blur detection threshold
    
    Methods
    -------
    extract(image: np.ndarray) -> ExtractionResult
        Compute blur score for the image
    
    Examples
    --------
    >>> detector = BlurDetector()
    >>> result = detector.extract(image)
    >>> print(result.data['is_sharp'])
    True
    """
    
    def __init__(self, threshold: float | None = None):
        """
        Initialize the blur detector.
        
        Parameters
        ----------
        threshold : float, optional
            Custom threshold (default: from settings)
        """
        super().__init__(name="Blur Detector")
        self.threshold = threshold if threshold is not None else settings.blur_threshold
        logger.info(f"Initialized {self.name} with threshold={self.threshold}")
    
    def extract(self, image: np.ndarray) -> ExtractionResult:
        """
        Compute blur/sharpness score using Laplacian variance.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format
        
        Returns
        -------
        ExtractionResult
            Contains:
            - variance: Laplacian variance (higher = sharper)
            - is_sharp: Boolean indicating if image passes threshold
            - sharpness_score: Normalized score (0-1, higher = sharper)
        
        Raises
        ------
        ValueError
            If image is invalid or empty
        
        Notes
        -----
        The Laplacian variance method is fast and effective for detecting
        blur in product images. However, it can be fooled by:
        - Intentional bokeh/depth-of-field effects
        - High-contrast edges in otherwise blurry images
        - Very low-texture images (e.g., solid colors)
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image: image is None or empty")
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Compute Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Calculate variance
            variance = laplacian.var()
            
            # Determine if sharp
            is_sharp = variance >= self.threshold
            
            # Normalize to 0-1 score (using reasonable max of 500 for product images)
            sharpness_score = normalize_score(variance, min_val=0, max_val=500)
            
            logger.debug(
                f"Blur detection: variance={variance:.2f}, "
                f"is_sharp={is_sharp}, score={sharpness_score:.3f}"
            )
            
            return ExtractionResult(
                feature_name="blur_detection",
                data={
                    "variance": float(variance),
                    "is_sharp": is_sharp,
                    "sharpness_score": sharpness_score,
                    "threshold": self.threshold
                },
                metadata={
                    "method": "laplacian_variance",
                    "image_shape": image.shape
                }
            )
        
        except Exception as e:
            logger.error(f"Blur detection failed: {str(e)}")
            raise ValueError(f"Blur detection failed: {str(e)}") from e