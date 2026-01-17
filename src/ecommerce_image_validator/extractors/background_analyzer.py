"""Background quality analysis using custom heuristics."""

import cv2
import numpy as np

from ecommerce_image_validator.config import settings
from ecommerce_image_validator.extractors.base import BaseExtractor, ExtractionResult
from ecommerce_image_validator.logger import setup_logger
from ecommerce_image_validator.utils import normalize_score

logger = setup_logger(__name__)


class BackgroundAnalyzer(BaseExtractor):
    """
    Analyze background quality using edge density and color variance.
    
    Professional product images typically have:
    - Clean, uncluttered backgrounds
    - Uniform or simple colors
    - Low edge density (few distracting elements)
    
    This analyzer combines edge detection and color variance to estimate
    background cleanliness without needing object segmentation.
    
    Parameters
    ----------
    threshold : float, optional
        Cleanliness threshold (0-1) for acceptable backgrounds
        (default: from settings.background_cleanliness_threshold)
    
    Attributes
    ----------
    threshold : float
        Background quality threshold
    
    Methods
    -------
    extract(image: np.ndarray) -> ExtractionResult
        Analyze background quality
    
    Examples
    --------
    >>> analyzer = BackgroundAnalyzer()
    >>> result = analyzer.extract(image)
    >>> print(result.data['is_clean'])
    False
    
    Notes
    -----
    This is a heuristic-based approach with limitations:
    - Cannot distinguish foreground from background without segmentation
    - May penalize textured products (e.g., patterned fabrics)
    - Sensitive to image resolution and compression artifacts
    
    A more sophisticated approach would use:
    - Semantic segmentation to isolate background
    - Depth estimation to separate foreground/background
    - ML-based background quality scoring
    
    However, this simple heuristic is fast, interpretable, and effective
    for most e-commerce use cases.
    """
    
    def __init__(self, threshold: float | None = None):
        """
        Initialize the background analyzer.
        
        Parameters
        ----------
        threshold : float, optional
            Custom cleanliness threshold (default: from settings)
        """
        super().__init__(name="Background Analyzer")
        self.threshold = (
            threshold 
            if threshold is not None 
            else settings.background_cleanliness_threshold
        )
        logger.info(f"Initialized {self.name} with threshold={self.threshold}")
    
    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """
        Calculate edge density using Canny edge detection.
        
        Parameters
        ----------
        image : np.ndarray
            Grayscale image
        
        Returns
        -------
        float
            Edge density (ratio of edge pixels to total pixels)
        """
        # Apply Canny edge detection
        edges = cv2.Canny(image, threshold1=50, threshold2=150)
        
        # Calculate ratio of edge pixels
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        
        return edge_pixels / total_pixels
    
    def _calculate_color_variance(self, image: np.ndarray) -> float:
        """
        Calculate color variance in the image.
        
        Parameters
        ----------
        image : np.ndarray
            BGR image
        
        Returns
        -------
        float
            Normalized color variance (0-1, higher = more varied)
        """
        # Convert to LAB color space (perceptually uniform)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate standard deviation for each channel
        l_std = np.std(lab[:, :, 0])
        a_std = np.std(lab[:, :, 1])
        b_std = np.std(lab[:, :, 2])
        
        # Average variance (normalize by typical max std of ~50)
        avg_variance = (l_std + a_std + b_std) / 3
        normalized_variance = normalize_score(avg_variance, min_val=0, max_val=50)
        
        return normalized_variance
    
    def extract(self, image: np.ndarray) -> ExtractionResult:
        """
        Analyze background quality.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format
        
        Returns
        -------
        ExtractionResult
            Contains:
            - edge_density: Ratio of edge pixels (0-1)
            - color_variance: Color variation score (0-1)
            - cleanliness_score: Overall background quality (0-1, higher = cleaner)
            - is_clean: Boolean indicating if background passes threshold
        
        Raises
        ------
        ValueError
            If image is invalid or analysis fails
        
        Notes
        -----
        Cleanliness score is computed as:
        cleanliness = 1 - (0.6 * edge_density + 0.4 * color_variance)
        
        This weights edge density more heavily since cluttered backgrounds
        tend to have many edges, while color variance alone can be high
        even in artistic but clean backgrounds.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image: image is None or empty")
        
        try:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate edge density
            edge_density = self._calculate_edge_density(gray)
            
            # Calculate color variance
            color_variance = self._calculate_color_variance(image)
            
            # Compute cleanliness score (inverse of complexity)
            # Weight edge density more heavily (0.6) than color variance (0.4)
            complexity = (0.6 * edge_density) + (0.4 * color_variance)
            cleanliness_score = 1 - complexity
            
            # Clamp to [0, 1]
            cleanliness_score = max(0.0, min(1.0, cleanliness_score))
            
            # Determine if clean
            is_clean = cleanliness_score >= self.threshold
            
            logger.debug(
                f"Background analysis: edge_density={edge_density:.3f}, "
                f"color_variance={color_variance:.3f}, "
                f"cleanliness={cleanliness_score:.3f}, is_clean={is_clean}"
            )
            
            return ExtractionResult(
                feature_name="background_analysis",
                data={
                    "edge_density": round(edge_density, 3),
                    "color_variance": round(color_variance, 3),
                    "cleanliness_score": round(cleanliness_score, 3),
                    "is_clean": is_clean,
                    "threshold": self.threshold
                },
                metadata={
                    "method": "edge_density_color_variance",
                    "weights": {"edge_density": 0.6, "color_variance": 0.4}
                }
            )
        
        except Exception as e:
            logger.error(f"Background analysis failed: {str(e)}")
            raise ValueError(f"Background analysis failed: {str(e)}") from e