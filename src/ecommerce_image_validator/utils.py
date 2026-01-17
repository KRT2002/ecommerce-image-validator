"""Utility functions for image processing and validation."""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from typing import Any

from ecommerce_image_validator.config import MAX_IMAGE_SIZE, SUPPORTED_FORMATS
from ecommerce_image_validator.logger import setup_logger

logger = setup_logger(__name__)

def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk.
    
    Parameters
    ----------
    image_path : str or Path
        Path to the image file
    
    Returns
    -------
    np.ndarray
        Loaded image as numpy array (BGR format)
    
    Raises
    ------
    FileNotFoundError
        If image file doesn't exist
    ValueError
        If image format is unsupported or image is corrupted
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {image_path.suffix}. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    try:
        # Load with PIL for better format support
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        
        # Convert to numpy array (RGB)
        image_rgb = np.array(pil_image)
        
        # Convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}") from e

def preprocess_image(image_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Preprocess an image for analysis.
    
    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR format
    
    Returns
    -------
    Tuple[np.ndarray, bool]
        - Preprocessed image as numpy array (BGR format)
        - Boolean indicating if image was resized
    """
    was_resized = False
    
    h, w = image_bgr.shape[:2]
    max_h, max_w = MAX_IMAGE_SIZE
    
    if h > max_h or w > max_w:
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        image_bgr = cv2.resize(
            image_bgr,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )
        
        was_resized = True
        logger.info(f"Resized image from ({w}, {h}) to ({new_w}, {new_h})")
    
    return image_bgr, was_resized


def calculate_image_stats(image: np.ndarray) -> dict:
    """
    Calculate basic statistics about an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (BGR format)
    
    Returns
    -------
    dict
        Dictionary containing image statistics:
        - width: int
        - height: int
        - channels: int
        - mean_brightness: float
        - std_brightness: float
    
    Examples
    --------
    >>> stats = calculate_image_stats(image)
    >>> print(stats['width'], stats['height'])
    1280 720
    """
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    # Convert to grayscale for brightness calculation
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "mean_brightness": float(np.mean(gray)),
        "std_brightness": float(np.std(gray))
    }


def normalize_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize a score to 0-1 range.
    
    Parameters
    ----------
    value : float
        Input value to normalize
    min_val : float, optional
        Minimum expected value (default: 0.0)
    max_val : float, optional
        Maximum expected value (default: 1.0)
    
    Returns
    -------
    float
        Normalized value clamped to [0.0, 1.0]
    
    Examples
    --------
    >>> normalize_score(150, 0, 300)
    0.5
    """
    if max_val == min_val:
        return 0.0
    
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def make_json_safe(obj: Any) -> Any:
    """
    Recursively convert objects into JSON-serializable Python types.
    
    This function traverses nested data structures (dicts, lists) and
    converts non-JSON-serializable objects—such as NumPy scalars and
    arrays—into their native Python equivalents. It is primarily used
    to sanitize model outputs before JSON serialization or persistence.
    
    Parameters
    ----------
    obj : Any
        Input object to be converted. May be a nested combination of
        dictionaries, lists, NumPy arrays, NumPy scalar types, or
        standard Python objects.
    
    Returns
    -------
    Any
        A JSON-serializable version of the input object, where:
        - NumPy scalar types are converted using `.item()`
        - NumPy arrays are converted to Python lists
        - Dictionaries and lists are processed recursively
        - JSON-safe Python types are returned unchanged
    
    Notes
    -----
    This function is intentionally recursive to support deeply nested
    feature extraction outputs and metadata structures commonly produced
    in ML and computer vision pipelines.
    
    Examples
    --------
    >>> import numpy as np
    >>> data = {
    ...     "score": np.float64(0.85),
    ...     "valid": np.bool_(True),
    ...     "values": np.array([1, 2, 3])
    ... }
    >>> make_json_safe(data)
    {'score': 0.85, 'valid': True, 'values': [1, 2, 3]}
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # np.float64, np.bool_, etc.
        return obj.item()
    else:
        return obj