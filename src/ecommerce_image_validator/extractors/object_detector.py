"""Object detection using YOLOv8."""

from typing import List

import numpy as np
from ultralytics import YOLO

from ecommerce_image_validator.config import YOLO_DEVICE, YOLO_MODEL_NAME, settings
from ecommerce_image_validator.extractors.base import BaseExtractor, ExtractionResult
from ecommerce_image_validator.logger import setup_logger

logger = setup_logger(__name__)


class ObjectDetector(BaseExtractor):
    """
    Detect objects in images using YOLOv8.
    
    Uses a pretrained YOLOv8 model to detect common objects in product images.
    The model can identify 80 object classes from the COCO dataset.
    
    Parameters
    ----------
    model_name : str, optional
        YOLO model variant (default: 'yolov8n.pt' - fastest)
    device : str, optional
        Device for inference ('cpu' or 'cuda')
    confidence_threshold : float, optional
        Minimum confidence for detections (default: from settings)
    
    Attributes
    ----------
    model : YOLO
        Loaded YOLO model
    confidence_threshold : float
        Minimum detection confidence
    
    Methods
    -------
    extract(image: np.ndarray) -> ExtractionResult
        Detect objects in the image
    
    Examples
    --------
    >>> detector = ObjectDetector()
    >>> result = detector.extract(image)
    >>> print(result.data['objects'])
    [{'class': 'shoe', 'confidence': 0.92}]
    
    Notes
    -----
    YOLOv8n (nano) is chosen for speed on CPU. Trade-offs:
    - YOLOv8n: Fast (~100-200ms on CPU), 80% accuracy
    - YOLOv8s/m: Medium (~300-500ms), 85% accuracy
    - YOLOv8l/x: Slow (~1-2s), 90% accuracy
    
    For product images, YOLOv8n is sufficient since we primarily need
    to know WHAT objects are present, not pixel-perfect localization.
    """
    
    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        device: str = YOLO_DEVICE,
        confidence_threshold: float | None = None
    ):
        """
        Initialize the object detector.
        
        Parameters
        ----------
        model_name : str, optional
            YOLO model variant
        device : str, optional
            Device for inference
        confidence_threshold : float, optional
            Minimum confidence threshold
        """
        super().__init__(name="Object Detector")
        self.device = device
        self.confidence_threshold = (
            confidence_threshold 
            if confidence_threshold is not None 
            else settings.min_object_confidence
        )
        
        try:
            logger.info(f"Loading YOLO model: {model_name} on {device}")
            self.model = YOLO(model_name)
            self.model.to(device)
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}") from e
    
    def extract(self, image: np.ndarray) -> ExtractionResult:
        """
        Detect objects in the image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format
        
        Returns
        -------
        ExtractionResult
            Contains:
            - objects: List of detected objects with class names and confidences
            - num_objects: Total number of detected objects
            - primary_object: Most confident detection (if any)
            - has_multiple_objects: Boolean indicating clutter
        
        Raises
        ------
        ValueError
            If image is invalid or detection fails
        
        Notes
        -----
        Detection failures can occur due to:
        - Very small or very large objects
        - Unusual camera angles or perspectives
        - Objects not in COCO dataset (80 classes)
        - Heavy occlusion or artistic rendering
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image: image is None or empty")
        
        try:
            # Run inference (verbose=False to suppress YOLO logs)
            results = self.model(image, verbose=False)[0]
            
            # Extract detections
            detections: List[dict] = []
            
            for box in results.boxes:
                confidence = float(box.conf[0])
                
                if confidence >= self.confidence_threshold:
                    class_id = int(box.cls[0])
                    class_name = results.names[class_id]
                    
                    detections.append({
                        "class": class_name,
                        "confidence": round(confidence, 3)
                    })
            
            # Sort by confidence
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Determine primary object
            primary_object = detections[0] if detections else None
            
            # Check for clutter (multiple distinct objects)
            has_multiple_objects = len(detections) > 1
            
            logger.debug(
                f"Detected {len(detections)} objects: "
                f"{[d['class'] for d in detections[:3]]}"
            )
            
            return ExtractionResult(
                feature_name="object_detection",
                data={
                    "objects": detections,
                    "num_objects": len(detections),
                    "primary_object": primary_object,
                    "has_multiple_objects": has_multiple_objects
                },
                metadata={
                    "model": YOLO_MODEL_NAME,
                    "device": self.device,
                    "confidence_threshold": self.confidence_threshold,
                    "total_detections": len(results.boxes)
                }
            )
        
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            raise ValueError(f"Object detection failed: {str(e)}") from e