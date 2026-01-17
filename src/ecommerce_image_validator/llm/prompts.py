"""Prompt templates for LLM reasoning."""

from typing import Any, Dict


def build_reasoning_prompt(features: Dict[str, Any]) -> str:
    """
    Build the reasoning prompt from extracted features.
    
    Parameters
    ----------
    features : Dict[str, Any]
        Dictionary containing all extracted features
    
    Returns
    -------
    str
        Formatted prompt for the LLM
    
    Examples
    --------
    >>> features = {"blur_detection": {...}, "object_detection": {...}}
    >>> prompt = build_reasoning_prompt(features)
    """
    # Extract feature data
    blur_data = features.get("blur_detection", {})
    object_data = features.get("object_detection", {})
    background_data = features.get("background_analysis", {})
    
    # Build feature summary
    feature_summary = _build_feature_summary(blur_data, object_data, background_data)
    
    prompt = f"""You are an expert e-commerce product image quality assessor. Your task is to evaluate whether a product image is suitable for professional e-commerce use based on extracted visual features.

**Extracted Features:**
{feature_summary}

**Task:**
Analyze these features and determine if this image is suitable for professional e-commerce use (e.g., Amazon, eBay, Shopify product listings).

**Evaluation Criteria:**
1. **Sharpness**: Professional images should be sharp and clear
2. **Background**: Should be clean, uncluttered, and not distracting
3. **Objects**: Main product should be clearly visible; avoid unnecessary objects (hands, clutter)
4. **Overall Quality**: Image should look professional and trustworthy

**Output Format:**
You must respond with ONLY a valid JSON object (no additional text) with this exact structure:
{{
    "quality_score": <float between 0 and 1>,
    "verdict": "<'suitable' or 'not_suitable' or 'uncertain'>",
    "reasoning": "<detailed explanation of your decision>",
    "issues_detected": [<list of specific issues found>],
    "confidence": <float between 0 and 1>,
    "feature_importance": {{
        "sharpness": <float 0-1>,
        "background": <float 0-1>,
        "objects": <float 0-1>
    }}
}}

**Guidelines:**
- quality_score: 0.7+ is suitable, 0.4-0.7 is uncertain, <0.4 is not suitable
- verdict: Use 'uncertain' only if features conflict significantly
- reasoning: Be specific about what makes it suitable/unsuitable
- issues_detected: List specific problems (e.g., "blurry image", "cluttered background", "hand visible in frame")
- confidence: How certain are you of this assessment?
- feature_importance: Relative weight (0-1) each feature had in your decision (should sum to ~1.0)

Remember: Return ONLY the JSON object, no preamble or explanation."""

    return prompt


def _build_feature_summary(
    blur_data: Dict[str, Any],
    object_data: Dict[str, Any],
    background_data: Dict[str, Any]
) -> str:
    """
    Build a formatted summary of extracted features.
    
    Parameters
    ----------
    blur_data : Dict[str, Any]
        Blur detection results
    object_data : Dict[str, Any]
        Object detection results
    background_data : Dict[str, Any]
        Background analysis results
    
    Returns
    -------
    str
        Formatted feature summary
    """
    summary_parts = []
    
    # Blur/Sharpness
    if blur_data:
        is_sharp = blur_data.get("is_sharp", False)
        sharpness_score = blur_data.get("sharpness_score", 0)
        variance = blur_data.get("variance", 0)
        
        summary_parts.append(
            f"1. **Sharpness Analysis:**\n"
            f"   - Is Sharp: {is_sharp}\n"
            f"   - Sharpness Score: {sharpness_score:.2f} (0=blurry, 1=very sharp)\n"
            f"   - Laplacian Variance: {variance:.2f}"
        )
    
    # Object Detection
    if object_data:
        objects = object_data.get("objects", [])
        num_objects = object_data.get("num_objects", 0)
        primary = object_data.get("primary_object")
        has_multiple = object_data.get("has_multiple_objects", False)
        
        objects_str = ", ".join([
            f"{obj['class']} ({obj['confidence']:.2f})" 
            for obj in objects[:5]
        ])
        
        summary_parts.append(
            f"2. **Object Detection:**\n"
            f"   - Number of Objects: {num_objects}\n"
            f"   - Primary Object: {primary['class'] if primary else 'None'} "
            f"({primary['confidence']:.2f} confidence)" if primary else "   - Primary Object: None\n"
            f"   - Detected Objects: {objects_str if objects_str else 'None'}\n"
            f"   - Multiple Objects Present: {has_multiple}"
        )
    
    # Background Analysis
    if background_data:
        is_clean = background_data.get("is_clean", False)
        cleanliness = background_data.get("cleanliness_score", 0)
        edge_density = background_data.get("edge_density", 0)
        color_var = background_data.get("color_variance", 0)
        
        summary_parts.append(
            f"3. **Background Analysis:**\n"
            f"   - Is Clean: {is_clean}\n"
            f"   - Cleanliness Score: {cleanliness:.2f} (0=cluttered, 1=very clean)\n"
            f"   - Edge Density: {edge_density:.2f} (higher = more edges/clutter)\n"
            f"   - Color Variance: {color_var:.2f} (higher = more varied colors)"
        )
    
    return "\n\n".join(summary_parts)