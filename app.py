"""Streamlit frontend for E-commerce Image Validator."""

import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.ecommerce_image_validator.pipeline import ImageValidationPipeline
from src.ecommerce_image_validator.utils import make_json_safe

# Page config
st.set_page_config(
    page_title="E-commerce Image Validator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline(llm_type: str = "groq"):
    """
    Load and cache the validation pipeline.
    
    Parameters
    ----------
    llm_type : str
        Which LLM to use: 'groq', 'claude', or 'gemini'
    
    Returns
    -------
    ImageValidationPipeline
        Initialized pipeline
    """
    return ImageValidationPipeline(llm_type=llm_type)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL image to OpenCV format.
    
    Parameters
    ----------
    pil_image : PIL.Image.Image
        Input PIL image
    
    Returns
    -------
    np.ndarray
        Image in BGR format for OpenCV
    """
    # Convert to RGB if not already
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    # Convert to numpy array
    image_rgb = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    return image_bgr


def display_verdict(verdict: str, confidence: float):
    """Display verdict with appropriate styling."""
    if verdict == "suitable":
        st.success(f"✅ **SUITABLE** for e-commerce (Confidence: {confidence:.1%})")
    elif verdict == "not_suitable":
        st.error(f"❌ **NOT SUITABLE** for e-commerce (Confidence: {confidence:.1%})")
    else:
        st.warning(f"⚠️ **UNCERTAIN** - Manual review recommended (Confidence: {confidence:.1%})")


def display_features(features: dict):
    """Display extracted features in organized tabs."""
    tabs = st.tabs(["🔍 Blur Detection", "🎯 Object Detection", "🎨 Background Analysis"])
    
    # Blur Detection
    with tabs[0]:
        blur_data = features.get("blur_detection", {})
        if "error" in blur_data:
            st.error(f"Feature extraction failed: {blur_data['error']}")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Sharpness Score", f"{blur_data.get('sharpness_score', 0):.2f}")
            col2.metric("Laplacian Variance", f"{blur_data.get('variance', 0):.1f}")
            col3.metric("Is Sharp?", "✅ Yes" if blur_data.get("is_sharp", False) else "❌ No")
    
    # Object Detection
    with tabs[1]:
        obj_data = features.get("object_detection", {})
        if "error" in obj_data:
            st.error(f"Feature extraction failed: {obj_data['error']}")
        else:
            st.metric("Objects Detected", obj_data.get("num_objects", 0))
            
            primary = obj_data.get("primary_object")
            if primary:
                st.info(f"**Primary Object:** {primary['class']} ({primary['confidence']:.1%} confidence)")
            
            objects = obj_data.get("objects", [])
            if objects:
                st.write("**Detected Objects:**")
                for obj in objects[:10]:  # Show top 10
                    st.write(f"- {obj['class']}: {obj['confidence']:.1%} confidence")
            else:
                st.write("No objects detected")
    
    # Background Analysis
    with tabs[2]:
        bg_data = features.get("background_analysis", {})
        if "error" in bg_data:
            st.error(f"Feature extraction failed: {bg_data['error']}")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Cleanliness Score", f"{bg_data.get('cleanliness_score', 0):.2f}")
            col2.metric("Edge Density", f"{bg_data.get('edge_density', 0):.3f}")
            col3.metric("Color Variance", f"{bg_data.get('color_variance', 0):.3f}")
            
            if bg_data.get("is_clean", False):
                st.success("✅ Background is clean")
            else:
                st.warning("⚠️ Background may be cluttered")


def main():
    """Main Streamlit app."""
    
    # Header
    st.title("🖼️ E-commerce Image Validator")
    st.markdown("""
    Upload a product image to assess its suitability for professional e-commerce use.
    The system extracts visual features and uses AI reasoning to provide a quality assessment.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool analyzes product images using:
        
        **Feature Extraction:**
        - 🔍 Blur/Sharpness Detection
        - 🎯 Object Detection (YOLOv8)
        - 🎨 Background Quality Analysis
        
        **AI Reasoning:**
        - 🤖 Multiple LLM options
        - Structured quality assessment
        - Explainable decisions
        """)
        
        st.divider()
        
        st.header("⚙️ Model Selection")
        selected_model = st.selectbox(
            "Choose LLM Model",
            ["groq", "claude", "gemini"],
            index=0,
            help="Select which AI model to use for reasoning"
        )
        
        model_info = {
            "groq": "🦙 Llama 3.3 70B (via Groq)\nFast, free, good reasoning",
            "claude": "🧠 Claude 3.5 Sonnet (via AWS)\nExcellent reasoning, costs money",
            "gemini": "✨ Gemini 2.5 Flash\nFree tier available"
        }
        
        st.caption(model_info[selected_model])
        
        st.divider()
        
        st.header("🔄 How it works")
        st.markdown("""
        1. Upload an image
        2. Preprocess the image
        3. System extracts visual features
        4. AI analyzes features
        5. Receive quality verdict + explanation
        """)
        
        compare_all = st.checkbox(
            "🔄 Compare All Models",
            help="Run validation with all 3 LLMs and compare results"
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a product image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a product image to validate"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, use_column_width=True)
        
        with col2:
            st.subheader("📊 Analysis")
            
            # Convert PIL to CV2
            cv2_image = pil_to_cv2(pil_image)
            
            # Check if comparing all models
            if compare_all:
                st.info("🔄 Running comparison with all models...")
                
                models = ["groq", "claude", "gemini"]
                results_dict = {}
                
                for model_name in models:
                    with st.spinner(f"Running {model_name.upper()}..."):
                        try:
                            pipeline = load_pipeline(llm_type=model_name)
                            result = pipeline.validate_from_array(
                                cv2_image,
                                image_name=uploaded_file.name
                            )
                            results_dict[model_name] = result
                        except Exception as e:
                            st.error(f"❌ {model_name.upper()} failed: {str(e)}")
                            results_dict[model_name] = None
                
                # Display comparison table
                st.subheader("🔄 Model Comparison")
                
                comparison_data = []
                for model_name, result in results_dict.items():
                    if result:
                        comparison_data.append({
                            "Model": model_name.upper(),
                            "Verdict": result.verdict,
                            "Score": f"{result.quality_score:.2f}",
                            "Confidence": f"{result.confidence:.2f}",
                            "Time": f"{result.metadata.get('processing_time_seconds', 0):.2f}s"
                        })
                
                st.dataframe(comparison_data, use_container_width=True)
                
                # Check agreement
                verdicts = [r.verdict for r in results_dict.values() if r]
                if len(set(verdicts)) == 1:
                    st.success("✅ All models agree!")
                else:
                    st.warning("⚠️ Models disagree - manual review recommended")
                
                # Use first successful result for detailed display
                result = next((r for r in results_dict.values() if r), None)
                if not result:
                    st.error("All models failed")
                    st.stop()
            
            else:
                # Single model validation
                with st.spinner(f"🔄 Analyzing with {selected_model.upper()}..."):
                    try:
                        pipeline = load_pipeline(llm_type=selected_model)
                        result = pipeline.validate_from_array(
                            cv2_image,
                            image_name=uploaded_file.name
                        )
                    except Exception as e:
                        st.error(f"❌ Validation failed: {str(e)}")
                        st.stop()
            
            # Display verdict
            display_verdict(result.verdict, result.confidence)
            
            # Display quality score
            st.metric("Quality Score", f"{result.quality_score:.2f}/1.00")
            
            # Display processing time
            processing_time = result.metadata.get("processing_time_seconds", 0)
            st.caption(f"⏱️ Processed in {processing_time:.2f} seconds")
        
        # Full width sections below
        st.divider()
        
        # LLM Reasoning
        st.subheader("🧠 AI Reasoning")
        st.info(result.reasoning)
        
        # Issues detected
        if result.issues_detected:
            st.subheader("⚠️ Issues Detected")
            for issue in result.issues_detected:
                st.write(f"- {issue}")
        
        # Feature importance
        st.subheader("📈 Feature Importance")
        importance = result.feature_importance
        if importance:
            cols = st.columns(len(importance))
            for idx, (feature, weight) in enumerate(importance.items()):
                cols[idx].metric(feature.title(), f"{weight:.2f}")
        
        # Extracted features
        st.subheader("🔬 Extracted Features")
        display_features(result.extracted_features)
        
        # Export results
        st.divider()
        st.subheader("💾 Export Results")
        
        # Prepare JSON
        safe_dict = make_json_safe(result.model_dump())
        result_json = json.dumps(safe_dict, indent=2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download JSON",
                data=result_json,
                file_name=f"validation_{uploaded_file.name}.json",
                mime="application/json"
            )
        
        with col2:
            with st.expander("👁️ View JSON"):
                st.json(json.loads(result_json))


if __name__ == "__main__":
    main()