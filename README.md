# E-commerce Image Validator

An intelligent system for assessing the quality of product images for e-commerce use. Combines computer vision feature extraction with LLM reasoning to provide explainable quality assessments.

## 🎯 Features

- **Multi-Feature Extraction:**
  - Blur/sharpness detection (Laplacian variance)
  - Object detection (YOLOv8)
  - Background quality analysis (custom heuristics)

- **AI-Powered Reasoning:**
  - Llama 3.3 70B via Groq for structured reasoning
  - Explainable quality assessments
  - Confidence scoring

- **Interactive UI:**
  - Streamlit web interface
  - Real-time analysis
  - JSON export functionality

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- UV package manager ([install](https://docs.astral.sh/uv/))
- Groq API key (free tier available at [groq.com](https://groq.com))

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/KRT2002/ecommerce-image-validator.git
cd ecommerce-image-validator
```

2. **Install dependencies with UV:**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

3. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

4. **Run the Streamlit app:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
ecommerce-image-validator/
├── src/validator/
│   ├── extractors/          # Feature extraction modules
│   │   ├── base.py          # Base extractor class
│   │   ├── blur_detector.py # Sharpness detection
│   │   ├── object_detector.py # YOLOv8 wrapper
│   │   └── background_analyzer.py # Background quality
│   ├── llm/                 # LLM reasoning
│   │   ├── base.py          # Base LLM class
│   │   ├── groq_reasoner.py # Groq implementation
│   │   └── prompts.py       # Prompt templates
│   ├── config.py            # Configuration
│   ├── logger.py            # Logging setup
│   ├── utils.py             # Utility functions
│   └── pipeline.py          # Main orchestrator
├── app.py                   # Streamlit frontend
├── examples/                # Sample images and outputs
├── pyproject.toml           # UV configuration
└── README.md
```

## 🔧 Usage

### Via Streamlit UI

1. Run `streamlit run app.py`
2. Upload a product image
3. View analysis results, reasoning, and export JSON

### Programmatic Usage

```python
from validator import ImageValidationPipeline

# Initialize pipeline
pipeline = ImageValidationPipeline()

# Validate an image
result = pipeline.validate("path/to/product.jpg")

# Access results
print(f"Verdict: {result.verdict}")
print(f"Quality Score: {result.quality_score:.2f}")
print(f"Reasoning: {result.reasoning}")
print(f"Issues: {result.issues_detected}")

# Export to JSON
with open("result.json", "w") as f:
    f.write(result.model_dump_json(indent=2))
```

## 📊 Example Output

```json
{
  "image_path": "product.jpg",
  "quality_score": 0.78,
  "verdict": "suitable",
  "reasoning": "The image demonstrates good sharpness and a clean background...",
  "issues_detected": [],
  "confidence": 0.85,
  "extracted_features": {
    "blur_detection": {
      "variance": 245.3,
      "is_sharp": true,
      "sharpness_score": 0.82
    },
    "object_detection": {
      "objects": [{"class": "shoe", "confidence": 0.94}],
      "num_objects": 1
    },
    "background_analysis": {
      "cleanliness_score": 0.76,
      "is_clean": true
    }
  },
  "feature_importance": {
    "sharpness": 0.35,
    "background": 0.40,
    "objects": 0.25
  }
}
```

## ⚙️ Configuration

Edit `.env` to customize:

```bash
# Groq API
GROQ_API_KEY=your_key_here
MODEL_NAME=llama-3.3-70b-versatile
TEMPERATURE=0.1

# Feature Thresholds
BLUR_THRESHOLD=100.0
BACKGROUND_CLEANLINESS_THRESHOLD=0.6
MIN_OBJECT_CONFIDENCE=0.5

# Logging
LOG_LEVEL=INFO
```

## 🚧 Limitations

- **Blur detection:** Can be fooled by intentional bokeh or high-contrast edges
- **Object detection:** Limited to 80 COCO classes; may miss niche products
- **Background analysis:** Cannot distinguish foreground from background without segmentation
- **Cultural bias:** "Professional" aesthetics may be culturally biased
- **LLM hallucinations:** May occasionally invent features not present

## 🔮 Future Improvements

- [ ] Add semantic segmentation for precise background isolation
- [ ] Implement OCR for brand/label text extraction
- [ ] Add CLIP embeddings for semantic "professionalism" scoring
- [ ] Support batch processing
- [ ] Add result caching
- [ ] Implement multiple LLM comparison
- [ ] Create evaluation dataset with ground truth labels

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Groq for fast LLM inference
- OpenCV community
- Streamlit team