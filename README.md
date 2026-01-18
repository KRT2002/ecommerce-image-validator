# E-commerce Image Validator

An intelligent system for assessing the quality of product images for e-commerce use. Combines computer vision feature extraction with multiple LLM reasoners to provide explainable quality assessments.

## 🎯 Features

- **Multi-Feature Extraction:**
  - Blur/sharpness detection (Laplacian variance)
  - Object detection (YOLOv8)
  - Background quality analysis (custom heuristics)

- **Multi-Model AI Reasoning:**
  - 🦙 **Llama 3.3 70B** (via Groq) - Fast, free, excellent reasoning
  - 🧠 **Claude 3.5 Sonnet** (via AWS Bedrock) - Premium reasoning, structured output
  - ✨ **Gemini 2.5 Flash** (via Google) - free tier available
  - Compare outputs across all models

- **Evaluation & Analysis:**
  - Ground truth evaluation with metrics (accuracy, precision, recall, F1)
  - Multi-model comparison script
  - Confidence scoring and uncertainty handling

- **Interactive UI:**
  - Streamlit web interface
  - Model selection dropdown
  - Multi-model comparison mode
  - Real-time analysis
  - JSON export functionality

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- UV package manager ([install](https://docs.astral.sh/uv/))
- At least one of the following API keys:
  - Groq API key (free tier at [groq.com](https://groq.com))
  - AWS credentials with Bedrock access (for Claude)
  - Google API key (free tier at [ai.google.dev](https://ai.google.dev))

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
# Edit .env and add your API keys (at minimum GROQ_API_KEY)
```

**Minimum .env setup (Groq only):**
```bash
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
TEMPERATURE=0.1
```

**Full .env setup (all models):**
```bash
# Groq
GROQ_API_KEY=your_groq_key

# AWS Bedrock (for Claude)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
CLAUDE_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

# Google (for Gemini)
GOOGLE_API_KEY=your_google_key
GEMINI_MODEL_ID=gemini-2.0-flash-exp
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
│   │   ├── groq_reasoner.py # Llama 3.3 via Groq
│   │   ├── claude_reasoner.py # Claude via AWS Bedrock
│   │   ├── gemini_reasoner.py # Gemini via Google
│   │   └── prompts.py       # Prompt templates
│   ├── config.py            # Configuration
│   ├── logger.py            # Logging setup
│   ├── utils.py             # Utility functions
│   └── pipeline.py          # Main orchestrator
├── scripts/                 # Utility scripts
│   ├── compare_models.py    # Multi-model comparison
│   └── evaluate.py          # Evaluation with metrics
├── app.py                   # Streamlit frontend
├── examples/
│   ├── images/              # Sample images
│   ├── evaluation/          # Test dataset
│   │   ├── good/            # Suitable images (ground truth)
│   │   └── bad/             # Unsuitable images (ground truth)
│   └── outputs/
│       └── comparison_results/ # Saved comparison JSONs
├── pyproject.toml           # UV configuration
└── README.md
```

## 🔧 Usage

### 1. Via Streamlit UI (Recommended)

```bash
streamlit run app.py
```

**Features:**
- Upload product images
- Select which LLM model to use (Groq/Claude/Gemini)
- Enable "Compare All Models" to run all 3 LLMs
- View detailed analysis and export results

### 2. Programmatic Usage

**Single model:**
```python
from validator import ImageValidationPipeline

# Initialize with specific model
pipeline = ImageValidationPipeline(llm_type="groq")  # or "claude" or "gemini"

# Validate an image
result = pipeline.validate("path/to/product.jpg")

# Access results
print(f"Verdict: {result.verdict}")
print(f"Quality Score: {result.quality_score:.2f}")
print(f"Reasoning: {result.reasoning}")
print(f"Issues: {result.issues_detected}")
```

### 3. Multi-Model Comparison Script

Compare all models on a single image:

```bash
# Compare all models (default)
python scripts/compare_models.py --image examples/images/product.jpg

# Compare specific models only
python scripts/compare_models.py --image product.jpg --models groq,claude

# Save results and show detailed reasoning
python scripts/compare_models.py --image product.jpg --output results.json --detailed
```

**Example output:**
```
╔══════════════════╦═══════════════╦═══════╦════════════╦═══════╗
║ Model            ║ Verdict       ║ Score ║ Confidence ║ Time  ║
╠══════════════════╬═══════════════╬═══════╬════════════╬═══════╣
║ GROQ             ║ ✅ suitable   ║ 0.78  ║ 0.85       ║ 2.34s ║
║ CLAUDE           ║ ✅ suitable   ║ 0.82  ║ 0.90       ║ 3.12s ║
║ GEMINI           ║ ⚠️  uncertain ║ 0.65  ║ 0.70       ║ 1.89s ║
╚══════════════════╩═══════════════╩═══════╩════════════╩═══════╝

✅ AGREEMENT: All models agree!
   Consensus verdict: suitable
```

### 4. Evaluation Script

Evaluate system accuracy on labeled dataset:

**Setup:**
```bash
# Create evaluation dataset
mkdir -p examples/evaluation/good examples/evaluation/bad

# Add images:
# - examples/evaluation/good/ → suitable product images
# - examples/evaluation/bad/ → unsuitable product images
```

**Run evaluation:**
```bash
# Evaluate single model
python scripts/evaluate.py --model groq

# Compare all models
python scripts/evaluate.py --all-models

# Custom evaluation directory
python scripts/evaluate.py --eval-dir my_dataset/ --model claude
```

**Example output:**
```
================================================================================
EVALUATION RESULTS: GROQ
================================================================================

Total Images: 20
Accuracy: 85.0% (17/20)
Precision: 88.9%
Recall: 80.0%
F1-Score: 84.2%

CONFUSION MATRIX
┌──────────────────────┬────────────────────┬──────────────────────────┐
│                      │ Predicted Suitable │ Predicted Not Suitable   │
├──────────────────────┼────────────────────┼──────────────────────────┤
│ Actually Suitable    │ 8                  │ 2                        │
│ Actually Not Suitable│ 1                  │ 9                        │
└──────────────────────┴────────────────────┴──────────────────────────┘
```

## 📊 Example Output (JSON)

```json
{
  "image_path": "product.jpg",
  "quality_score": 0.78,
  "verdict": "suitable",
  "reasoning": "The image demonstrates good sharpness with a Laplacian variance of 245.3...",
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
      "num_objects": 1,
      "primary_object": {"class": "shoe", "confidence": 0.94}
    },
    "background_analysis": {
      "cleanliness_score": 0.76,
      "is_clean": true,
      "edge_density": 0.15,
      "color_variance": 0.22
    }
  },
  "feature_importance": {
    "sharpness": 0.35,
    "background": 0.40,
    "objects": 0.25
  },
  "metadata": {
    "processing_time_seconds": 2.34,
    "llm_model": "llama-3.3-70b-versatile"
  }
}
```

## ⚙️ Configuration

### Model Selection

Edit `.env` to choose default model:
```bash
DEFAULT_MODEL=groq  # Options: groq, claude, gemini
```

### Feature Extraction Thresholds

Adjust sensitivity in `.env`:
```bash
BLUR_THRESHOLD=100.0                      # Lower = more strict on blur
BACKGROUND_CLEANLINESS_THRESHOLD=0.6      # Higher = cleaner background required
MIN_OBJECT_CONFIDENCE=0.5                 # Confidence threshold for object detection
```

### Logging

```bash
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR
```

## 📝 Technical Write-up

See [technical_writeup.md](technical_writeup.md) for:
- System architecture details
- Model comparison and trade-offs
- Design decisions and justifications
- Limitations and failure modes
- Production deployment considerations
- Evaluation methodology

## 🔍 Model Comparison

| Model | Provider | Speed | Cost | Reasoning Quality | Best For |
|-------|----------|-------|------|-------------------|----------|
| **Llama 3.3 70B** | Groq | ⚡⚡⚡ Fast (2-3s) | Free | ⭐⭐⭐⭐ Excellent | Development, fast iteration |
| **Claude 3.5 Sonnet** | AWS Bedrock | ⚡⚡ Medium (5-7s) | $$$ Paid | ⭐⭐⭐⭐⭐ Best | Production, critical decisions |
| **Gemini 2.5 Flash** | Google | ⚡⚡ Medium (5-7s) | Free tier | ⭐⭐⭐⭐ Very Good | Large-context tasks, structured reasoning |

**Recommendation:** 
- **Development:** Use Groq (free, fast, good quality)
- **Production:** Use Claude for best accuracy, Gemini for speed
- **Validation:** Run all 3 and use consensus voting

## 🚧 Limitations

- **Blur detection:** Can be fooled by intentional bokeh or high-contrast edges
- **Object detection:** Limited to 80 COCO classes; may miss niche products
- **Background analysis:** Cannot distinguish foreground from background without segmentation
- **Cultural bias:** "Professional" aesthetics may vary across cultures
- **LLM hallucinations:** Models may occasionally invent features not present
- **Model disagreement:** Different LLMs can give conflicting verdicts

## 🔮 Future Improvements

- [ ] Add semantic segmentation for precise background isolation
- [ ] Implement OCR for brand/label text extraction
- [ ] Add CLIP embeddings for semantic "professionalism" scoring
- [ ] Support batch processing of multiple images
- [ ] Add result caching for duplicate images
- [ ] Implement ensemble voting for multi-model consensus
- [ ] Create larger evaluation dataset with domain expert labels
- [ ] Add confidence calibration and uncertainty quantification
- [ ] Support custom fine-tuning of quality thresholds per use case

## 💡 Design Decisions & Trade-offs

### Why Multiple LLMs?

**Benefits:**
- Reduces single-model bias
- Provides confidence through consensus
- Allows fallback if one API fails
- Enables cost/speed/quality trade-offs

**Trade-offs:**
- More complex setup (multiple API keys)
- Higher latency when comparing all models
- Potential disagreements require handling

### Why These Specific Models?

**Llama 3.3 70B (Groq):**
- ✅ Free and fast via Groq infrastructure
- ✅ Excellent reasoning for this task
- ❌ Requires Groq account

**Claude 3.5 Sonnet (AWS Bedrock):**
- ✅ Best-in-class reasoning and structured output
- ✅ Enterprise-grade reliability
- ❌ Costs money, requires AWS setup

**Gemini 2.5 Flash (Google):**
- ✅ Strong reasoning and structured outputs
- ✅ Generous free tier
- ✅ Good at structured tasks
- ❌ Slightly higher latency than ultra-light models

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Groq for fast LLM inference
- AWS Bedrock for Claude access
- Google for Gemini API
- OpenCV community
- Streamlit team
- LangChain for LLM wrappers

## 📚 Additional Resources

- [Technical Write-up](technical_writeup.md) - Detailed architecture and decisions
- [API Documentation](https://docs.claude.com) - Claude API docs
- [Groq Documentation](https://console.groq.com/docs) - Groq API docs
- [Gemini Documentation](https://ai.google.dev/docs) - Gemini API docs
- [YOLOv8 Documentation](https://docs.ultralytics.com) - Object detection

## 🆘 Troubleshooting

**Issue:** "Missing GROQ_API_KEY"
- **Solution:** Add `GROQ_API_KEY=your_key` to `.env` file

**Issue:** Claude reasoner fails
- **Solution:** Verify AWS credentials and Bedrock access in your region

**Issue:** Gemini rate limit exceeded
- **Solution:** Check your Google API quota, wait, or upgrade tier

**Issue:** YOLO model download fails
- **Solution:** Check internet connection, model will auto-download (~6MB)

**Issue:** Models disagree on verdict
- **Solution:** This is normal! Use the comparison output to make informed decisions

---

For detailed technical documentation, see [technical_writeup.md](technical_writeup.md)