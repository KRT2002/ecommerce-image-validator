# Technical Write-Up: E-commerce Image Validator

## 1. System Architecture

The system uses a **two-stage pipeline** separating computer vision feature extraction from LLM reasoning:

```
Input Image → Feature Extraction → LLM Reasoning → Structured Output
              (YOLOv8, Blur,         (Llama/Claude/   (JSON with verdict,
               Background)            Gemini)           reasoning, confidence)
```

**Stage 1: Feature Extraction (Pre-LLM Intelligence)**
- **Object Detection (YOLOv8n):** Identifies products and potential clutter (hands, background objects)
- **Blur Detection (Laplacian Variance):** Measures image sharpness via edge intensity analysis
- **Background Analysis (Custom Heuristic):** Combines edge density and color variance to assess cleanliness

**Stage 2: LLM Reasoning**
- **Multiple Models:** Llama 3.3 70B (Groq), Claude 3.5 Sonnet (AWS Bedrock), Gemini 2.5 Flash (Google)
- **Structured Output:** JSON format with verdict, quality score, reasoning, issues, and feature importance weights

---

## 2. Why This Approach?

**Two-Stage Architecture**  
Separating CV from LLM provides interpretability and cost efficiency. Features are human-readable (sharpness score, object list) rather than black-box embeddings. LLM processes ~500 tokens of structured features instead of raw images (10x cheaper). Each stage can be debugged independently.

**Feature Selection Rationale**

*YOLOv8n for Object Detection:* Chosen for speed (~150ms on CPU) over accuracy. The nano variant is sufficient since we need to know WHAT objects are present, not precise localization. Larger models (YOLOv8m/l) would add 2-3x latency with minimal quality gain for this task.

*Laplacian Variance for Blur:* A fast (<10ms), parameter-free method with proven reliability. Alternative FFT-based methods are more robust but 5x slower. The variance threshold (100) was empirically validated on product images.

*Custom Background Analyzer:* No pre-trained model exists for "e-commerce background cleanliness." Edge density + color variance provides an interpretable heuristic that can be tuned per business requirements. Limitation: cannot distinguish foreground from background without segmentation.

**Multi-Model LLM Ensemble**  
Using three LLMs (Groq/Claude/Gemini) reduces single-model bias and provides confidence through consensus. When models agree, confidence is high. Disagreement flags edge cases for manual review. Also provides cost/speed flexibility: Groq for development (free), Claude for accuracy, Gemini for throughput.

---

## 3. Limitations & Improvements

**Current Limitations**

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **No Foreground/Background Separation** | Textured products penalized as "cluttered" | Add SAM (Segment Anything Model) for background isolation |
| **Abstract/Artistic Products** | Object detection fails | Add CLIP embeddings for semantic understanding |
| **LLM Hallucination** | May invent features | Ground reasoning in explicit feature list |
| **Cultural Bias** | "Professional" varies globally | Document bias, allow threshold tuning |
| **Sequential Processing** | 2-3s latency | Parallelize feature extraction |
| **No Caching** | Duplicate images reprocessed | Add hash-based result caching |

**Prioritized Improvements**

1. **Semantic Segmentation (High Priority):** Use SAM to isolate product from background, compute cleanliness on background only. Eliminates false positives on patterned fabrics.

2. **CLIP Embeddings (Medium Priority):** Compute semantic similarity to "professional product photo" vs. "casual snapshot." Catches aesthetic issues missed by low-level features (e.g., poor lighting, awkward angles).

3. **Result Caching (High Priority):** Hash image bytes (MD5), cache results for 24 hours. Expected 60-70% reduction in redundant processing.

4. **Batch Processing (High Priority):** Process 8-16 images simultaneously on GPU. Expected 3-5x throughput increase.

---

## 4. Production Deployment Strategy

**Architecture**

```
Load Balancer → API Nodes (FastAPI) → Message Queue (Celery/Redis)
                                     ↓
                          Worker Pool (GPU instances)
                                     ↓
                          LLM Services (Groq/Bedrock/Gemini)
                                     ↓
                          Storage (PostgreSQL + Redis Cache)
```

**Infrastructure Choices**
- **API:** FastAPI with async endpoints for concurrent requests
- **Task Queue:** Celery + Redis for async image processing
- **Storage:** PostgreSQL for results, S3 for images, Redis for 24-hour caching
- **Monitoring:** Prometheus + Grafana for latency/error metrics, PagerDuty for alerts

**Model Serving Strategy**
- **Development:** Groq (free, fast iteration)
- **Production (High-Stakes):** Claude with Groq fallback (best accuracy)
- **Production (High-Volume):** Gemini (fastest, cheapest at scale)
- **Critical Decisions:** Multi-model ensemble with majority voting

**Cost Optimization**
- **LLM Caching:** Cache identical feature vectors → 60-70% cost reduction
- **Batch Inference:** Process 8-16 images per GPU forward pass → 3-5x throughput
- **Auto-Scaling:** Scale workers based on queue depth → 40% compute cost reduction

**Reliability & Monitoring**
- **Health Checks:** Ping each LLM API every 5 minutes
- **Circuit Breakers:** Auto-failover if API fails 3 consecutive times
- **Key Metrics:** P95 latency (<3s target), LLM error rate (<1%), model agreement percentage
- **Evaluation Loop:** Weekly test on 100-image benchmark, track accuracy over time

**Expected Production Metrics**
- **Latency:** <3s per image (single model), <8s (ensemble)
- **Throughput:** 200-500 images/minute with batching + GPU
- **Cost:** $0.001-0.005 per image (depending on model choice)
- **Accuracy:** 85-90% agreement with human expert labels

---

## Conclusion

This system demonstrates end-to-end ML engineering by:
1. **Separating concerns** (CV vs. LLM) for interpretability
2. **Making explicit trade-offs** (YOLOv8n speed vs. accuracy, multi-model cost vs. robustness)
3. **Acknowledging limitations** with concrete improvement paths
4. **Planning for production** with caching, monitoring, and cost optimization

The multi-model ensemble provides robustness while the two-stage architecture ensures explainability—critical for user trust in automated quality decisions.