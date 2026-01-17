"""Multi-model comparison script for e-commerce image validation."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from tabulate import tabulate

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ecommerce_image_validator.logger import setup_logger
from src.ecommerce_image_validator.pipeline import ImageValidationPipeline

logger = setup_logger(__name__)


def compare_models(image_path: str, models: list[str]) -> dict:
    """
    Run validation with multiple LLM models and compare results.
    
    Parameters
    ----------
    image_path : str
        Path to the image file
    models : list[str]
        List of model names to compare (e.g., ['groq', 'claude', 'gemini'])
    
    Returns
    -------
    dict
        Comparison results containing all model outputs
    
    Examples
    --------
    >>> results = compare_models('product.jpg', ['groq', 'claude'])
    >>> print(results['agreement'])
    False
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    logger.info(f"Running comparison on: {image_path.name}")
    logger.info(f"Models to compare: {', '.join(models)}")
    
    results = {
        "image": str(image_path.name),
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "agreement": None,
        "verdict_consensus": None
    }
    
    # Run validation with each model
    for model_name in models:
        logger.info(f"Running validation with {model_name.upper()}...")
        
        try:
            pipeline = ImageValidationPipeline(llm_type=model_name)
            result = pipeline.validate(image_path)
            
            results["models"][model_name] = {
                "verdict": result.verdict,
                "quality_score": result.quality_score,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "issues_detected": result.issues_detected,
                "feature_importance": result.feature_importance,
                "processing_time": result.metadata.get("processing_time_seconds", 0)
            }
            
            logger.info(
                f"{model_name.upper()}: {result.verdict} "
                f"(score={result.quality_score:.2f}, conf={result.confidence:.2f})"
            )
        
        except Exception as e:
            logger.error(f"{model_name.upper()} failed: {str(e)}")
            results["models"][model_name] = {
                "verdict": "error",
                "quality_score": None,
                "confidence": None,
                "reasoning": f"Error: {str(e)}",
                "issues_detected": [],
                "feature_importance": {},
                "processing_time": 0
            }
    
    # Analyze agreement
    verdicts = [
        m["verdict"] for m in results["models"].values() 
        if m["verdict"] != "error"
    ]
    
    if verdicts:
        # Check if all models agree
        results["agreement"] = len(set(verdicts)) == 1
        
        # Determine consensus
        if results["agreement"]:
            results["verdict_consensus"] = verdicts[0]
        else:
            # Count votes
            verdict_counts = {}
            for v in verdicts:
                verdict_counts[v] = verdict_counts.get(v, 0) + 1
            
            # Majority vote
            max_count = max(verdict_counts.values())
            if list(verdict_counts.values()).count(max_count) == 1:
                results["verdict_consensus"] = max(verdict_counts, key=verdict_counts.get)
            else:
                results["verdict_consensus"] = "no_consensus"
    
    return results


def display_comparison_table(results: dict):
    """
    Display comparison results as formatted table.
    
    Parameters
    ----------
    results : dict
        Comparison results from compare_models()
    """
    print("\n" + "="*80)
    print(f"MULTI-MODEL COMPARISON: {results['image']}")
    print("="*80 + "\n")
    
    # Prepare table data
    table_data = []
    
    for model_name, model_results in results["models"].items():
        verdict = model_results["verdict"]
        score = model_results["quality_score"]
        confidence = model_results["confidence"]
        time = model_results["processing_time"]
        
        # Format values
        score_str = f"{score:.2f}" if score is not None else "N/A"
        conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        time_str = f"{time:.2f}s" if time > 0 else "N/A"
        
        # Add emoji for verdict
        if verdict == "suitable":
            verdict_display = "✅ suitable"
        elif verdict == "not_suitable":
            verdict_display = "❌ not_suitable"
        elif verdict == "uncertain":
            verdict_display = "⚠️  uncertain"
        else:
            verdict_display = "❗ error"
        
        table_data.append([
            model_name.upper(),
            verdict_display,
            score_str,
            conf_str,
            time_str
        ])
    
    # Display table
    headers = ["Model", "Verdict", "Score", "Confidence", "Time"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Display agreement status
    print("\n" + "-"*80)
    if results["agreement"]:
        print("✅ AGREEMENT: All models agree!")
        print(f"   Consensus verdict: {results['verdict_consensus']}")
    else:
        print("❌ DISAGREEMENT: Models have different verdicts")
        if results["verdict_consensus"] != "no_consensus":
            print(f"   Majority verdict: {results['verdict_consensus']}")
        else:
            print("   No clear consensus")
        print("   → Manual review recommended")
    print("-"*80 + "\n")


def display_detailed_reasoning(results: dict):
    """
    Display detailed reasoning from each model.
    
    Parameters
    ----------
    results : dict
        Comparison results from compare_models()
    """
    print("\n" + "="*80)
    print("DETAILED REASONING BY MODEL")
    print("="*80 + "\n")
    
    for model_name, model_results in results["models"].items():
        print(f"📊 {model_name.upper()}")
        print("-" * 80)
        
        if model_results["verdict"] == "error":
            print(f"❗ Error: {model_results['reasoning']}\n")
            continue
        
        print(f"Verdict: {model_results['verdict']}")
        print(f"Quality Score: {model_results['quality_score']:.2f}")
        print(f"Confidence: {model_results['confidence']:.2f}")
        
        if model_results['issues_detected']:
            print(f"\nIssues Detected:")
            for issue in model_results['issues_detected']:
                print(f"  - {issue}")
        
        print(f"\nReasoning:")
        print(f"  {model_results['reasoning']}")
        
        if model_results['feature_importance']:
            print(f"\nFeature Importance:")
            for feature, weight in model_results['feature_importance'].items():
                print(f"  - {feature}: {weight:.2f}")
        
        print("\n")


def save_results(results: dict, output_path: str):
    """
    Save comparison results to JSON file.
    
    Parameters
    ----------
    results : dict
        Comparison results
    output_path : str
        Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    print(f"💾 Results saved to: {output_path}")


def main():
    """Main entry point for comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare multiple LLM models for image validation"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="groq,claude,gemini",
        help="Comma-separated list of models to compare (default: groq,claude,gemini)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save comparison results JSON (optional)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed reasoning from each model"
    )
    
    args = parser.parse_args()
    
    # Parse models list
    models = [m.strip().lower() for m in args.models.split(",")]
    
    # Validate models
    valid_models = {"groq", "claude", "gemini"}
    invalid_models = set(models) - valid_models
    if invalid_models:
        print(f"❌ Invalid models: {', '.join(invalid_models)}")
        print(f"   Valid models: {', '.join(valid_models)}")
        sys.exit(1)
    
    try:
        # Run comparison
        results = compare_models(args.image, models)
        
        # Display results
        display_comparison_table(results)
        
        if args.detailed:
            display_detailed_reasoning(results)
        
        # Save results if output path provided
        if args.output:
            save_results(results, args.output)
        else:
            # Auto-generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = Path(args.image).stem
            output_path = f"examples/outputs/comparison_results/comparison_{image_name}_{timestamp}.json"
            save_results(results, output_path)
    
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()