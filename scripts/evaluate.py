"""Evaluation script for e-commerce image validator."""

import argparse
import sys
from pathlib import Path

from tabulate import tabulate

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ecommerce_image_validator.logger import setup_logger
from src.ecommerce_image_validator.pipeline import ImageValidationPipeline

logger = setup_logger(__name__)


class EvaluationMetrics:
    """
    Container for evaluation metrics.
    
    Attributes
    ----------
    tp : int
        True positives (predicted suitable, actually suitable)
    tn : int
        True negatives (predicted not_suitable, actually not_suitable)
    fp : int
        False positives (predicted suitable, actually not_suitable)
    fn : int
        False negatives (predicted not_suitable, actually suitable)
    """
    
    def __init__(self):
        self.tp = 0  # True Positive
        self.tn = 0  # True Negative
        self.fp = 0  # False Positive
        self.fn = 0  # False Negative
        self.failed_cases = []
    
    def update(self, predicted: str, actual: str, image_name: str):
        """
        Update metrics based on prediction.
        
        Parameters
        ----------
        predicted : str
            Predicted verdict ('suitable' or 'not_suitable')
        actual : str
            Actual ground truth label ('suitable' or 'not_suitable')
        image_name : str
            Name of the image file
        """
        # Treat 'uncertain' as 'not_suitable' for binary classification
        if predicted == "uncertain":
            predicted = "not_suitable"
        
        if actual == "suitable" and predicted == "suitable":
            self.tp += 1
        elif actual == "not_suitable" and "not" in predicted:
            self.tn += 1
        elif actual == "not_suitable" and predicted == "suitable":
            self.fp += 1
            self.failed_cases.append((image_name, predicted, actual))
        elif actual == "suitable" and "not" in predicted:
            self.fn += 1
            self.failed_cases.append((image_name, predicted, actual))
    
    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0
    
    @property
    def precision(self) -> float:
        """Calculate precision."""
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Calculate recall."""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        p = self.precision
        r = self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def total(self) -> int:
        """Total number of samples."""
        return self.tp + self.tn + self.fp + self.fn


def load_evaluation_dataset(eval_dir: Path) -> dict:
    """
    Load evaluation dataset from directory structure.
    
    Expected structure:
    eval_dir/
    ├── good/       # Suitable images
    │   ├── img1.jpg
    │   └── img2.jpg
    └── bad/        # Not suitable images
        ├── img3.jpg
        └── img4.jpg
    
    Parameters
    ----------
    eval_dir : Path
        Path to evaluation directory
    
    Returns
    -------
    dict
        Dictionary mapping image paths to labels
    
    Raises
    ------
    ValueError
        If directory structure is invalid
    """
    good_dir = eval_dir / "good"
    bad_dir = eval_dir / "bad"
    
    if not good_dir.exists() or not bad_dir.exists():
        raise ValueError(
            f"Invalid evaluation directory structure. Expected:\n"
            f"  {eval_dir}/good/  (suitable images)\n"
            f"  {eval_dir}/bad/   (not suitable images)"
        )
    
    dataset = {}
    
    # Load good images
    for img_path in good_dir.glob("*"):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            dataset[img_path] = "suitable"
    
    # Load bad images
    for img_path in bad_dir.glob("*"):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            dataset[img_path] = "not_suitable"
    
    if not dataset:
        raise ValueError(f"No images found in {eval_dir}")
    
    logger.info(
        f"Loaded {len(dataset)} images: "
        f"{len([l for l in dataset.values() if l == 'suitable'])} suitable, "
        f"{len([l for l in dataset.values() if l == 'not_suitable'])} not suitable"
    )
    
    return dataset


def evaluate_model(model_name: str, dataset: dict) -> EvaluationMetrics:
    """
    Evaluate a model on the dataset.
    
    Parameters
    ----------
    model_name : str
        Name of the model ('groq', 'claude', or 'gemini')
    dataset : dict
        Dictionary mapping image paths to ground truth labels
    
    Returns
    -------
    EvaluationMetrics
        Computed evaluation metrics
    """
    logger.info(f"Evaluating {model_name.upper()} on {len(dataset)} images...")
    
    pipeline = ImageValidationPipeline(llm_type=model_name)
    metrics = EvaluationMetrics()
    
    for idx, (image_path, ground_truth) in enumerate(dataset.items(), 1):
        logger.info(f"[{idx}/{len(dataset)}] Processing {image_path.name}...")
        
        try:
            result = pipeline.validate(image_path)
            predicted = result.verdict
            
            logger.info(
                f"  Predicted: {predicted}, Actual: {ground_truth}, "
                f"Score: {result.quality_score:.2f}"
            )
            
            metrics.update(predicted, ground_truth, image_path.name)
        
        except Exception as e:
            logger.error(f"  Failed to process {image_path.name}: {str(e)}")
            # Treat failures as incorrect predictions
            metrics.update("error", ground_truth, image_path.name)
    
    return metrics


def display_results(model_name: str, metrics: EvaluationMetrics):
    """
    Display evaluation results.
    
    Parameters
    ----------
    model_name : str
        Name of the evaluated model
    metrics : EvaluationMetrics
        Computed metrics
    """
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS: {model_name.upper()}")
    print("="*80 + "\n")
    
    # Summary metrics
    print(f"Total Images: {metrics.total}")
    print(f"Accuracy: {metrics.accuracy:.1%} ({metrics.tp + metrics.tn}/{metrics.total})")
    print(f"Precision: {metrics.precision:.1%}")
    print(f"Recall: {metrics.recall:.1%}")
    print(f"F1-Score: {metrics.f1_score:.1%}")
    
    # Confusion matrix
    print("\n" + "-"*80)
    print("CONFUSION MATRIX")
    print("-"*80)
    
    confusion_data = [
        ["Actually Suitable", metrics.tp, metrics.fn],
        ["Actually Not Suitable", metrics.fp, metrics.tn]
    ]
    headers = ["", "Predicted Suitable", "Predicted Not Suitable"]
    print(tabulate(confusion_data, headers=headers, tablefmt="grid"))
    
    print("\nKey:")
    print(f"  TP (True Positive): {metrics.tp} - Correctly identified suitable images")
    print(f"  TN (True Negative): {metrics.tn} - Correctly identified unsuitable images")
    print(f"  FP (False Positive): {metrics.fp} - Incorrectly marked unsuitable as suitable")
    print(f"  FN (False Negative): {metrics.fn} - Incorrectly marked suitable as unsuitable")
    
    # Failed cases
    if metrics.failed_cases:
        print("\n" + "-"*80)
        print("FAILED CASES")
        print("-"*80 + "\n")
        
        for image_name, predicted, actual in metrics.failed_cases:
            print(f"❌ {image_name}")
            print(f"   Predicted: {predicted}, Should be: {actual}\n")
    else:
        print("\n✅ No failed cases - Perfect accuracy!")
    
    print("="*80 + "\n")


def compare_all_models(dataset: dict):
    """
    Evaluate all available models and compare results.
    
    Parameters
    ----------
    dataset : dict
        Evaluation dataset
    """
    models = ["groq", "claude", "gemini"]
    all_metrics = {}
    
    print("\n" + "="*80)
    print("RUNNING MULTI-MODEL EVALUATION")
    print("="*80 + "\n")
    
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Evaluating {model_name.upper()}...")
        print('='*80)
        
        try:
            metrics = evaluate_model(model_name, dataset)
            all_metrics[model_name] = metrics
            
            print(f"\n{model_name.upper()} completed:")
            print(f"  Accuracy: {metrics.accuracy:.1%}")
            print(f"  F1-Score: {metrics.f1_score:.1%}")
        
        except Exception as e:
            logger.error(f"{model_name.upper()} evaluation failed: {str(e)}")
            print(f"❌ {model_name.upper()} failed: {str(e)}")
    
    # Display comparison table
    if all_metrics:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80 + "\n")
        
        comparison_data = []
        for model_name, metrics in all_metrics.items():
            comparison_data.append([
                model_name.upper(),
                f"{metrics.accuracy:.1%}",
                f"{metrics.precision:.1%}",
                f"{metrics.recall:.1%}",
                f"{metrics.f1_score:.1%}"
            ])
        
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
        print(tabulate(comparison_data, headers=headers, tablefmt="grid"))
        
        # Best model
        best_model = max(all_metrics.items(), key=lambda x: x[1].accuracy)
        print(f"\n🏆 Best Model: {best_model[0].upper()} "
              f"(Accuracy: {best_model[1].accuracy:.1%})")
        print("="*80 + "\n")


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate image validator on labeled dataset"
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="examples/evaluation",
        help="Path to evaluation directory (default: examples/evaluation)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["groq", "claude", "gemini"],
        default=None,
        help="Specific model to evaluate (default: compare all)"
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Evaluate and compare all models"
    )
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    
    try:
        # Load dataset
        dataset = load_evaluation_dataset(eval_dir)
        
        if args.all_models or args.model is None:
            # Compare all models
            compare_all_models(dataset)
        else:
            # Evaluate single model
            metrics = evaluate_model(args.model, dataset)
            display_results(args.model, metrics)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()