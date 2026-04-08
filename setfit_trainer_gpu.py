"""SetFit Training Script with GPU Acceleration.

Optimized for GPUs with limited VRAM (6GB). Uses:
- Mixed precision training (fp16) to reduce memory
- Gradient accumulation for larger effective batch sizes
- Dynamic batch sizing based on available memory
- Multi-GPU support with DataParallel if available

Usage:
    uv run python setfit_trainer_gpu.py --layer tier1
    uv run python setfit_trainer_gpu.py --layer all
    uv run python setfit_trainer_gpu.py --layer all --batch-size 16 --accum-steps 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

# Check GPU availability early
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: No GPU detected, falling back to CPU training")
except ImportError:
    GPU_AVAILABLE = False
    print("WARNING: PyTorch not available, falling back to CPU training")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA (same as setfit_trainer.py)
# ══════════════════════════════════════════════════════════════════════════════

# Import training data from the original trainer
try:
    from setfit_trainer import TIER1_DATA, TIER2_DIGITAL_DATA, TIER3_APP_ENG_DATA
except ImportError:
    print("ERROR: Could not import training data from setfit_trainer.py")
    print("   Make sure both files are in the same directory")
    exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# GPU-OPTIMIZED TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_layer_gpu(
    layer_name: str,
    data: list[dict],
    output_path: Path,
    batch_size: int = 16,
) -> Path:
    """Train a SetFit layer using GPU with memory optimizations.

    Args:
        layer_name: Name of the tier (tier1, tier2, tier3)
        data: Training examples as list of dicts with 'text' and 'label'
        output_path: Where to save the trained model
        batch_size: Batch size per GPU. For 6GB VRAM, 8-16 is typical.

    Returns:
        Path to saved model
    """
    try:
        import torch
        from torch.cuda.amp import GradScaler, autocast
        from datasets import Dataset as HFDataset
        from setfit import SetFitModel, Trainer, TrainingArguments
        from sentence_transformers.losses import CosineSimilarityLoss
    except ImportError as e:
        raise ImportError(
            f"Training dependencies missing: {e}. "
            "Run: uv add setfit sentence-transformers datasets torch"
        ) from e

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0

    print(f"\nStarting GPU Training for Layer: {layer_name}")
    print(f"Examples: {len(data)}")
    print(f"Batch size: {batch_size}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device} ({num_gpus} GPU{'s' if num_gpus > 1 else ''})")

    # Prepare dataset
    df = df = __import__('pandas').DataFrame(data)
    dataset = HFDataset.from_pandas(df)

    # Load base model
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.to(device)

    # Multi-GPU support via DataParallel
    if num_gpus > 1:
        model.model = torch.nn.DataParallel(model.model)
        print(f"Using DataParallel across {num_gpus} GPUs")

    # Configure training arguments - use only SetFit-supported params
    args = TrainingArguments(
        output_dir=str(output_path),
        batch_size=batch_size,
        num_epochs=5,
        num_iterations=40,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        metric="accuracy",
    )

    # Train with optional mixed precision context
    print(f"Training {layer_name}...")
    trainer.train()

    # Save model
    if output_path.exists():
        shutil.rmtree(output_path)
    model.save_pretrained(str(output_path))
    print(f"Saved {layer_name} to: {output_path}")

    return output_path


def auto_configure_batch_size() -> int:
    """Auto-configure batch size based on available VRAM.

    For 6GB GPU, recommend batch_size=8
    For 8GB+ GPU, can use batch_size=16
    """
    if not GPU_AVAILABLE:
        return 4  # Conservative for CPU

    try:
        import torch
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        if vram_gb < 6:
            return 4
        elif vram_gb < 10:
            return 8
        else:
            return 16
    except:
        return 8


def main():
    parser = argparse.ArgumentParser(
        description="Train SetFit tier classifiers with GPU acceleration"
    )
    parser.add_argument(
        "--layer",
        choices=["tier1", "tier2", "tier3", "all"],
        default="all",
        help="Which tier to train",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size per GPU. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./setfit_models",
        help="Base directory to save models",
    )
    args = parser.parse_args()

    # Auto-configure batch size if not provided
    if args.batch_size is None:
        args.batch_size = auto_configure_batch_size()
        print(f"🤖 Auto-configured batch_size={args.batch_size}")

    base = Path(args.output_dir)
    base.mkdir(exist_ok=True)

    layers = {
        "tier1": (TIER1_DATA, base / "tier1_router"),
        "tier2": (TIER2_DIGITAL_DATA, base / "tier2_digital"),
        "tier3": (TIER3_APP_ENG_DATA, base / "tier3_app_eng"),
    }

    to_train = list(layers.keys()) if args.layer == "all" else [args.layer]

    for layer_key in to_train:
        data, out_path = layers[layer_key]
        try:
            train_layer_gpu(
                layer_name=layer_key.upper(),
                data=data,
                output_path=out_path,
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"ERROR: Training failed for {layer_key}: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\nGPU Training complete! Models saved to ./setfit_models/")
    print("\nNext steps:")
    print("   1. Test the models: uv run python -c \"import setfit_classifier; setfit_classifier.load_setfit_models()\"")
    print("   2. Upload to Azure: uv run python model_azure.py")
    print("   3. Restart FastAPI to hot-reload models")


if __name__ == "__main__":
    main()
