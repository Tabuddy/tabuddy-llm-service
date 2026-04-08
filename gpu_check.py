"""Quick GPU verification script for SetFit training.

Checks:
- PyTorch installation
- CUDA availability
- VRAM amount
- Recommended batch size for 6GB VRAM
"""

import sys

def main():
    print("=" * 60)
    print("GPU Verification for SetFit Training")
    print("=" * 60)

    # Check PyTorch
    try:
        import torch
        print(f"YES PyTorch: {torch.__version__}")
    except ImportError:
        print("NO PyTorch not installed")
        print("   Install: uv add torch --index https://download.pytorch.org/whl/cu118")
        sys.exit(1)

    # Check CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / 1024**3

        print(f"YES CUDA Available: {torch.version.cuda}")
        print(f"YES GPU: {device_name}")
        print(f"YES VRAM: {vram_gb:.1f} GB")

        # Recommend batch size
        if vram_gb < 6:
            batch_rec = 4
            accum_rec = 4
        elif vram_gb < 10:
            batch_rec = 8
            accum_rec = 4
        else:
            batch_rec = 16
            accum_rec = 2

        print(f"\n💡 Recommended settings for SetFit training:")
        print(f"   --batch-size {batch_rec} --accum-steps {accum_rec}")

        effective_batch = batch_rec * accum_rec
        print(f"   Effective batch size: {effective_batch}")

        if vram_gb >= 6:
            print(f"\nYES Sufficient VRAM for GPU training with mixed precision")
        else:
            print(f"\nWARNING  Limited VRAM ({vram_gb:.1f}GB). Consider:")
            print("   - Using smaller batch_size (4)")
            print("   - Using CPU training instead (omit GPU script)")

    else:
        print("NO CUDA not available")
        print("   Possible causes:")
        print("   - PyTorch installed without CUDA support")
        print("   - NVIDIA drivers not installed")
        print("\n💡 Install CUDA-enabled PyTorch:")
        print("   uv add torch --index https://download.pytorch.org/whl/cu118")
        print("   (or cu121 for CUDA 12.1)")

    print("\n" + "=" * 60)
    print("📦 Checking other dependencies...")
    print("=" * 60)

    # Check other dependencies
    deps = {
        'setfit': 'setfit',
        'sentence_transformers': 'sentence-transformers',
        'datasets': 'datasets',
        'pandas': 'pandas',
    }

    for module, package in deps.items():
        try:
            __import__(module)
            print(f"YES {package}")
        except ImportError:
            print(f"NO {package} not installed")
            print(f"   Install: uv add {package}")

    print("\nYES Verification complete!")

if __name__ == "__main__":
    main()
