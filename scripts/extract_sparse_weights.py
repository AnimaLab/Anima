#!/usr/bin/env python3
"""Extract BGE-M3 sparse linear weights to a compact binary format for Rust consumption.

Reads sparse_linear.pt from the BGE-M3 HuggingFace checkpoint and writes sparse_linear.bin:
  [u32 count=1024][f32 weight x 1024][f32 bias]
  Total: 4104 bytes, little-endian.
"""
import struct
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

def main():
    # Download sparse_linear.pt from HuggingFace if not present
    pt_path = Path("sparse_linear.pt")
    if not pt_path.exists():
        print("Downloading sparse_linear.pt from BAAI/bge-m3...")
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id="BAAI/bge-m3", filename="sparse_linear.pt")
        import shutil
        shutil.copy(downloaded, pt_path)
        print(f"Downloaded to {pt_path}")

    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    weight = state["weight"].squeeze().float().numpy()  # [1024]
    bias = state["bias"].squeeze().float().item()        # scalar

    assert len(weight) == 1024, f"Expected 1024 weights, got {len(weight)}"

    out_path = Path("models/bge-m3/sparse_linear.bin")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", len(weight)))  # u32 count
        for w in weight:
            f.write(struct.pack("<f", float(w)))  # f32 weights
        f.write(struct.pack("<f", bias))          # f32 bias

    print(f"Written {out_path} ({out_path.stat().st_size} bytes)")
    print(f"  Weight shape: [{len(weight)}], range: [{weight.min():.4f}, {weight.max():.4f}]")
    print(f"  Bias: {bias:.6f}")

if __name__ == "__main__":
    main()
