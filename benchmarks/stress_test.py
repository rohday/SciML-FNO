"""Stress test for FNO throughput benchmarking."""

import torch
import numpy as np
import time
import psutil
import os
import sys
import argparse
import json
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from model.fno2d import FNO2d
from model.config import FNOConfig
from model.utils import load_data, load_checkpoint


def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def stress_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint_dir = Path(args.checkpoint)
    if checkpoint_dir.is_file(): checkpoint_dir = checkpoint_dir.parent
    metadata_path = checkpoint_dir / "training_metadata.json"
    
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found")
        return

    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
        
    config = FNOConfig(
        modes_x=meta.get('modes', 12),
        modes_y=meta.get('modes', 12),
        width=meta.get('width', 32),
        depth=meta.get('depth', 4)
    )
    
    model = FNO2d(config).to(device)
    checkpoint_path = args.checkpoint
    if Path(checkpoint_path).is_dir():
         checkpoint_path = f"{checkpoint_path}/model.pth"
         
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        epoch = load_checkpoint(model, None, checkpoint_path)
        print(f"Model loaded from epoch {epoch}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    model.eval()

    print(f"Loading test data from {args.data_path}")
    try:
        data_loader, _ = load_data(args.data_path, batch_size=args.batch_size)
        batch = next(iter(data_loader))
        a, f, u = batch
        a, f = a.to(device), f.to(device)
        print(f"Input shape: {a.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        if args.dry_run:
            print("Dry run: Creating synthetic data")
            a = torch.randn(args.batch_size, 1, 64, 64).to(device)
            f = torch.randn(args.batch_size, 1, 64, 64).to(device)
        else:
            return

    total_runs = 250
    start_measure_idx = 50
    measure_count = total_runs - start_measure_idx
    
    print(f"\nThroughput Test: {total_runs} runs, measuring {start_measure_idx} to {total_runs}")
    print("-" * 60)

    start_ram = measure_memory()
    t_start_measurement = 0
    t_end_measurement = 0
    
    try:
        with torch.no_grad():
            for i in range(1, total_runs + 1):
                if i == start_measure_idx + 1:
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    t_start_measurement = time.time()
                    print(f"Warmup done at iteration {i-1}")
                
                _ = model(a, f)
                
                if i % 50 == 0:
                    curr_ram = measure_memory()
                    curr_vram = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                    print(f"Iter {i}: RAM={curr_ram:.2f}MB, VRAM={curr_vram:.2f}MB")

            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_end_measurement = time.time()
                    
    except KeyboardInterrupt:
        print("\nInterrupted")
        return

    total_time_sec = t_end_measurement - t_start_measurement
    avg_latency_ms = (total_time_sec * 1000) / measure_count
    throughput = measure_count / total_time_sec
    end_ram = measure_memory()
    
    print("-" * 60)
    print("Results")
    print("-" * 60)
    print(f"Samples: {measure_count}")
    print(f"Time: {total_time_sec:.4f}s")
    print(f"Latency: {avg_latency_ms:.4f} ms/sample")
    print(f"Throughput: {throughput:.2f} samples/s")
    print(f"RAM Growth: {end_ram - start_ram:.4f} MB")
    
    with open("benchmark_report.txt", "w") as f:
        f.write("FNO Throughput Test\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data: {args.data_path}\n\n")
        f.write(f"Time: {total_time_sec:.4f}s\n")
        f.write(f"Throughput: {throughput:.2f} samples/s\n")
        f.write(f"Latency: {avg_latency_ms:.4f} ms\n")
    
    print(f"\nSaved to benchmark_report.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="benchmarks/val.npz")
    parser.add_argument("--runs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    
    args = parser.parse_args()
    stress_test(args)
