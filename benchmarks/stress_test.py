import torch
import numpy as np
import time
import psutil
import os
import sys
import argparse
import json
from pathlib import Path

# Add root path to sys.path to import model modules
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from model.fno2d import FNO2d
from model.config import FNOConfig
from model.utils import load_data, load_checkpoint

def measure_memory():
    """Get current RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def stress_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Metadata & Configuration
    checkpoint_dir = Path(args.checkpoint)
    if checkpoint_dir.is_file(): checkpoint_dir = checkpoint_dir.parent
    metadata_path = checkpoint_dir / "training_metadata.json"
    
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found. Cannot determine model config.")
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
    
    # 2. Load Model
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

    # 3. Load Data
    print(f"Loading test data from {args.data_path}")
    # Load a small batch for repeated testing
    try:
        data_loader, _ = load_data(args.data_path, batch_size=args.batch_size)
        # Get one batch for stress testing
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

    # 4. Stress Test Loop
    # User Request: Run 250 times, measure time from sample 50 to 250 (200 samples)
    total_runs = 250
    start_measure_idx = 50
    measure_count = total_runs - start_measure_idx
    
    print(f"\nStarting Throughput Test on {args.data_path}")
    print(f"Total iterations: {total_runs}")
    print(f"Measurement window: Sample {start_measure_idx} to {total_runs} ({measure_count} samples)")
    print("-" * 60)
    print(f"{'Run':<10} | {'Status':<15} | {'RAM (MB)':<15} | {'VRAM (MB)':<15}")
    print("-" * 60)

    start_ram = measure_memory()
    t_start_measurement = 0
    t_end_measurement = 0
    
    try:
        with torch.no_grad():
            for i in range(1, total_runs + 1):
                # Synchronize before timing critical sections if needed
                # For throughput, we want "wall clock time for the batch", so we just mark start/end
                
                if i == start_measure_idx + 1:
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    t_start_measurement = time.time()
                    print(f"{i-1:<10} | {'Warmup Done':<15} | {measure_memory():<15.2f}")
                
                # Inference
                _ = model(a, f)
                
                # Check memory occasionally
                if i % 50 == 0:
                    curr_ram = measure_memory()
                    curr_vram = 0
                    if torch.cuda.is_available():
                        curr_vram = torch.cuda.memory_allocated() / 1024 / 1024
                    print(f"{i:<10} | {'Running...':<15} | {curr_ram:<15.2f} | {curr_vram:<15.2f}")

            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_end_measurement = time.time()
                    
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        return

    # 5. Report
    total_time_sec = t_end_measurement - t_start_measurement
    avg_latency_ms = (total_time_sec * 1000) / measure_count
    throughput = measure_count / total_time_sec
    end_ram = measure_memory()
    
    print("-" * 60)
    print("Throughput Test Results")
    print("-" * 60)
    print(f"Measured Samples: {measure_count} (from {start_measure_idx} to {total_runs})")
    print(f"Total Time:       {total_time_sec:.4f} s")
    print(f"Avg Latency:      {avg_latency_ms:.4f} ms_per_sample")
    print(f"Throughput:       {throughput:.2f} samples/s")
    print(f"RAM Growth:       {end_ram - start_ram:.4f} MB")
    
    # Save report
    with open("benchmark_report.txt", "w") as f:
        f.write("FNO Throughput Test Report\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data: {args.data_path}\n")
        f.write(f"Window: Sample {start_measure_idx} to {total_runs}\n\n")
        f.write(f"Total Time:  {total_time_sec:.4f} s\n")
        f.write(f"Throughput:  {throughput:.2f} sample/s\n")
        f.write(f"Avg Latency: {avg_latency_ms:.4f} ms\n")
        f.write(f"RAM Growth:  {end_ram - start_ram:.4f} MB\n")
    
    print(f"\nReport saved to benchmark_report.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir or file")
    parser.add_argument("--data_path", type=str, default="benchmarks/val.npz")
    parser.add_argument("--runs", type=int, default=250, help="Total runs (default 250)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--dry-run", action="store_true", help="Run without existing data specific path")
    
    args = parser.parse_args()
    stress_test(args)
