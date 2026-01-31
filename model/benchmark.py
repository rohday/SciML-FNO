import torch
import time
import psutil
import os
import argparse
import numpy as np
from pathlib import Path

from model.config import FNOConfig
from model.fno2d import FNO2d
from model.utils import load_data, load_checkpoint

# Import classical solver components
import sys
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path)) # Add root to path
sys.path.append(str(root_path / 'data_generation')) # Add data_generation for 'import config' to work
from data_generation.solve_poisson import solve_poisson
from data_generation.config import DataGenConfig

def measure_memory():
    """Get current RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}")
    
    # 1. Load Data (Single Batch for latency testing)
    print(f"Loading test data from {args.data_path}")
    test_loader, _ = load_data(args.data_path, batch_size=1) 
    # Use batch_size=1 for pure latency metrics
    
    # Get one sample
    a, f, u_true = next(iter(test_loader))
    a, f = a.to(device), f.to(device)
    
    # 2. Load Model
    config = FNOConfig(
        modes_x=args.modes,
        modes_y=args.modes,
        width=args.width,
        depth=args.depth
    )
    model = FNO2d(config).to(device)
    
    checkpoint_path = args.checkpoint
    if Path(checkpoint_path).is_dir():
         checkpoint_path = f"{checkpoint_path}/best_model.pth"
         
    if Path(checkpoint_path).exists():
        load_checkpoint(model, None, checkpoint_path)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, benchmarking initialized model (performance is same).")
        
    model.eval()
    
    # ==========================================
    # FNO Benchmark
    # ==========================================
    print("\n" + "="*40)
    print("FNO Model Benchmark")
    print("="*40)
    
    # Warmup
    print("Warming up (10 runs)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(a, f)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
    # Latency Measurement
    latencies = []
    
    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_ram = measure_memory()
    
    print(f"Running {args.runs} inference runs...")
    with torch.no_grad():
        for _ in range(args.runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            
            out = model(a, f)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            latencies.append((t1 - t0) * 1000) # ms
            
    fno_avg_latency = np.mean(latencies)
    fno_std_latency = np.std(latencies)
    fno_throughput = 1000 / fno_avg_latency
    
    fno_ram = measure_memory() - start_ram # Approx diff
    fno_vram = 0
    if torch.cuda.is_available():
        fno_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
        
    print(f"Avg Latency:   {fno_avg_latency:.4f} ms ± {fno_std_latency:.4f}")
    print(f"Throughput:    {fno_throughput:.2f} samples/s")
    print(f"RAM Usage:     ~{fno_ram:.2f} MB (Process growth)")
    if torch.cuda.is_available():
        print(f"Peak VRAM:     {fno_vram:.2f} MB")
        
    # ==========================================
    # Classical Solver Benchmark
    # ==========================================
    if args.compare:
        print("\n" + "="*40)
        print("Classical Solver (SciPy) Benchmark")
        print("="*40)
        
        # Prepare inputs (CPU)
        a_cpu = a.cpu()
        f_cpu = f.cpu()
        gen_config = DataGenConfig(grid_size=a.shape[-1])
        
        # Warmup (Just 1-2 runs, it's slow)
        print("Warming up (2 runs)...")
        for _ in range(2):
            _ = solve_poisson(a_cpu[0,0], f_cpu[0,0], gen_config)
            
        classical_latencies = []
        classical_runs = max(1, args.runs // 5) # Run fewer times
        
        print(f"Running {classical_runs} solver runs...")
        for i in range(classical_runs):
            t0 = time.time()
            _ = solve_poisson(a_cpu[0,0], f_cpu[0,0], gen_config)
            t1 = time.time()
            classical_latencies.append((t1 - t0) * 1000)
            
        cls_avg_latency = np.mean(classical_latencies)
        speedup = cls_avg_latency / fno_avg_latency
        
        print(f"Avg Latency:   {cls_avg_latency:.4f} ms")
        print(f"Throughput:    {1000/cls_avg_latency:.2f} samples/s")
        print("-" * 40)
        print(f"FNO Speedup:   {speedup:.1f}x FASTER")
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints")
    parser.add_argument("--data_path", type=str, default="data/test.npz")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--compare", action="store_true", help="Run classical solver for comparison")
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    
    args = parser.parse_args()
    benchmark(args)
