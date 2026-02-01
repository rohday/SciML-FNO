import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import psutil
import os
import sys

# Add root path for data_generation imports if needed
root_path = Path(__file__).parent.parent
# Put data_generation FIRST to avoid shadowing by model/config.py
if str(root_path / 'data_generation') not in sys.path:
    sys.path.insert(0, str(root_path / 'data_generation'))
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from model.config import FNOConfig
from model.fno2d import FNO2d
from model.utils import load_data, GaussianNormalizer, load_checkpoint
from model.train import LpLoss

# Conditional import for classical solver to avoid breaking if dependencies missing
try:
    from data_generation.solve_poisson import solve_poisson
    from data_generation.config import DataGenConfig
except ImportError:
    solve_poisson = None
    DataGenConfig = None

def measure_memory():
    """Get current RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Configuration (Must match training)
    config = FNOConfig(
        modes_x=args.modes,
        modes_y=args.modes,
        width=args.width,
        depth=args.depth
    )
    
    # 2. Model
    model = FNO2d(config).to(device)
    
    # 3. Load Checkpoint
    checkpoint_path = args.checkpoint
    if Path(checkpoint_path).is_dir():
         checkpoint_path = f"{checkpoint_path}/best_model.pth"
    
    if not Path(checkpoint_path).exists():
        # Fallback for benchmarking without a trained model if just testing speed
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Benchmarking initialized model.")
    else:
        print(f"Loading checkpoint: {checkpoint_path}")
        epoch = load_checkpoint(model, None, checkpoint_path) # Optimizer=None for inference
        print(f"Model loaded from epoch {epoch}")
    
    # 4. Data Loading
    print(f"Loading test data from {args.data_path}")
    test_loader, _ = load_data(args.data_path, batch_size=32)
    # For benchmarking latency, we might need a single sample
    bench_loader, _ = load_data(args.data_path, batch_size=1)
    
    # 5. Accuracy Evaluation Loop
    model.eval()
    myloss = LpLoss(size_average=True)
    total_l2 = 0.0
    num_batches = 0
    
    # Store first batch for visualization
    vis_a, vis_f, vis_u, vis_pred = None, None, None, None
    
    with torch.no_grad():
        for i, (a, f, u) in enumerate(test_loader):
            a, f, u = a.to(device), f.to(device), u.to(device)
            
            out = model(a, f)
            loss = myloss(out, u)
            
            total_l2 += loss.item()
            num_batches += 1
            
            if i == 0:
                # Capture up to 8 samples
                n_vis = min(8, a.shape[0])
                vis_a = a[:n_vis].cpu()
                vis_f = f[:n_vis].cpu()
                vis_u = u[:n_vis].cpu()
                vis_pred = out[:n_vis].cpu()

    # Buffer for stats to save to file
    stats_buffer = []
    def log(msg=""):
        print(msg)
        stats_buffer.append(str(msg))

    avg_l2 = total_l2 / num_batches if num_batches > 0 else 0
    log(f"Average Relative L2 Error: {avg_l2:.5f}")
    acc_approx = (1 - avg_l2) * 100
    log(f"Accuracy: {acc_approx:.2f}% (Approx)")

    log("\n" + "-"*40)
    log("FNO Model Performance Benchmark")
    log("-"*40)
    
    # Get one sample for latency testing
    a_bench, f_bench, _ = next(iter(bench_loader))
    a_bench, f_bench = a_bench.to(device), f_bench.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(a_bench, f_bench)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
    # Latency Measurement
    latencies = []
    
    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_ram = measure_memory()
    
    with torch.no_grad():
        for _ in range(args.runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            
            out = model(a_bench, f_bench)
            
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
        
    log(f"Avg Latency:   {fno_avg_latency:.4f} ms ± {fno_std_latency:.4f}")
    log(f"Throughput:    {fno_throughput:.2f} samples/s")
    log(f"RAM Usage:     ~{fno_ram:.2f} MB (Process growth)")
    if torch.cuda.is_available():
        log(f"Peak VRAM:     {fno_vram:.2f} MB")

    # 7. Classical Solver Benchmark (Optional)
    if args.compare and solve_poisson is not None:
        log("\n" + "-"*40)
        log("Classical Solver (SciPy) Benchmark")
        log("-"*40)
        
        # Prepare inputs (CPU)
        a_cpu = a_bench.cpu()
        f_cpu = f_bench.cpu()
        gen_config = DataGenConfig(grid_size=a_bench.shape[-1])
        
        # Warmup
        for _ in range(2):
            _ = solve_poisson(a_cpu[0,0], f_cpu[0,0], gen_config)
            
        classical_latencies = []
        classical_runs = max(1, args.runs // 5) # Run fewer times as it is slower
        
        for i in range(classical_runs):
            t0 = time.time()
            _ = solve_poisson(a_cpu[0,0], f_cpu[0,0], gen_config)
            t1 = time.time()
            classical_latencies.append((t1 - t0) * 1000)
            
        cls_avg_latency = np.mean(classical_latencies)
        speedup = cls_avg_latency / fno_avg_latency
        
        log(f"Avg Latency:   {cls_avg_latency:.4f} ms")
        log(f"Throughput:    {1000/cls_avg_latency:.2f} samples/s")
    elif args.compare and solve_poisson is None:
        log("\nWarning: Could not import classical solver dependencies. Skipping comparison.")
    
    # Save stats to file
    if args.output_stats:
        try:
            with open(args.output_stats, 'w') as f:
                f.write('\n'.join(stats_buffer))
            print(f"\n\"{args.output_stats}\"")
        except Exception as e:
            print(f"\nCould not save stats: {e}")

    # 8. Visualization
    if args.plot and vis_a is not None:
        try:
            plot_prediction(vis_a, vis_f, vis_u, vis_pred, args.output_plot)
            print(f"Plot saved to {args.output_plot}")
        except Exception as e:
            print(f"Could not save plot: {e}")

def plot_prediction(a, f, u, pred, filename):
    n_samples = a.shape[0]
    fig, axs = plt.subplots(n_samples, 4, figsize=(15, 3 * n_samples))
    
    # Handle single sample case (1D array of axes)
    if n_samples == 1:
        axs = axs[None, :]
    
    # Helper to plot
    def plot_field(ax, data, title):
        im = ax.imshow(data.squeeze(), cmap='jet')
        if title:
            ax.set_title(title)
        plt.colorbar(im, ax=ax)
        ax.axis('off')

    for i in range(n_samples):
        # Only set titles for the first row
        titles = ["Coefficient a(x,y)", "Source f(x,y)", "True u(x,y)", "Pred u(x,y)"] if i == 0 else [None]*4
        
        plot_field(axs[i, 0], a[i], titles[0])
        plot_field(axs[i, 1], f[i], titles[1])
        plot_field(axs[i, 2], u[i], titles[2])
        plot_field(axs[i, 3], pred[i], titles[3])
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints")
    parser.add_argument("--data_path", type=str, default="data/test.npz")
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--plot", action='store_true', help="Save a visualization plot")
    parser.add_argument("--output_plot", type=str, default="eval_result.png")
    
    # Benchmarking arguments
    parser.add_argument("--runs", type=int, default=100, help="Number of runs for latency benchmarking")
    parser.add_argument("--compare", action="store_true", help="Run classical solver for comparison")
    parser.add_argument("--output_stats", type=str, default="eval_stats.txt", help="File to save evaluation statistics")
    
    args = parser.parse_args()
    evaluate(args)
