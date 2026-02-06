import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import psutil
import os
import sys

root_path = Path(__file__).parent.parent
if str(root_path / 'data_generation') not in sys.path:
    sys.path.insert(0, str(root_path / 'data_generation'))
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from model.config import FNOConfig
from model.fno2d import FNO2d
from model.utils import load_data, GaussianNormalizer, load_checkpoint
from model.train import LpLoss

try:
    from data_generation.solve_poisson import solve_poisson
    from data_generation.config import DataGenConfig
except ImportError:
    solve_poisson = None
    DataGenConfig = None


def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    args.plot = True
    
    checkpoint_dir = Path(args.checkpoint)
    if checkpoint_dir.is_file(): checkpoint_dir = checkpoint_dir.parent
    metadata_path = checkpoint_dir / "training_metadata.json"
    
    if metadata_path.exists():
        import json
        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            
        args.modes = meta.get('modes', args.modes)
        args.width = meta.get('width', args.width)
        args.depth = meta.get('depth', args.depth)
        
        if args.train_samples is None: args.train_samples = meta.get('train_samples')
        if args.train_epochs is None: args.train_epochs = meta.get('epochs')
        if args.train_batch_size is None: args.train_batch_size = meta.get('batch_size')
        if args.train_lr is None: args.train_lr = meta.get('learning_rate')

    config = FNOConfig(
        modes_x=args.modes,
        modes_y=args.modes,
        width=args.width,
        depth=args.depth
    )
    
    model = FNO2d(config).to(device)
    
    checkpoint_path = args.checkpoint
    if Path(checkpoint_path).is_dir():
         checkpoint_path = f"{checkpoint_path}/model.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    else:
        print(f"Loading checkpoint: {checkpoint_path}")
        epoch = load_checkpoint(model, None, checkpoint_path)
        print(f"Model loaded from epoch {epoch}")
    
    print(f"Loading test data from {args.data_path}")
    test_loader, _ = load_data(args.data_path, batch_size=32)
    bench_loader, _ = load_data(args.data_path, batch_size=1)
    
    model.eval()
    myloss = LpLoss(size_average=True)
    total_l2 = 0.0
    num_batches = 0
    
    vis_a, vis_f, vis_u, vis_pred = None, None, None, None
    
    with torch.no_grad():
        for i, (a, f, u) in enumerate(test_loader):
            a, f, u = a.to(device), f.to(device), u.to(device)
            
            out = model(a, f)
            loss = myloss(out, u)
            
            total_l2 += loss.item()
            num_batches += 1
            
            if i == 0:
                n_vis = min(8, a.shape[0])
                vis_a = a[:n_vis].cpu()
                vis_f = f[:n_vis].cpu()
                vis_u = u[:n_vis].cpu()
                vis_pred = out[:n_vis].cpu()

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
    log("-" * 40)

    log(f"Model Architecture:")
    log(f"  Modes: {args.modes}")
    log(f"  Width: {args.width}")
    log(f"  Depth: {args.depth}")
    log("-" * 40)

    if args.train_samples:
        log(f"Training Samples: {args.train_samples}")
    if args.train_epochs:
        log(f"Training Epochs:  {args.train_epochs}")
    if args.train_batch_size:
        log(f"Training Batch:   {args.train_batch_size}")
    if args.train_samples or args.train_epochs:
        log("-" * 40)
    
    a_bench, f_bench, _ = next(iter(bench_loader))
    a_bench, f_bench = a_bench.to(device), f_bench.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(a_bench, f_bench)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
    latencies = []
    
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
            latencies.append((t1 - t0) * 1000)
            
    fno_avg_latency = np.mean(latencies)
    fno_std_latency = np.std(latencies)
    fno_throughput = 1000 / fno_avg_latency
    
    fno_ram = measure_memory() - start_ram
    fno_vram = 0
    if torch.cuda.is_available():
        fno_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
        
    log(f"Avg Latency:   {fno_avg_latency:.4f} ms ± {fno_std_latency:.4f}")
    log(f"Throughput:    {fno_throughput:.2f} samples/s")
    log(f"RAM Usage:     ~{fno_ram:.2f} MB (Process growth)")
    if torch.cuda.is_available():
        log(f"Peak VRAM:     {fno_vram:.2f} MB")

    if args.compare and solve_poisson is not None:
        log("\n" + "-"*40)
        log("Classical Solver (SciPy) Benchmark")
        log("-"*40)
        
        a_cpu = a_bench.cpu()
        f_cpu = f_bench.cpu()
        gen_config = DataGenConfig(grid_size=a_bench.shape[-1])
        
        for _ in range(2):
            _ = solve_poisson(a_cpu[0,0], f_cpu[0,0], gen_config)
            
        classical_latencies = []
        classical_runs = max(1, args.runs // 5)
        
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
        log("\nWarning: Could not import classical solver. Skipping comparison.")
    
    if args.output_stats:
        try:
            with open(args.output_stats, 'w') as f:
                f.write('\n'.join(stats_buffer))
            print(f"\n\"{args.output_stats}\"")
        except Exception as e:
            print(f"\nCould not save stats: {e}")

    if args.plot and vis_a is not None:
        try:
            plot_prediction(vis_a, vis_f, vis_u, vis_pred, args.output_plot)
            print(f"Plot saved to {args.output_plot}")
        except Exception as e:
            print(f"Could not save plot: {e}")


def plot_prediction(a, f, u, pred, filename):
    n_samples = a.shape[0]
    fig, axs = plt.subplots(n_samples, 4, figsize=(15, 3 * n_samples))
    
    if n_samples == 1:
        axs = axs[None, :]
    
    def plot_field(ax, data, title):
        im = ax.imshow(data.squeeze(), cmap='jet')
        if title:
            ax.set_title(title)
        plt.colorbar(im, ax=ax)
        ax.axis('off')

    for i in range(n_samples):
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
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--output_plot", type=str, default="eval_result.png")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output_stats", type=str, default="eval_stats.txt")
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--train_epochs", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--train_lr", type=float, default=None)
    
    args = parser.parse_args()
    evaluate(args)
