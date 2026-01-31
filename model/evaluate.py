import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model.config import FNOConfig
from model.fno2d import FNO2d
from model.utils import load_data, GaussianNormalizer, load_checkpoint
from model.train import LpLoss

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
             raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    epoch = load_checkpoint(model, None, checkpoint_path) # Optimizer=None for inference
    print(f"Model loaded from epoch {epoch}")
    
    # 4. Data Loading
    print(f"Loading test data from {args.data_path}")
    test_loader, _ = load_data(args.data_path, batch_size=32)
    
    # 5. Evaluation Loop
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

    avg_l2 = total_l2 / num_batches
    print(f"\nResults on Test Set:")
    print(f"====================")
    print(f"Average Relative L2 Error: {avg_l2:.5f}")
    print(f"Accuracy: {(1 - avg_l2)*100:.2f}% (Approx)")
    
    # 6. Visualization (Optional)
    if args.plot:
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
    parser.add_argument("--modes", type=int, default=12) # Reverted default
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--plot", action='store_true', help="Save a visualization plot")
    parser.add_argument("--output_plot", type=str, default="eval_result.png")
    
    args = parser.parse_args()
    evaluate(args)
