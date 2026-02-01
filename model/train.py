import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import matplotlib.pyplot as plt # For plt.pause
from pathlib import Path

from model.config import FNOConfig
from model.fno2d import FNO2d
from model.fno2d import FNO2d
from model.utils import load_data, GaussianNormalizer, save_checkpoint
try:
    from model.plotter import LivePlotter
except ImportError:
    LivePlotter = None

# --- Relative L2 Loss ---

class LpLoss(object):
    """
    Relative L2 (Lp with p=2) error.
    Loss = ||y_pred - y_true||_2 / ||y_true||_2
    """
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        # Reduction dimension: All dimensions except batch (dim 0)
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def __call__(self, x, y):
        # x, y shape: (batch, ..., ...)
        # Norm over all dimensions except batch
        num_examples = x.size()[0]
        
        # Calculate L2 norms
        # We flatten everything after batch dim to compute norm easily
        diff_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.view(num_examples,-1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        
        return diff_norms / y_norms

# --- Training Loop ---

def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = FNOConfig(
        modes_x=args.modes,
        modes_y=args.modes,
        width=args.width,
        depth=args.depth
    )
    
    # Model
    model = FNO2d(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params}")
    
    # Data Loading (Assuming data is already generated)
    # If using dry-run, we might not have data, so generate synthetic batch
    if args.dry_run:
        print("Dry Run: Generating synthetic data...")
        # (batch, 1, 64, 64)
        N, H, W = 16, 64, 64
        train_loader = [(
            torch.randn(N, 1, H, W), # a
            torch.randn(N, 1, H, W), # f
            torch.randn(N, 1, H, W)  # u (target)
        )]
        test_loader = train_loader
        # Fake normalizers
        y_normalizer = GaussianNormalizer(torch.zeros(1))
        y_normalizer.mean = 0.0
        y_normalizer.std = 1.0
    else:
        if not args.data_path:
             raise ValueError("data_path must be specified when not in dry-run mode.")
        
        print(f"Loading data from {args.data_path}")
        # Load Data
        train_loader, train_stats = load_data(args.data_path, batch_size=args.batch_size)
        test_loader = train_loader # TODO: Allow separate test file
        
        # Calculate normalizer from training data
        # We need to get all 'u' data to compute mean/std
        # Since DataLoaders are lazy, let's load full tensor for normalization init
        # Ideally we save mean/std in dataset metadata, but calculating on fly is okay for now
        # WARNING: This loads entire dataset into RAM. For Edge, we should avoid this.
        # But for training on desktop, it's fine.
        
        # Quick hack to get data for normalizer
        all_u = train_loader.dataset.tensors[2]
        y_normalizer = GaussianNormalizer(all_u)
        
        if torch.cuda.is_available():
            y_normalizer.cuda()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss
    myloss = LpLoss(size_average=True)
    
    # Training Loop
    best_test_l2 = float('inf')
    history = {'train_loss': [], 'test_loss': [], 'epochs': []}
    
    # Early Stopping Variables
    patience_counter = 0
    best_loss_for_stopping = float('inf')
    
    # Initialize Plotter
    plotter = None
    if args.plot and LivePlotter:
        plotter = LivePlotter(title="FNO Training Accuracy")
        print("Live plotting started.")
    
    for ep in range(args.epochs):
        model.train()
        t1 = time.time()
        train_l2 = 0
        
        for batch_idx, (a, f, u) in enumerate(train_loader):
            a, f, u = a.to(device), f.to(device), u.to(device)
            
            optimizer.zero_grad()
            
            # Forward: u_pred = Model(a, f)
            # Physics context: Mapping medium 'a' and source 'f' to field 'u'
            out = model(a, f)
            
            # We usually normalize output for training stability, then denormalize for loss
            # But specific implementation detail: Simple approach vs normalized approach
            # Here we assume data coming in is already somewhat normalized or we learn direct mapping
            # Standard FNO uses normalized targets (y_normalizer.encode(y))
            # Let's train to predict parameterized output directly for simplicity now
            
            loss = myloss(out, u)
            loss.backward()
            optimizer.step()
            
            train_l2 += loss.item()
            
            if args.dry_run and batch_idx >= 2: break # Short loop for dry run

        scheduler.step()
        
        train_l2 /= len(train_loader)
        t2 = time.time()
        
        # Validation
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for a, f, u in test_loader:
                a, f, u = a.to(device), f.to(device), u.to(device)
                out = model(a, f)
                test_l2 += myloss(out, u).item()
                if args.dry_run: break
                
        test_l2 /= len(test_loader)
        
        print(f"Epoch {ep}: Train L2={train_l2:.5f}, Test L2={test_l2:.5f}, Time={t2-t1:.2f}s")
        
        if test_l2 < best_test_l2:
            best_test_l2 = test_l2
            save_checkpoint(model, optimizer, ep, f"{args.output_dir}/model.pth")
            
        # Update history
        history['train_loss'].append(train_l2)
        history['test_loss'].append(test_l2)
        history['epochs'].append(ep)

        # Early Stopping Logic (Monitor Test Loss)
        if test_l2 < best_loss_for_stopping:
            best_loss_for_stopping = test_l2
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Converter to Accuracy (%)
        train_acc = (1 - train_l2) * 100
        test_acc = (1 - test_l2) * 100

        # Update Plot & UI Checks
        if plotter:
            plotter.update(ep, train_acc, test_acc, optimizer.param_groups[0]['lr'])
            
            # check stopping
            if plotter.stop_requested:
                print(f"\nTraining stopped by user at epoch {ep}!")
                break
                
            # check pausing
            while plotter.paused:
                plt.pause(0.5) # Keep UI alive
                if plotter.stop_requested: # Allow stopping while paused
                    break
            
            if plotter.stop_requested:
                 break

            
        # Early Stop Check
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {ep}! No improvement for {args.patience} epochs.")
            break
    
    if plotter:
        plotter.save(f"{args.output_dir}/training_curve.png")
        plotter.close()

    # Save history
    import json
    with open(f"{args.output_dir}/history.json", 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {args.output_dir}/history.json")

    # Save Metadata for standalone evaluation
    metadata = {
        "train_samples": len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 16,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "modes": args.modes,
        "width": args.width,
        "depth": args.depth
    }
    with open(f"{args.output_dir}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Training metadata saved to {args.output_dir}/training_metadata.json")

    # --- Auto-Evaluation ---
    print("\n" + "="*50)
    print("Starting Auto-Evaluation...")
    print("="*50)
    
    import subprocess
    import sys
    
    # Construct command
    eval_cmd = [
        sys.executable, "model/evaluate.py",
        "--data_path", args.data_path,
        "--checkpoint", args.output_dir,
        "--modes", str(args.modes),
        "--width", str(args.width),
        "--depth", str(args.depth),
        "--plot",
        "--output_plot", "eval_result.png",
        "--output_stats", "eval_stats.txt",
        "--train_samples", str(len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 16),
        "--train_epochs", str(args.epochs),
        "--train_batch_size", str(args.batch_size),
        "--train_lr", str(args.learning_rate)
    ]
    
    print(f"Running command: {' '.join(eval_cmd)}")
    try:
        subprocess.run(eval_cmd, check=True)
        print("Auto-evaluation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Auto-evaluation failed with error code {e.returncode}")
    except Exception as e:
        print(f"Auto-evaluation failed: {e}")

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/poisson.npz")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    parser.add_argument("--dry-run", action="store_true", help="Run with synthetic data for testing")
    parser.add_argument("--plot", action="store_true", help="Show live training dashboard")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience epochs")
    
    args = parser.parse_args()
    train(args)
