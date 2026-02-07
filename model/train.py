import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import matplotlib.pyplot as plt
from pathlib import Path

from model.config import FNOConfig
from model.fno2d import FNO2d
from model.fno2d import FNO2d
from model.utils import load_data, GaussianNormalizer, save_checkpoint
try:
    from model.plotter import LivePlotter
except ImportError:
    LivePlotter = None


class LpLoss:
    """Relative L2 loss: ||pred - true||_2 / ||true||_2"""
    
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def __call__(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.view(num_examples,-1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    config = FNOConfig(
        modes_x=args.modes,
        modes_y=args.modes,
        width=args.width,
        depth=args.depth
    )
    
    model = FNO2d(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params}")
    
    # Training speedup optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for matmuls (Ada Lovelace)
        torch.backends.cudnn.allow_tf32 = True  # TF32 for cuDNN
    
    # Note: torch.compile() disabled - doesn't support complex64 tensors in FFT
    
    if args.dry_run:
        print("Dry Run: Generating synthetic data...")
        N, H, W = 16, 64, 64
        train_loader = [(
            torch.randn(N, 1, H, W),
            torch.randn(N, 1, H, W),
            torch.randn(N, 1, H, W)
        )]
        test_loader = train_loader
        y_normalizer = GaussianNormalizer(torch.zeros(1))
        y_normalizer.mean = 0.0
        y_normalizer.std = 1.0
    else:
        if not args.data_path:
             raise ValueError("data_path must be specified when not in dry-run mode.")
        
        print(f"Loading data from {args.data_path}")
        train_loader, train_stats = load_data(args.data_path, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = train_loader
        
        all_u = train_loader.dataset.tensors[2]
        y_normalizer = GaussianNormalizer(all_u)
        
        if torch.cuda.is_available():
            y_normalizer.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    myloss = LpLoss(size_average=True)
    
    best_test_l2 = float('inf')
    history = {'train_loss': [], 'test_loss': [], 'epochs': []}
    patience_counter = 0
    best_loss_for_stopping = float('inf')
    
    plotter = None
    if args.plot and LivePlotter:
        plotter = LivePlotter(title="FNO Training Accuracy")
        print("Live plotting started.")
    
    for ep in range(args.epochs):
        model.train()
        t1 = time.time()
        train_l2 = 0
        
        for batch_idx, (a, f, u) in enumerate(train_loader):
            a, f, u = a.to(device, non_blocking=True), f.to(device, non_blocking=True), u.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            out = model(a, f)
            loss = myloss(out, u)
            loss.backward()
            
            # Gradient clipping for training stability
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            
            optimizer.step()
            
            train_l2 += loss.item()
            
            if args.dry_run and batch_idx >= 2: break

        scheduler.step()
        train_l2 /= len(train_loader)
        
        # Only evaluate every N epochs (or on last epoch) for speed
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            model.eval()
            test_l2 = 0.0
            with torch.no_grad():
                for a, f, u in test_loader:
                    a, f, u = a.to(device), f.to(device), u.to(device)
                    out = model(a, f)
                    test_l2 += myloss(out, u).item()
                    if args.dry_run: break
                    
            test_l2 /= len(test_loader)
            last_test_l2 = test_l2
        else:
            test_l2 = last_test_l2 if 'last_test_l2' in dir() else float('inf')
        
        t2 = time.time()
        print(f"Epoch {ep}: Train L2={train_l2:.5f}, Test L2={test_l2:.5f}, Time={t2-t1:.2f}s")
        
        if test_l2 < best_test_l2:
            best_test_l2 = test_l2
            save_checkpoint(model, optimizer, ep, f"{args.output_dir}/model.pth")
            
        history['train_loss'].append(train_l2)
        history['test_loss'].append(test_l2)
        history['epochs'].append(ep)

        if test_l2 < best_loss_for_stopping:
            best_loss_for_stopping = test_l2
            patience_counter = 0
        else:
            patience_counter += 1
            
        train_acc = (1 - train_l2) * 100
        test_acc = (1 - test_l2) * 100

        if plotter:
            plotter.update(ep, train_acc, test_acc, optimizer.param_groups[0]['lr'])
            
            if plotter.stop_requested:
                print(f"\nTraining stopped by user at epoch {ep}!")
                break
                
            while plotter.paused:
                plt.pause(0.5)
                if plotter.stop_requested: break
            
            if plotter.stop_requested: break
    
    if plotter:
        plotter.save(f"{args.output_dir}/training_curve.png")
        plotter.close()

    import json
    with open(f"{args.output_dir}/history.json", 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {args.output_dir}/history.json")

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

    # Auto-evaluation
    print("\n" + "="*50)
    print("Starting Auto-Evaluation...")
    print("="*50)
    
    import subprocess
    import sys
    
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
    
    print(f"Running: {' '.join(eval_cmd)}")
    try:
        subprocess.run(eval_cmd, check=True)
        print("Auto-evaluation completed.")
    except subprocess.CalledProcessError as e:
        print(f"Auto-evaluation failed: {e.returncode}")
    except Exception as e:
        print(f"Auto-evaluation failed: {e}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/poisson.npz")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Run with synthetic data")
    parser.add_argument("--plot", action="store_true", help="Show live training dashboard")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--grad_clip", type=float, default=0.25, help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate every N epochs (default 10)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (default 4)")
    
    args = parser.parse_args()
    train(args)
