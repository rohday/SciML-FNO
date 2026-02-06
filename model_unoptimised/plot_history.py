import matplotlib.pyplot as plt
import json
import argparse


def plot_history(history_path, output_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
        
    epochs = history['epochs']
    train_loss = history['train_loss']
    test_loss = history['test_loss']
    
    train_acc = [(1 - l) * 100 for l in train_loss]
    test_acc = [(1 - l) * 100 for l in test_loss]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(epochs, train_acc, label='Training Accuracy', color='tab:blue')
    ax1.plot(epochs, test_acc, label='Validation Accuracy', color='tab:orange')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training Progress')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(epochs, train_loss, label='Training Loss (Rel L2)', color='tab:blue')
    ax2.plot(epochs, test_loss, label='Validation Loss (Rel L2)', color='tab:orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (Relative L2)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, default="checkpoints/history.json")
    parser.add_argument("--output", type=str, default="training_plot.png")
    args = parser.parse_args()
    
    try:
        plot_history(args.history, args.output)
    except FileNotFoundError:
        print(f"Error: History file not found at {args.history}")
