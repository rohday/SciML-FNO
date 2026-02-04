import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time


class LivePlotter:
    """Live training dashboard."""
    
    def __init__(self, title="FNO Training Progress"):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 6))
        self.fig.canvas.manager.set_window_title(title)
        self.start_time = time.time()
        
        self.gs = self.fig.add_gridspec(2, 3, height_ratios=[5, 1])
        
        self.ax_acc = self.fig.add_subplot(self.gs[:, :2])
        self.ax_acc.set_title("Model Accuracy (%)")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.set_ylim(0, 100)
        self.ax_acc.grid(True, linestyle='--', alpha=0.6)
        
        self.ax_info = self.fig.add_subplot(self.gs[0, 2])
        self.ax_info.axis('off')
        self.ax_info.set_title("Training Status")
        
        self.epochs = []
        self.train_accs = []
        self.val_accs = []
        
        self.line_train, = self.ax_acc.plot([], [], 'b-', label='Training', alpha=0.7)
        self.line_val, = self.ax_acc.plot([], [], 'r-', label='Validation', linewidth=2)
        self.ax_acc.legend(loc='lower right')
        
        self.best_val_acc = 0.0
        self.paused = False
        self.stop_requested = False
        
        self.ax_stop = self.fig.add_subplot(self.gs[1, 2])
        self.btn_stop = Button(self.ax_stop, 'Stop Training', color='salmon', hovercolor='red')
        self.btn_stop.on_clicked(self.stop)
        
        self.ax_stop.set_position([0.75, 0.05, 0.1, 0.075]) 
        self.ax_pause = self.fig.add_axes([0.86, 0.05, 0.1, 0.075])
        self.btn_pause = Button(self.ax_pause, 'Pause', color='lightblue', hovercolor='skyblue')
        self.btn_pause.on_clicked(self.toggle_pause)

    def stop(self, event):
        self.stop_requested = True
        print("\nStop requested...")
        
    def toggle_pause(self, event):
        self.paused = not self.paused
        self.btn_pause.label.set_text("Resume" if self.paused else "Pause")
        print(f"\n{'PAUSED' if self.paused else 'Resuming...'}")

    def update(self, epoch, train_acc, val_acc, learning_rate):
        self.epochs.append(epoch)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.best_val_acc = max(self.best_val_acc, val_acc)
        
        self.line_train.set_data(self.epochs, self.train_accs)
        self.line_val.set_data(self.epochs, self.val_accs)
        
        self.ax_acc.relim()
        self.ax_acc.autoscale_view()
        
        elapsed = time.time() - self.start_time
        info_text = (
            f"Epoch: {epoch}\n\n"
            f"Time Elapsed: {elapsed:.1f} s\n\n"
            f"Train Acc:  {train_acc:.2f}%\n"
            f"Val Acc:    {val_acc:.2f}%\n\n"
            f"Best Val:   {self.best_val_acc:.2f}%\n\n"
            f"LR:         {learning_rate:.2e}\n\n"
            "Status:     Running..."
        )
        
        self.ax_info.clear()
        self.ax_info.axis('off')
        self.ax_info.set_title("Training Status")
        self.ax_info.text(0.1, 0.9, info_text, 
                          transform=self.ax_info.transAxes, 
                          verticalalignment='top', 
                          fontsize=12, family='monospace')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def save(self, filename):
        self.fig.savefig(filename)
        print(f"Training plot saved to {filename}")

    def close(self):
        plt.ioff()
        plt.close(self.fig)
