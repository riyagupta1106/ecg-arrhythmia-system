import os
import torch
from src.datasets import download_and_extract, normalize, get_dataloaders
from src.model import CNNLSTM, count_parameters
from src.train import train, plot_training
from src.evaluate import full_evaluation

os.makedirs('models', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load data
print("\n[1/4] Loading data...")
segments, labels = download_and_extract(data_dir='mitdb/')
segments = normalize(segments)
train_loader, val_loader, test_loader, (y_train, y_val, y_test) = get_dataloaders(
    segments, labels, batch_size=64
)

# 2. Build model
print("\n[2/4] Building model...")
model = CNNLSTM(num_classes=5).to(device)
print(f"Trainable parameters: {count_parameters(model):,}")

# 3. Train
print("\n[3/4] Training...")
history = train(model, train_loader, val_loader, y_train, device, epochs=100)
plot_training(history)

# 4. Evaluate
print("\n[4/4] Evaluating best model on test set...")
model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
full_evaluation(model, test_loader, device)