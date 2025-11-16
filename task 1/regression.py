

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt


TABULAR_PATH = r"d:\UTS DL\Room-Climate-Datasets-master\Room-Climate-Datasets-master\Dataset"
EPOCHS = 200
BATCH_SIZE = 64
OUTPUT_DIR = "output_regression"



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class GRURegressor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()


def normalize_data(train_data, val_data, test_data):
    min_vals = train_data.min(axis=0)
    max_vals = train_data.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0

    train_norm = (train_data - min_vals) / range_vals
    val_norm = (val_data - min_vals) / range_vals
    test_norm = (test_data - min_vals) / range_vals

    return train_norm, val_norm, test_norm, min_vals, max_vals


def prepare_sequences(data, window_size=1):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        # column 0 = temperature (after slicing to 4:8)
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

        # NaN / Inf check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[train] NaN/Inf loss at batch {batch_idx}")
            bx = batch_x.detach().cpu().numpy()
            by = batch_y.detach().cpu().numpy()
            print("  batch_x min/max/mean:", np.nanmin(bx), np.nanmax(bx), np.nanmean(bx))
            print("  batch_y min/max/mean:", np.nanmin(by), np.nanmax(by), np.nanmean(by))
            return float("inf")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # strong clipping to avoid explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float("inf")


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            with autocast():
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[val] NaN/Inf loss at batch {batch_idx}")
                return float("inf")

            total_loss += float(loss.item())
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float("inf")


def get_predictions(model, loader, device):
    """Return predictions and ground-truth labels (normalized)."""
    model.eval()
    preds = []
    acts = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            with autocast():
                outputs = model(batch_x)
            preds.extend(outputs.cpu().numpy())
            acts.extend(batch_y.numpy())

    return np.array(preds), np.array(acts)


def main():
    set_seed(42)

    tabular_path = TABULAR_PATH
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    if os.path.isfile(tabular_path):
        df = pd.read_csv(tabular_path)
        print(f"Loaded single CSV: {tabular_path}  rows={len(df)}")
    else:
        csv_files = sorted([f for f in os.listdir(tabular_path) if f.lower().endswith(".csv")])
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in folder: {tabular_path}")
        df_list = []
        for f in csv_files:
            path = os.path.join(tabular_path, f)
            temp_df = pd.read_csv(path)
            df_list.append(temp_df)
            print(f"Loaded CSV: {path}  rows={len(temp_df)}")
        df = pd.concat(df_list, ignore_index=True)
        print(f"Final concatenated dataset size: {len(df)} rows from {len(csv_files)} CSV files.")
 
    data = df.iloc[:, 4:8].values.astype(np.float32)


    finite_mask = np.isfinite(data)
    col_means = np.nanmean(np.where(finite_mask, data, np.nan), axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    data = np.where(finite_mask, data, col_means)

    print("DATA STATS (per column)  min | max | mean | std")
    print(
        np.nanmin(data, axis=0),
        np.nanmax(data, axis=0),
        np.nanmean(data, axis=0),
        np.nanstd(data, axis=0),
    )
  
    N = len(data)
    train_end = int(0.7 * N)
    val_end = int(0.85 * N)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    train_norm, val_norm, test_norm, min_vals, max_vals = normalize_data(
        train_data, val_data, test_data
    )

   
    X_train, y_train = prepare_sequences(train_norm, window_size=1)
    X_val, y_val = prepare_sequences(val_norm, window_size=1)
    X_test, y_test = prepare_sequences(test_norm, window_size=1)
   

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = GRURegressor(input_size=4, hidden_size=64, num_layers=2).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = GradScaler()

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss = validate(model, val_loader, criterion, device)

        try:
            scheduler.step()
        except Exception:
            pass

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        
        if (
            np.isnan(train_loss)
            or np.isinf(train_loss)
            or np.isnan(val_loss)
            or np.isinf(val_loss)
        ):
            print(
                f"Epoch {epoch + 1}/{epochs} produced NaN/Inf loss. "
                f"Reducing LR and skipping checkpoint."
            )
            for g in optimizer.param_groups:
                g["lr"] = g.get("lr", 1e-4) * 0.5
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "checkpoints", "last_model.pt"),
            )
        else:
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, "checkpoints", "best_model.pt"),
                )

            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "checkpoints", "last_model.pt"),
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

    
    best_path = os.path.join(output_dir, "checkpoints", "best_model.pt")
    last_path = os.path.join(output_dir, "checkpoints", "last_model.pt")

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    elif os.path.exists(last_path):
        print("best_model.pt not found — loading last_model.pt instead.")
        model.load_state_dict(torch.load(last_path, map_location=device))
    else:
        print("No checkpoint found; evaluating with current model weights.")
    

    
    temp_min = float(min_vals[0])
    temp_max = float(max_vals[0])

    
    predictions, actuals = get_predictions(model, test_loader, device)
    predictions_denorm = predictions * (temp_max - temp_min) + temp_min
    actuals_denorm = actuals * (temp_max - temp_min) + temp_min

    mae = np.mean(np.abs(predictions_denorm - actuals_denorm))
    rmse = np.sqrt(np.mean((predictions_denorm - actuals_denorm) ** 2))

    print(f"Test MAE: {mae:.6f}")
    print(f"Test RMSE: {rmse:.6f}")

    
    y_true = y_test
    y_pred_naive = X_test[:, 0, 0]  

    pred_naive_denorm = y_pred_naive * (temp_max - temp_min) + temp_min
    true_denorm = y_true * (temp_max - temp_min) + temp_min

    mae_naive = float(np.mean(np.abs(pred_naive_denorm - true_denorm)))
    rmse_naive = float(np.sqrt(np.mean((pred_naive_denorm - true_denorm) ** 2)))

    print(f"Naive Baseline MAE: {mae_naive:.6f}")
    print(f"Naive Baseline RMSE: {rmse_naive:.6f}")
   
    n = min(200, len(predictions_denorm))
    t_axis = np.arange(n)

    plt.figure(figsize=(12, 5))
    plt.plot(t_axis, actuals_denorm[:n], label="True Temp")
    plt.plot(t_axis, predictions_denorm[:n], label="Pred Temp")
    plt.title("Prediction vs Ground Truth (first 200 test samples)")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_v_groundtruth.png"))
    plt.close()

    residuals = (actuals_denorm - predictions_denorm)
    plt.figure(figsize=(12, 4))
    plt.plot(t_axis, residuals[:n])
    plt.title("Residual Plot (True - Pred) first 200 samples")
    plt.xlabel("Time Step")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals.png"))
    plt.close()

    print("Saved diagnostic plots:")
    print(" - prediction_v_groundtruth.png")
    print(" - residuals.png")
    

    temp_test = test_norm[:, 0]  
    diffs = temp_test[1:] - temp_test[:-1]
    print("Test temp diff stats (normalized): min, max, mean, std")
    print(diffs.min(), diffs.max(), diffs.mean(), diffs.std())

    
    np.save(os.path.join(output_dir, "logs", "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(output_dir, "logs", "val_losses.npy"), np.array(val_losses))

    epochs_axis = np.arange(1, epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_axis, train_losses, label="Train Loss")
    plt.plot(epochs_axis, val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_val_loss.png"))
    plt.close()


if __name__ == "__main__":
    main()
