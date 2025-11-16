

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

TABULAR_PATH = r"D:\UTS DL\Room-Climate-Datasets-master\Room-Climate-Datasets-master\datasets-location_A\room_climate-location_A-measurement01.csv"
OUTPUT_DIR = "output_regression"  
CHECKPOINT_NAME = "best_model.pt"  
BATCH_SIZE = 64



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
        
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)


def load_and_prepare_data(tabular_path):
    
    if os.path.isfile(tabular_path):
        df = pd.read_csv(tabular_path)
        print(f"[TEST] Loaded single CSV: {tabular_path}  rows={len(df)}")
    else:
        csv_files = sorted([f for f in os.listdir(tabular_path) if f.lower().endswith(".csv")])
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in folder: {tabular_path}")
        df_list = []
        for f in csv_files:
            path = os.path.join(tabular_path, f)
            temp_df = pd.read_csv(path)
            df_list.append(temp_df)
            print(f"[TEST] Loaded CSV: {path}  rows={len(temp_df)}")
        df = pd.concat(df_list, ignore_index=True)
        print(f"[TEST] Final concatenated dataset size: {len(df)} rows from {len(csv_files)} CSV files.")

    
    data = df.iloc[:, 4:8].values.astype(np.float32)

    
    finite_mask = np.isfinite(data)
    col_means = np.nanmean(np.where(finite_mask, data, np.nan), axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    data = np.where(finite_mask, data, col_means)

    print("[TEST] DATA STATS (per column)  min | max | mean | std")
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

    return (X_train, y_train, X_val, y_val, X_test, y_test, min_vals, max_vals)


def evaluate_model(model, loader, device, temp_min, temp_max):
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

    preds = np.array(preds)
    acts = np.array(acts)

    preds_denorm = preds * (temp_max - temp_min) + temp_min
    acts_denorm = acts * (temp_max - temp_min) + temp_min

    mae = np.mean(np.abs(preds_denorm - acts_denorm))
    rmse = np.sqrt(np.mean((preds_denorm - acts_denorm) ** 2))

    return mae, rmse, preds_denorm, acts_denorm


def main():
    set_seed(42)

    X_train, y_train, X_val, y_val, X_test, y_test, min_vals, max_vals = load_and_prepare_data(TABULAR_PATH)

    
    temp_min = float(min_vals[0])
    temp_max = float(max_vals[0])

    test_dataset = TimeSeriesDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[TEST] Using device:", device)

    
    model = GRURegressor(input_size=4, hidden_size=64, num_layers=2).to(device)

    
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", CHECKPOINT_NAME)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[TEST] Loaded checkpoint: {ckpt_path}")

    
    mae, rmse, preds_denorm, acts_denorm = evaluate_model(model, test_loader, device, temp_min, temp_max)
    print(f"[TEST] GRU Test MAE: {mae:.6f}")
    print(f"[TEST] GRU Test RMSE: {rmse:.6f}")

    
    y_true = y_test  
    y_pred_naive = X_test[:, 0, 0]  

    pred_naive_denorm = y_pred_naive * (temp_max - temp_min) + temp_min
    true_denorm = y_true * (temp_max - temp_min) + temp_min

    mae_naive = float(np.mean(np.abs(pred_naive_denorm - true_denorm)))
    rmse_naive = float(np.sqrt(np.mean((pred_naive_denorm - true_denorm) ** 2)))

    print(f"[TEST] Naive Baseline MAE: {mae_naive:.6f}")
    print(f"[TEST] Naive Baseline RMSE: {rmse_naive:.6f}")


if __name__ == "__main__":
    main()
