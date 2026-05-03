import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import os

RECORD_IDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
    122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
    209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
    222, 223, 228, 230, 231, 232, 233, 234
]

LABEL_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4,
}

SEGMENT_LENGTH = 187
CLASS_NAMES = ['Normal (N)', 'SVEB (A)', 'VEB (V)', 'Fusion (F)', 'Unknown (Q)']


def download_and_extract(data_dir='mitdb/'):
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print("Local MIT-BIH data not found. Downloading from PhysioNet...")
        os.makedirs(data_dir, exist_ok=True)
        wfdb.dl_database('mitdb', dl_dir=data_dir)
        print("Download complete.")
    else:
        print(f"Using local MIT-BIH data from '{data_dir}'")

    all_segments = []
    all_labels = []

    for record_id in RECORD_IDS:
        record_path = os.path.join(data_dir, str(record_id))

        if not os.path.exists(record_path + '.hea'):
            print(f"  Missing record {record_id}, skipping...")
            continue

        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            signal = record.p_signal[:, 0]
            r_peaks = annotation.sample
            symbols = annotation.symbol

            for peak, symbol in zip(r_peaks, symbols):
                if symbol not in LABEL_MAP:
                    continue

                start = peak - 90
                end = peak + 97

                if start < 0 or end > len(signal):
                    continue

                segment = signal[start:end]
                if len(segment) != SEGMENT_LENGTH:
                    continue

                all_segments.append(segment)
                all_labels.append(LABEL_MAP[symbol])

        except Exception as e:
            print(f"  Error reading record {record_id}: {e}")
            continue

    segments = np.array(all_segments, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    print(f"\nTotal segments extracted: {len(segments)}")
    for i, name in enumerate(CLASS_NAMES):
        count = (labels == i).sum()
        pct = count / len(labels) * 100
        print(f"  Class {i} ({name}): {count} ({pct:.1f}%)")

    return segments, labels


def normalize(segments):
    mean = segments.mean(axis=1, keepdims=True)
    std = segments.std(axis=1, keepdims=True) + 1e-8
    return (segments - mean) / std


class ECGDataset(Dataset):
    def __init__(self, segments, labels):
        self.X = torch.FloatTensor(segments).unsqueeze(1)
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(segments, labels, batch_size=64):
    X_temp, X_test, y_temp, y_test = train_test_split(
        segments, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
    )

    train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ECGDataset(X_val,   y_val),   batch_size=batch_size)
    test_loader  = DataLoader(ECGDataset(X_test,  y_test),  batch_size=batch_size)

    return train_loader, val_loader, test_loader, (y_train, y_val, y_test)