import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.bn(self.conv(x))))


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv_block1 = ConvBlock(1, 32, kernel_size=5)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv_block2 = ConvBlock(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv_block3 = ConvBlock(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.lstm = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.attention = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool1(self.conv_block1(x))
        x = self.pool2(self.conv_block2(x))
        x = self.pool3(self.conv_block3(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        x = (lstm_out * attn_weights).sum(dim=1)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)