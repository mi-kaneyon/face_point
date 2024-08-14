import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# データセットクラス
class LandmarkDataset(Dataset):
    def __init__(self, landmark_dir):
        self.landmark_dir = landmark_dir
        self.landmark_files = os.listdir(landmark_dir)

    def __len__(self):
        return len(self.landmark_files)

    def __getitem__(self, idx):
        landmark_path = os.path.join(self.landmark_dir, self.landmark_files[idx])
        landmarks = np.loadtxt(landmark_path, delimiter=',').flatten()  # 1次元配列に変換
        return torch.tensor(landmarks, dtype=torch.float32)

# モデルの定義
class SimpleLandmarkModel(nn.Module):
    def __init__(self):
        super(SimpleLandmarkModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(136, 128),  # 入力サイズを136に設定 (68ランドマーク * x, y)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 136)  # 68ランドマーク * x, y の出力サイズ
        )

    def forward(self, x):
        return self.fc(x)

# トレーニング関数
def train_model(train_loader, model, criterion, optimizer, num_epochs=1200):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "data/models/custom_landmark_model.pth")
    print("Model trained and saved to data/models/custom_landmark_model.pth")

# メインスクリプト
if __name__ == "__main__":
    dataset = LandmarkDataset("data/landmarks")
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleLandmarkModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0007)

    train_model(train_loader, model, criterion, optimizer)
