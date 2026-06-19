import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from net import TinyUNet

class SyntheticLaneDataset(Dataset):
    """Generates synthetic road and lane images on-the-fly for simulation training."""
    def __init__(self, num_samples=200):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create road background (grayish)
        img = np.ones((180, 320, 3), dtype=np.uint8) * 40
        # Add background sky/horizon (top 70 pixels)
        img[0:70, :] = [120, 100, 90]
        
        # Draw road polygon
        road_pts = np.array([[0, 180], [130, 70], [190, 70], [320, 180]], dtype=np.int32)
        cv2.fillPoly(img, [road_pts], (80, 80, 80))
        
        mask = np.zeros((180, 320), dtype=np.uint8)
        
        # Randomize curvature and horizontal offset
        curve = np.random.uniform(-40, 40)
        offset = np.random.uniform(-20, 20)
        
        # Draw curved lane boundaries using quadratic Bezier curves
        # Left lane
        l_start = (int(40 + offset), 180)
        l_end = (int(145 + offset + curve), 70)
        l_control = (int(90 + offset + curve * 0.5), 125)
        
        # Right lane
        r_start = (int(280 + offset), 180)
        r_end = (int(175 + offset + curve), 70)
        r_control = (int(230 + offset + curve * 0.5), 125)
        
        # Plot points
        pts = np.linspace(0, 1, 20)
        l_curve_pts = np.array([
            (1 - t)**2 * np.array(l_start) + 2 * (1 - t) * t * np.array(l_control) + t**2 * np.array(l_end)
            for t in pts
        ], dtype=np.int32)
        
        r_curve_pts = np.array([
            (1 - t)**2 * np.array(r_start) + 2 * (1 - t) * t * np.array(r_control) + t**2 * np.array(r_end)
            for t in pts
        ], dtype=np.int32)
        
        # Draw curves on image and mask
        cv2.polylines(img, [l_curve_pts], isClosed=False, color=(255, 255, 255), thickness=4)
        cv2.polylines(mask, [l_curve_pts], isClosed=False, color=255, thickness=4)
        
        cv2.polylines(img, [r_curve_pts], isClosed=False, color=(255, 255, 255), thickness=4)
        cv2.polylines(mask, [r_curve_pts], isClosed=False, color=255, thickness=4)
        
        # Add random noise and brightness variations
        noise = np.random.randint(-15, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Normalize and convert to tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return img_tensor, mask_tensor

class DiceLoss(nn.Module):
    """Dice Loss to handle highly unbalanced class distribution of lane pixels."""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        intersection = torch.sum(predict * target)
        union = torch.sum(predict) + torch.sum(target)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

def train_model(epochs=10, batch_size=16, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Initialize model, loss, and optimizer
    model = TinyUNet().to(device)
    bce_loss = nn.BCELoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize datasets
    train_dataset = SyntheticLaneDataset(num_samples=400)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = SyntheticLaneDataset(num_samples=80)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    # Create output directories if needed
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth")
    torch.save(model.state_dict(), weight_path)
    print(f"Model successfully saved to {weight_path}")

if __name__ == "__main__":
    train_model(epochs=5, batch_size=16)
