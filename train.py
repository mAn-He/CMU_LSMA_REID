import torch
import torch.nn as nn
from tqdm import tqdm
# from loss import ContrastiveLoss  # loss.py에서 ContrastiveLoss 가져오기
from model import ResNetFeatureExtractor  # model.py에서 모델 가져오기
import wandb
from config import CONFIG
from loss import SupConLoss  # SupConLoss로 변경

class SupervisedContrastiveEngine:
    def __init__(self, train_loader, val_loader, model, optimizer, scheduler=None, use_gpu=True):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.model.to(self.device)
        self.contrastive_loss = SupConLoss(temperature=0.07)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def train(self, num_epochs, save_path='best_model_color.pth'):
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.run_epoch(epoch, num_epochs, train=True)

            # Validation phase
            val_loss = self.validate(epoch, num_epochs)

            # Save model if it's the best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model at epoch {epoch+1} with loss {best_val_loss:.4f}")

            # Log metrics
            if 'wandb' in globals():
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch+1})

    def validate(self, epoch, num_epochs):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            val_progress = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
            for original, total_images, labels in val_progress:
                original, total_images, labels = (
                    original.to(self.device),
                    [img.to(self.device) for img in total_images],
                    labels.to(self.device),
                )

                # Forward pass
                features, logits = self.model(original)
                features_total = [self.model(img)[0] for img in total_images]

                # Compute losses
                contrastive_loss = self.contrastive_loss(features, labels)
                cross_entropy_loss = self.cross_entropy_loss(logits, labels)
                total_loss += contrastive_loss.item() + cross_entropy_loss.item()

                val_progress.set_postfix(loss=f"{(contrastive_loss + cross_entropy_loss).item():.4f}")

        avg_loss = total_loss / len(self.val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def run_epoch(self, epoch, num_epochs, train=True):
        loader = self.train_loader if train else self.val_loader
        self.model.train() if train else self.model.eval()
        total_loss = 0
        phase = "Train" if train else "Val"

        with torch.set_grad_enabled(train):
            progress = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [{phase}]")
            for original, total_images, labels in progress:
                loss = self.process_batch(original, total_images, labels, train=train)
                total_loss += loss
                progress.set_postfix(loss=f"{loss:.4f}")

        avg_loss = total_loss / len(loader)
        return avg_loss

    def process_batch(self, original, total_images, labels, train=True):
        # Move to device
        original, total_images, labels = (
            original.to(self.device),
            [img.to(self.device) for img in total_images],
            labels.to(self.device),
        )

        # Forward pass
        # print(f"Original input shape: {original.shape}, dtype: {original.dtype}, device: {original.device}")
        # print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}, min: {labels.min()}, max: {labels.max()}")

        # Ensure labels are within range and correct dtype
        if labels.dtype != torch.long:
            labels = labels.long()
        assert labels.min() >= 0, "Labels contain negative values"
        assert labels.max() < self.model.classifier[-1].out_features, f"Label value {labels.max()} exceeds number of classes"

        # Forward pass
        features, logits = self.model(original)
        # print(f"Logits shape: {logits.shape}")
        features_total = [self.model(img)[0] for img in total_images]

        # Compute losses
        contrastive_loss = self.contrastive_loss(features, labels)
        cross_entropy_loss = self.cross_entropy_loss(logits, labels)
        total_loss = contrastive_loss + cross_entropy_loss

        if train:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return total_loss.item()
