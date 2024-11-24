import torch
from tqdm import tqdm
# from loss import ContrastiveLoss
import wandb
from config import CONFIG
from loss import SupConLoss  # SupConLoss로 변경
class ContrastivePhase:
    def __init__(self, train_loader, val_loader, model, optimizer, scheduler=None, device="cuda"):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn = SupConLoss(temperature=0.07)  # SupConLoss 사용

    def train(self, num_epochs, save_path="encoder_weights.pth"):
        best_val_loss = float("inf")
        for epoch in range(num_epochs):
            train_loss = self.run_epoch(epoch, num_epochs, train=True)
            val_loss = self.run_epoch(epoch, num_epochs, train=False)

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best encoder at epoch {epoch + 1} with val loss {best_val_loss:.4f}")

            # Log metrics
            wandb.log({"phase": "contrastive", "train_loss": train_loss, "val_loss": val_loss, "epoch": epoch + 1})

    def run_epoch(self, epoch, num_epochs, train=True):
        phase = "Train" if train else "Validation"
        loader = self.train_loader if train else self.val_loader
        self.model.train() if train else self.model.eval()
        total_loss = 0

        with torch.set_grad_enabled(train):
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs} [{phase}]")
            for batch in progress:
                original, total_images, labels = batch
                original = original.to(self.device)
                labels = labels.to(self.device)

                # Reshape total_images to combine batch and view dimensions
                total_images = total_images.view(-1, *total_images.shape[2:]).to(self.device)  # Shape: [batch_size * n_views, 3, 256, 128]

                # Correctly expand labels to match reshaped total_images
                total_labels = labels.repeat_interleave(CONFIG["n_views"] + CONFIG["num_inpainted"]).to(self.device)

                # Forward pass through the model
                features = self.model(total_images)  # Shape: [batch_size * n_views, feature_dim]
                
                # Reshape features to [batch_size, n_views, feature_dim]
                features = features.view(labels.size(0), -1, features.size(1))

                # Compute Contrastive Loss
                loss = self.loss_fn(features, labels)  # SupConLoss expects [batch_size, n_views, feature_dim]
                    
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{num_epochs} [{phase}] Loss: {avg_loss:.4f}")
        return avg_loss
