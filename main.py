# main.py
import torch
from accelerate import Accelerator
import wandb
from dataset import CustomDataManager
from model import ResNetFeatureExtractor
from train import SupervisedContrastiveEngine
from contrastive_phase import ContrastivePhase
from config import CONFIG
# Configuration


# Initialize WandB

wandb.login(key="4479360fe28288a4508a4ee8b76be303493e3ab1") 
wandb.init(project=CONFIG["project_name"], config=CONFIG)
config = wandb.config

# Data paths
ORIGINAL_DIR = '/home/fisher/fisher/Peoples/hseung/카네기/LSMA/Project/ReIDataset/Market-1501/bounding_box_train'
MASK_DIR = '/home/fisher/fisher/Peoples/hseung/카네기/LSMA/Project/LSMA_trial/new_one/sam2_masks'
INPAINTED_DIR = '/home/fisher/fisher/Peoples/hseung/카네기/LSMA/Project/LSMA_trial/new_one/aug_sam2_resize_normalize'

# Data loaders
datamanager = CustomDataManager(ORIGINAL_DIR, MASK_DIR, INPAINTED_DIR,
                                n_views=CONFIG["n_views"], num_inpainted=CONFIG["num_inpainted"])
train_loader, val_loader = datamanager.get_loaders()

# Phase 1: Contrastive Learning
print("Starting Phase 1: Contrastive Learning")
encoder = ResNetFeatureExtractor(num_classes=None, use_head=False)  # No classification head
optimizer = torch.optim.Adam(encoder.parameters(), lr=CONFIG["learning_rate"])
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
contrastive_phase = ContrastivePhase(
    train_loader=train_loader,
    val_loader=val_loader,
    model=encoder,
    optimizer=optimizer,
    scheduler=scheduler,
    device=config["device"],
)

contrastive_phase.train(num_epochs=config["num_epochs_phase1"], save_path="encoder_weights.pth")

# Phase 2: Supervised Learning with Classification
# Phase 2: Supervised Learning
print("Starting Phase 2: Supervised Learning")
encoder = ResNetFeatureExtractor(num_classes=751, use_head=True)  # Classification head enabled
encoder.load_state_dict(torch.load("encoder_weights.pth"), strict=False) # Load pretrained weights

optimizer = torch.optim.Adam(encoder.parameters(), lr=config["learning_rate"])
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1, step_size_up=500,mode="triangular")

engine = SupervisedContrastiveEngine(
    train_loader=train_loader,
    val_loader=val_loader,
    model=encoder,
    optimizer=optimizer,
    scheduler=scheduler,
    use_gpu=True,
)
engine.train(num_epochs=config["num_epochs_phase2"], save_path="final_model.pth")

wandb.finish()
