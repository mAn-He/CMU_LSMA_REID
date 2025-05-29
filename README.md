## Supervised Contrastive Framework for Re-Identification with Inpainting

<hr>

This repository implements a Supervised Contrastive Framework for Re-Identification (Re-ID) with Inpainting. The framework leverages ResNet backbone models, integrates inpainting techniques, and applies custom augmentations to enhance feature robustness and re-identification accuracy. The pipeline is designed for end-to-end learning, encompassing image preprocessing, feature extraction, and model training.

## Project Structure

The data for this project should be organized under a directory named `REID_scratch` at the root of the project. The expected subdirectory structure is as follows:

```
REID_scratch/
├── Market-1501/
│   └── bounding_box_train/  # Original training images
└── LSMA_trial/
    └── new_one/
        ├── sam2_masks/               # Mask images
        └── aug_sam2_resize_normalize/ # Inpainted images
```

## Workflow Overview

The training process consists of two main phases:

1.  **Phase 1: Contrastive Learning**
    *   This phase pre-trains the `ResNetFeatureExtractor` (encoder) using a contrastive learning approach.
    *   The `ContrastivePhase` class manages this stage.
    *   The primary loss function used is `SupConLoss` (Supervised Contrastive Loss), which helps in learning discriminative features by pulling samples from the same class closer in the embedding space while pushing samples from different classes apart.

2.  **Phase 2: Supervised Learning**
    *   In this phase, the pre-trained encoder is fine-tuned, and a classifier head is trained for the Re-ID task.
    *   The `SupervisedContrastiveEngine` class orchestrates this stage.
    *   The loss function combines cross-entropy loss (for the classification task) and the contrastive loss (to retain the benefits of the learned discriminative features).

### Key Modules and Classes:

*   **`main.py`**: Orchestrates the two training phases, from data loading to model training and saving.
*   **`model.py`**: Defines the `ResNetFeatureExtractor`, which serves as the backbone for feature extraction. It can be configured to use different ResNet versions (e.g., ResNet-101).
*   **`dataset.py`**:
    *   Contains `CustomDataManager` for loading and preparing the data for both training phases.
    *   Implements `ContrastiveTransformations`, which include a variety of data augmentations crucial for robust feature learning. These augmentations include:
        *   Random Horizontal Flip
        *   Color Jitter (adjusting brightness, contrast, saturation, hue)
        *   Random Rotation
        *   Cutout (randomly masking out regions of the image)
        *   Random Boxes (overlaying random boxes on images)
        *   Special handling for incorporating inpainted images alongside original and masked images.
*   **`config.py`**: Manages all configuration parameters for the project, such as learning rates, batch sizes, model choices, and data paths.

## Feature Extraction

*   **Backbone Model**: Utilizes ResNet models (e.g., ResNet-101, configurable via `CONFIG["backbone"]` in `config.py`) for powerful feature extraction from images.
*   **Projection Head**: Optionally, a projection head can be used (common in contrastive learning setups) to transform features before the contrastive loss calculation, though the primary focus is on the encoder's features for the downstream classification task.

<hr>

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/mAn-He/CMU24_LSMA_KC.git
    cd CMU24_LSMA_KC
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure a `requirements.txt` file is present in the repository for this command to work.)*

### Weights & Biases (WandB) Configuration:
This project uses WandB for experiment tracking. After installation, when you run the main script (`main.py`), you might be prompted to log in to WandB if your API key is not already configured locally. Alternatively, you can set the `WANDB_API_KEY` environment variable with your API key before running the script.

### LSMA Project Repository
Made by Hyeonbin, Yongsik, Hoseung. Thanks!
Upgrade ..ing
