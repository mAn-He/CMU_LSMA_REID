import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomqueryDataset  # Custom Dataset for query and gallery
from distance import compute_distance_matrix  # Distance matrix computation
from rank import evaluate_rank  # Rank and mAP evaluation
from model import ResNetFeatureExtractor  # Your pre-trained model
from config import CONFIG  # Import CONFIG

def extract_features(loader, model, device):
    """
    Extract features from a DataLoader using the provided model.

    Args:
        loader (DataLoader): DataLoader for the dataset.
        model (nn.Module): Trained model.
        device (torch.device): Device to use (CPU or GPU).

    Returns:
        features (torch.Tensor): Extracted features.
        pids (list): Person IDs.
        camids (list): Camera IDs.
    """
    model.eval()
    features, pids, camids = [], [], []

    with torch.no_grad():
        for images, pids_batch, camids_batch in tqdm(loader, desc="Extracting Features"):
            images = images.to(device)
            feats, _ = model(images)
            features.append(feats.cpu())
            pids.extend(pids_batch)
            camids.extend(camids_batch)

    return torch.cat(features, dim=0), pids, camids


def test_model(model, query_loader, gallery_loader, metric='euclidean', top_k=[1, 5, 10]):
    """
    Test the model by comparing query and gallery features.

    Args:
        model (nn.Module): Trained model.
        query_loader (DataLoader): DataLoader for the query dataset.
        gallery_loader (DataLoader): DataLoader for the gallery dataset.
        metric (str): Distance metric ('euclidean' or 'cosine').
        top_k (list): List of ranks to evaluate.

    Returns:
        cmc (np.ndarray): Cumulative Matching Characteristic values.
        mAP (float): Mean Average Precision.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract query and gallery features
    print("Extracting gallery features...")
    gallery_features, g_pids, g_camids = extract_features(gallery_loader, model, device)
    print("Extracting query features...")
    query_features, q_pids, q_camids = extract_features(query_loader, model, device)

    # Compute distance matrix
    print("Computing distance matrix...")
    distmat = compute_distance_matrix(query_features, gallery_features, metric=metric)

    # Evaluate CMC and mAP
    print("Evaluating results...")
    cmc, mAP = evaluate_rank(
        distmat=distmat,
        query_pids=q_pids,
        gallery_pids=g_pids,
        query_camids=q_camids,
        gallery_camids=g_camids,
        topk=top_k,
    )

    # Print results
    print(f"Results: mAP: {mAP:.2%}")
    for k in top_k:
        print(f"Rank-{k}: {cmc[k-1]:.2%}")

    return cmc, mAP


# Main script
if __name__ == "__main__":
    # Load pre-trained model
    model_path = 'best_model.pth'
    # num_classes = 751  # Update based on your dataset
    model = ResNetFeatureExtractor(num_classes=CONFIG["num_classes"], use_head=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Dataset paths
    query_dir = '/home/fisher/fisher/Peoples/hseung/카네기/LSMA/Project/ReIDataset/Market-1501/query'
    gallery_dir = '/home/fisher/fisher/Peoples/hseung/카네기/LSMA/Project/ReIDataset/Market-1501/bounding_box_test'

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # DataLoaders for query and gallery datasets
    query_loader = DataLoader(CustomqueryDataset(query_dir, transform=transform), batch_size=1, shuffle=False)
    gallery_loader = DataLoader(CustomqueryDataset(gallery_dir, transform=transform), batch_size=32, shuffle=False)

    # Test the model
    cmc, mAP = test_model(model, query_loader, gallery_loader, metric='euclidean', top_k=[1, 5, 10])
