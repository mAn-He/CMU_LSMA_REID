# dataset.py
import os
import numpy as np
import random
from PIL import Image, ImageDraw
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from config import CONFIG
# Initialize WandB
# wandb.init(project="supervised-contrastive-reid", config={"architecture": "SimCLR2", "task": "re-identification"})
# config = wandb.config


def process_images(total_images):
    """
    Ensures all images in total_images are tensors.
    If an image is a PIL Image or ndarray, it is converted to a tensor using T.ToTensor().
    """
    processed_images = []
    for img in total_images:
        if isinstance(img, torch.Tensor):
            # 이미 Tensor라면 변환 생략
            processed_images.append(img)
        elif isinstance(img, Image.Image) or isinstance(img, np.ndarray):
            # PIL Image 또는 ndarray라면 Tensor로 변환
            processed_images.append(T.ToTensor()(img))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
    return torch.stack(processed_images)  # Stack all tensors into a single tensor


# 랜덤 박스 생성
def add_random_box(image, box_color=(0, 0, 0)):
    """이미지에 랜덤 박스를 추가합니다."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    box_width = random.randint(int(width * 0.1), int(width * 0.3))
    box_height = random.randint(int(height * 0.1), int(height * 0.3))
    top_left_x = random.randint(0, width - box_width)
    top_left_y = random.randint(0, height - box_height)
    bottom_right_x = top_left_x + box_width
    bottom_right_y = top_left_y + box_height
    draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], fill=box_color)
    return T.ToTensor()(image)


# # 랜덤 노이즈 추가
# def apply_random_noise(image, mask=None):
#     """마스크를 기반으로 랜덤 노이즈를 추가하여 inpainting 이미지를 생성."""
#     # Ensure image is a NumPy array with dtype float32
#     image = np.array(image, dtype=np.float32)

#     # Ensure channel dimension is last (H, W, C)
#     if image.shape[0] == 3:  # Channel-first format
#         image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)

#     # Generate noise
#     if mask is not None:
#         mask = np.array(mask, dtype=np.float32)
#         if mask.shape[0] == 3:  # Ensure mask has the correct shape
#             mask = np.transpose(mask, (1, 2, 0))
#         noise = np.random.normal(0, 25, image.shape)  # Generate noise
#         inpainted = image * mask + noise * (1 - mask)
#     else:
#         noise = np.random.normal(0, 25, image.shape)
#         inpainted = image + noise

#     # Clip values to valid range and convert to uint8
#     inpainted = np.clip(inpainted, 0, 255).astype(np.uint8)

#     return Image.fromarray(inpainted)
def apply_colored_noise(image, mask=None):
    """특정 색상으로 이미지를 수정하거나 테두리에 색상을 칠함."""
    # 색상 리스트 (RGB)
    colors = [
        (0, 255, 0),    # 초록색
        (0, 0, 255),    # 파란색
        (255, 165, 0),  # 주황색
        (255, 255, 255),# 하얀색
        (255, 0, 0),    # 빨간색
        (245, 245, 220),# 베이지색
        (128, 128, 128),# 회색
        (255, 255, 0),  # 노란색
        (135, 206, 235) # 하늘색
    ]

    # Ensure image is a NumPy array with dtype float32
    image = np.array(image, dtype=np.float32)

    # Ensure channel dimension is last (H, W, C)
    if image.shape[0] == 3:  # Channel-first format
        image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)

    # 랜덤 색상 선택
    chosen_color = random.choice(colors)

    # 마스크가 있을 경우
    if mask is not None:
        mask = np.array(mask, dtype=np.float32)
        if mask.shape[0] == 3:  # Ensure mask has the correct shape
            mask = np.transpose(mask, (1, 2, 0))
        inpainted = image * mask + np.array(chosen_color) * (1 - mask)
    else:
        # 테두리 기준 20% 영역에 색상 칠하기
        h, w, _ = image.shape
        border_h = int(h * 0.2)
        border_w = int(w * 0.2)

        # Copy image to avoid modifying original
        inpainted = image.copy()

        # Fill top border
        inpainted[:border_h, :, :] = chosen_color
        # Fill bottom border
        inpainted[-border_h:, :, :] = chosen_color
        # Fill left border
        inpainted[:, :border_w, :] = chosen_color
        # Fill right border
        inpainted[:, -border_w:, :] = chosen_color

    # Clip values to valid range and convert to uint8
    inpainted = np.clip(inpainted, 0, 255).astype(np.uint8)

    return Image.fromarray(inpainted)

# Cutout Transformation
def cutout(image, num_holes=1, hole_size=(50, 50)):
    """이미지에 랜덤으로 구멍을 생성."""
    img = np.array(image)
    h, w, _ = img.shape
    for _ in range(num_holes):
        y = random.randint(0, h - hole_size[0])
        x = random.randint(0, w - hole_size[1])
        img[y:y + hole_size[0], x:x + hole_size[1], :] = 0
    return T.ToTensor()(Image.fromarray(img))


# Contrastive Transformations Class
from torchvision.transforms.functional import to_pil_image
import torch
class ContrastiveTransformations:
    def __init__(self, base_transforms, extra_aug=None, n_views=2):
        self.base_transforms = base_transforms
        # self.extra_aug = extra_aug if extra_aug else []
        self.n_views = n_views
        aug_prob = random.randint(0,11)
        if aug_prob < 5:
            self.extra_aug = extra_aug[0]
        else:
            self.extra_aug = extra_aug[1]

    def __call__(self, x):
        # Ensure the input is a PIL Image
        if isinstance(x, torch.Tensor):
            x = to_pil_image(x)  # Convert Tensor to PIL Image
        elif not isinstance(x, Image.Image):
            raise TypeError(f"Expected input type PIL Image or Tensor, but got {type(x)}")

        random_view = random.randint(1, self.n_views - 1)
        views = [self.base_transforms(x) for _ in range(self.n_views-random_view)]  # 기본 증강
        # print(views[0].type)
        
        extra_views = [self.extra_aug(x) for _ in range(random_view)]  # 추가 증강
        # print(extra_views[0].type)
        return views + extra_views


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, original_dir, mask_dir, inpainted_dir, base_transforms, extra_aug, n_views=4, num_inpainted=2):
        self.original_dir = original_dir
        self.mask_dir = mask_dir
        self.inpainted_dir = inpainted_dir
        self.n_views = n_views
        self.num_inpainted = num_inpainted
        self.image_paths = [os.path.join(original_dir, fname) for fname in os.listdir(original_dir) if fname.endswith('.jpg')]
        self.label_mapping = self.create_label_mapping()

        # Augmentation 설정
        self.transform = ContrastiveTransformations(base_transforms=base_transforms, extra_aug=extra_aug, n_views=n_views)
    def create_label_mapping(self):
        """라벨 값을 0부터 시작하도록 매핑 생성."""
        labels = [int(os.path.basename(path).split('_')[0]) for path in self.image_paths]
        unique_labels = sorted(set(labels))
        return {pid: idx for idx, pid in enumerate(unique_labels)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)

        # Original image loading and conversion to tensor
        original_image = self.load_image(img_path)
        original_image = T.ToTensor()(original_image)  # Convert PIL Image to Tensor

        # Label extraction
        label = int(filename.split('_')[0])
        label = self.label_mapping[label]  # 매핑된 라벨 사용

        # Contrastive + Extra augmentations for views
        views = self.transform(original_image)

        # Inpainting images (processed as tensors)
        inpainted_images = self.load_or_generate_inpainted_images(original_image, filename)

        # Combine views and inpainted images into a single tensor batch
        total_images = process_images(views + inpainted_images[:self.num_inpainted])

        return original_image, total_images, label


    def load_image(self, image_path):
        """이미지를 로드하고 RGB로 변환."""
        return Image.open(image_path).convert('RGB').resize((128,256))

    # def load_or_generate_inpainted_images(self, original_image, filename):
    #     """이미 존재하는 inpainted 이미지를 로드하거나 새로 생성."""
    #     inpainted_images = []
    #     # print(filename)s
    #     for i in range(self.num_inpainted):
    #         inpainted_path = os.path.join(self.inpainted_dir, f"{filename.split('.')[0]}_augmented_{i}.png")
    #         if os.path.exists(inpainted_path):
    #             inpainted_images.append(self.load_image(inpainted_path))
    #         else:
    #             inpainted_images.append(apply_random_noise(original_image))
    #     return inpainted_images
    def load_or_generate_inpainted_images(self, original_image, filename):
    # """이미 존재하는 inpainted 이미지를 로드하거나 새로 생성."""
        inpainted_images = []
        for i in range(self.num_inpainted):
            inpainted_path = os.path.join(self.inpainted_dir, f"{filename.split('.')[0]}_augmented_{i}.png")
            if os.path.exists(inpainted_path):
                inpainted_images.append(self.load_image(inpainted_path))
            else:
                # Generate a new image with random color modifications
                inpainted_images.append(apply_colored_noise(original_image))

        # 임의로 `self.num_inpainted` 개수 선택
        return random.sample(inpainted_images, self.num_inpainted)

# Custom Data Manager Class
class CustomDataManager:
    def __init__(self, original_dir, mask_dir, inpainted_dir, batch_size=8, validation_split=0.2, n_views=4, num_inpainted=2):
        self.original_dir = original_dir
        self.mask_dir = mask_dir
        self.inpainted_dir = inpainted_dir
        self.batch_size = batch_size
        self.validation_split = validation_split

        # Base Transforms
        base_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)], p=0.6),
            T.RandomRotation(degrees=[45, 135]), 
            T.RandomResizedCrop(size=(128, 64), scale=(0.8, 1.0)),
            
            
            T.Resize(size=(256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Extra Augmentations
        extra_aug = [cutout, add_random_box]

        # Dataset 생성
        self.dataset = CustomDataset(original_dir, mask_dir, inpainted_dir, base_transforms, extra_aug,
                                     n_views=CONFIG['n_views'], num_inpainted=CONFIG['num_inpainted'])

        # Train-Validation Split
        train_size = int(len(self.dataset) * (1 - validation_split))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        # Data Loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


    def get_loaders(self):
        return self.train_loader, self.val_loader
    
    
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CustomqueryDataset(Dataset):
    def __init__(self, original_dir, transform=None):
        """
        Custom Dataset for query and gallery images.

        Args:
            original_dir (str): Directory containing images.
            transform (callable): Transform to apply to images.
        """
        self.original_dir = original_dir
        self.image_paths = [os.path.join(original_dir, fname) for fname in os.listdir(original_dir) if fname.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor.
            pid: Person ID.
            camid: Camera ID.
        """
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        image = self.load_image(img_path)

        # Extract Person ID and Camera ID
        pid = int(filename.split('_')[0])
        camid = int(filename.split('_')[1][1])

        if self.transform:
            image = self.transform(image)

        return image, pid, camid

    def load_image(self, image_path):
        """Loads an image and converts it to RGB."""
        return Image.open(image_path).convert('RGB')
