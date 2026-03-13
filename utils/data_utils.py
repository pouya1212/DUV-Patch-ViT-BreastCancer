import logging
import os
import torch
from PIL import Image
import random
import numpy as np
import pandas as pd

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler, , random_split


logger = logging.getLogger(__name__)


# dataset class for delivering patches of WSI images along with their names, WSI name, index of patch, and its corresponding coordinates on the WSI image. 
class TumorImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, resize_size=(224, 224), transform=True):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.resize_size = resize_size
        self.transform = transform
        
        # Load CSV and process labels
        self.metadata = pd.read_csv(csv_file)
        # Generate the transformation pipeline
        self.transforms = self.generate_transform(self.resize_size)
        # self.process_labels()

    def generate_transform(self, resize_size):
        """Generate transformation pipeline for preprocessing image data."""
        transform_list = []

        # Convert image to tensor
        transform_list.append(self.contiguous_tensor)

        # Resize image if size is provided
        if resize_size:
            transform_list.append(transforms.Resize(resize_size))

        # Rescale image to range [0, 1]
        transform_list.append(self.rescale_tensor)

        # Normalize image (optional)
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        return transforms.Compose(transform_list)

    def contiguous_tensor(self, image):
        """Convert image to tensor and ensure contiguous memory layout."""
        # Ensure the image is in (H, W, C) format
        image = np.array(image)  # Convert PIL image to numpy array
        if image.ndim == 2:  # For grayscale images, add the channel dimension
            image = np.expand_dims(image, axis=-1)
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        return torch.from_numpy(image).contiguous()

    def rescale_tensor(self, tensor):
        """Rescale tensor to range [0, 1]."""
        return tensor.to(dtype=torch.get_default_dtype()).div(255)

    def __getitem__(self, idx):
        # Get the sample info from the metadata
        sample_row = self.metadata.iloc[idx]
        wsi_name = sample_row['Sample']  # Example: '2_1'
        patch_index = sample_row['Index']  # Example: '1'
        row = sample_row['Row']  # Example: '5600'
        column = sample_row['Column']  # Example: '2000'
        
        # These weights were suppressed to zero if they were belowe 0.25
        patch_densenet_gradcam_importance_weight = sample_row['Densenet_Gradcam_Weight']
        #the weights are not suppressed to zero below 0.25
        Densenet_importance = sample_row['Densenet_Gradcam_Saliency_Importance']
        
        # Format coordinates as "row_column" (e.g., "5600_2000")
        formatted_coordinates = f"{row}_{column}"
        
        label = int(sample_row['Binary_Label'])  # 0 (benign) or 1 (malignant), it will read the label from CSV file

        # Construct the image path
        patch_name = f"PS{wsi_name}_{patch_index}_{formatted_coordinates}.tif"
        patch_path = os.path.join(self.root_dir, wsi_name, patch_name)

        # Load the image
        image = Image.open(patch_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transforms(image)  # Apply the transformation pipeline

        # Return the image, label, image name, and the metadata as separate elements
        return image, torch.tensor(label, dtype=torch.long), patch_name, (wsi_name, patch_index, (row,column)), patch_densenet_gradcam_importance_weight

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.metadata)



def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    
    # Define the transformations
    transform_bn = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),    
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])  
    
    # Only load the tumor dataset if args.dataset is set to "tumor"
    if args.dataset == "tumor":
        true_labels = '/data/users4/pafshin1/My_Projects/RANDS/DATA/BLOCKS/INPUT_BLOCKS/metadata_output.csv'
        patch_dir = '/data/users4/pafshin1/My_Projects/RANDS/DATA/BLOCKS/INPUT_BLOCKS'
        dataset = TumorImageDataset(csv_file=true_labels, root_dir=patch_dir, transform=transform_bn)
        
        # Total size of the dataset
        total_size = len(dataset)
        
        # Define 70/15/15 split sizes
        train_size = int(0.64 * total_size)   # 70% for training
        val_size = int(0.16 * total_size)    # 15% for validation
        test_size = total_size - train_size - val_size  # Remaining for test (15%)
        
        # Randomly shuffle the dataset indices
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        
        # Split indices into train, val, and test sets
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subsets for train, validation, and test
        trainset = torch.utils.data.Subset(dataset, train_indices)
        valset = torch.utils.data.Subset(dataset, val_indices)
        testset = torch.utils.data.Subset(dataset, test_indices)
        
    
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported. Please set `args.dataset` to 'tumor'.")
    
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset)
    test_sampler = SequentialSampler(testset)

    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    
    val_loader = DataLoader(valset,
                            sampler=val_sampler,
                            batch_size=args.eval_batch_size,
                            num_workers=4,
                            pin_memory=True)
    
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, val_loader, test_loader
