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

#Dataset and Loader
true_labels = '/data/users4/pafshin1/My_Projects/RANDS/DATA/BLOCKS/INPUT_BLOCKS/metadata_output.csv'
patch_dir = '/data/users4/pafshin1/My_Projects/RANDS/DATA/BLOCKS/INPUT_BLOCKS'

class TumorImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with image paths and labels.
            root_dir (string): Directory with all the subfolders containing image patches.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get the sample info from the metadata
        sample_row = self.metadata.iloc[idx]
        sample_name = sample_row['Sample']  # Example: '2_1'
        index = sample_row['Index']          # Example: '1'
        coordinates = sample_row['Coordinates'].strip("()")  # Example: "(5600, 2000)" to "5600, 2000"
        
        # Remove any spaces before and after the row and column
        row, column = [coord.strip() for coord in coordinates.split(",")]  # Strip spaces
        formatted_coordinates = f"{row}_{column}"  # Format as row_column
        label = int(sample_row['Label'])     # Example: '0' or '1'

        # Construct the image path
        image_name = f"PS{sample_name}_{index}_{formatted_coordinates}.tif"
        image_path = os.path.join(self.root_dir, sample_name, image_name)

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)



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
