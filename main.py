# coding=utf-8
from __future__ import absolute_import, division, print_function

# Standard library
import os
import re
import time
import json
import random
import logging
from itertools import chain
from collections import defaultdict, Counter
from datetime import timedelta
import argparse

# Third-party libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyfiglet import Figlet
from PIL import Image, ImageDraw
import logging
import os
import torch
from PIL import Image
import random
import numpy as np
import pandas as pd

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset

# Torch and related
import torch
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter

# Apex for mixed precision + distributed training
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

# Torchvision
from torchvision import transforms, datasets
import torch.nn.functional as F  # for softmax computation

# Scikit-learn
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
)

# Custom modules
from utils.data_utils import TumorImageDataset, get_loader
from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.dist_util import get_world_size
# from utils.data_utils import get_loader

# Logging
logger = logging.getLogger(__name__)

# Prevent PIL image size warnings
Image.MAX_IMAGE_PIXELS = None

#after running grad-cam++, the updated metadata can be obtained
#load and create dataset from CSV file and patches directories
patchdataset = TumorImageDataset(
            csv_file='/data/users4/pafshin1/Implementation/Vision Transformer/EMBC_Updated_Automatic_Thresholding_Implementation/CSV_Files/metadata_patches_with_grad_cam++_binary_label.csv',
            root_dir='/data/users4/pafshin1/My_Projects copy/Rands/DATA/PATCHES/INPUT_PATCHES',
            resize_size=(224, 224), transform=True)

#Load the CSV file that contains the groundtruth labels of WSI obtained from the ground-truth labels of patches
wsi_csv_labels1 = "/data/users4/pafshin1/My_Projects/All_Labels/filtered_wsi_labels.csv"
wsidata_df = pd.read_csv(wsi_csv_labels1)
#The CSV contains all patches information, including the pathologist labels (not binary ones) and the weights obtained by VIT and DenseNet Gradcams
patch_csv_weights = '/data/users4/pafshin1/Implementation/Vision Transformer/EMBC_Updated_Automatic_Thresholding_Implementation/CSV_Files/metadata_patches_with_grad_cam++_binary_label.csv'
Binary_Label_patch_csv = patch_csv_weights
meta = pd.read_csv(Binary_Label_patch_csv)




