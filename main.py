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
from utils.metrics import AverageMeter, simple_accuracy, compute_wsi_metrics
from models.model_setup import setup, count_parameters, save_model
from models.modeling import VisionTransformer, CONFIGS
from engine.train_evaluate import train_test, test, valid
from utils.metrics import AverageMeter, simple_accuracy, compute_wsi_metrics
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
# wsi_csv_labels1 = "/data/users4/pafshin1/My_Projects/All_Labels/filtered_wsi_labels.csv"
# wsidata_df = pd.read_csv(wsi_csv_labels1)
#The CSV contains all patches information, including the pathologist labels (not binary ones) and the weights obtained by VIT and DenseNet Gradcams
patch_csv_weights = '/data/users4/pafshin1/Implementation/Vision Transformer/EMBC_Updated_Automatic_Thresholding_Implementation/CSV_Files/metadata_patches_with_grad_cam++_binary_label.csv'
Binary_Label_patch_csv = patch_csv_weights
meta = pd.read_csv(Binary_Label_patch_csv)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def save_plots_for_fold(fold, output_dir, train_losses, train_epoch_losses, val_losses, val_accuracies):
    """
    Save the training and validation loss/accuracy plots for each fold.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.xlabel("Total number of Training Steps")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - Fold {fold}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"training_loss_fold{fold}.png"))
    plt.close()  # Close the plot to free up memory

    # Plot training average loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_epoch_losses, label='Average Loss', color='blue', marker='o', linestyle='-')
    plt.title(f'Average Loss Over Training Steps - Fold {fold}')
    plt.xlabel('Epochs (each point corresponds to avg loss over 50 training steps = 1 epoch )')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"average_loss_plot_fold{fold}.png"))  # Save with fold info
    plt.close()

    # Plot validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(val_losses, label="Validation Loss", color='orange')
    plt.xlabel("Epochs (each point corresponds to evaluation after 50 training steps = 1 epoch)")
    plt.ylabel("Loss")
    plt.title(f"Validation Loss - Fold {fold}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"validation_loss_fold{fold}.png"))
    plt.close()  # Close the plot to free up memory

    # Plot validation accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(val_accuracies, label="Validation Accuracy", color='green')
    plt.xlabel("Epochs (each point corresponds to evaluation after 50 training steps = 1 epoch)")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy - Fold {fold}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"validation_accuracy_fold{fold}.png"))
    plt.close()  # Close the plot to free up memory

#################################################################################################################################
# function that generates box plots if args, all patch_names, all patch_labels, and their predictions, and all WSI image names will be provided to it

def visualize_patch_locations(args, all_patch_names, all_patch_labels, all_patch_predictions, all_wsi_names, all_patch_coordinates, patch_size=400):
    
    # Dictionary to store WSI-level data
    wsi_data = defaultdict(list)

    # Group patches by their corresponding WSI image
    for patch_name, wsi_name, label, prediction, coordinates in zip(all_patch_names, all_wsi_names, all_patch_labels, all_patch_predictions, all_patch_coordinates):
        # Extract coordinates from the patch name (assumed to be part of the name, e.g., PS2_1_1_5600_2000.tif)
        pattern = r'_(\d+)_(\d+)\.tif$'
        match = re.search(pattern, patch_name)
        
        if match:
            y_coord, x_coord = int(match.group(1)), int(match.group(2))
            # Store data for each WSI
            wsi_data[wsi_name].append((patch_name, x_coord, y_coord, label, prediction))  # Save patch name, coordinates, true label, and predicted label
    
    csv_data = []  # For creating CSV files for all information
    
    # Iterate over each WSI image and plot patches
    for wsi_name, patches in wsi_data.items():
        # Define the path to the WSI image
        wsi_path = os.path.join(args.wsi_img_path, f"{wsi_name}.jpg")
        print(f"Checking WSI path: {wsi_path}")
        
        if os.path.exists(wsi_path):
            print(f"Found WSI image: {wsi_name}")
            # Load the WSI image
            wsi_image = Image.open(wsi_path)
            predicted_image = wsi_image.copy()  # For predicted patches
            ground_truth_image = wsi_image.copy()  # For ground truth patches
            
            draw_predicted = ImageDraw.Draw(predicted_image)
            draw_ground_truth = ImageDraw.Draw(ground_truth_image)

            # Loop over each patch and draw its corresponding bounding box
            for patch_name, x_coord, y_coord, true_label, pred_label in patches:
                # Define the bounding box for the patch on WSI
                box = [
                    x_coord, y_coord,                         # Top-left corner (x, y)
                    x_coord + patch_size, y_coord + patch_size  # Bottom-right corner (x + width, y + height)
                ]
                
                # Predicted Visualization: Color based on predicted label
                pred_color = "green" if pred_label == 0 else "red"  # Green for benign (0), red for malignant (1)
                draw_predicted.rectangle(box, outline=pred_color, width=15)

                # Ground Truth Visualization: Color based on true label
                true_color = "green" if true_label == 0 else "red"  # Blue for benign (0), yellow for malignant (1)
                draw_ground_truth.rectangle(box, outline=true_color, width=15)
                
                # Append data for CSV
                csv_data.append({
                    'WSIName': wsi_name,
                    'PatchName': patch_name,
                    'X_Coordinate': x_coord,
                    'Y_Coordinate': y_coord,
                    'TrueLabel': true_label,
                    'Prediction': pred_label
                })

            # Save the result images with highlighted patches
            box_plot_dir = os.path.join(args.output_dir, "Box_Plot_Images")
            os.makedirs(box_plot_dir, exist_ok=True)
            
            # Save predicted patch visualization
            predicted_image_path = os.path.join(box_plot_dir, f"{wsi_name}_predicted.jpg")
            predicted_image.save(predicted_image_path)
            
            # Save ground truth patch visualization
            ground_truth_image_path = os.path.join(box_plot_dir, f"{wsi_name}_ground_truth.jpg")
            ground_truth_image.save(ground_truth_image_path)

        else:
            print(f"WSI image {wsi_name}.jpg not found.")

    # Ensure output directory exists before saving CSV
    os.makedirs(args.output_dir, exist_ok=True)
    # Save all patch data to CSV
    results_dir = os.path.join(args.output_dir, "Results")
    patch_csv_path = os.path.join(results_dir, "all_patches_information.csv")
    pd.DataFrame(csv_data).to_csv(patch_csv_path, index=False)

    print("Patch locations visualized and saved for all WSIs.")
    print("Predicted and ground truth visualizations saved in:", box_plot_dir)
    print("Patch information saved to CSV in:", patch_csv_path)


##################################################################################################################################
# function to compute weighted average voting using the Highest probability of softmax and the networks prediction label whether -1  for benign and +1 as malignant

def compute_wsi_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (y_true == y_pred).mean()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    return acc, sens, spec, prec, f1

def summarize(metric_list, name, method, f):
    mean = np.mean(metric_list)
    std = np.std(metric_list)
    print(f"{method:<18} | {name:<12}: {mean:.4f} ± {std:.4f}")
    f.write(f"{method:<18} | {name:<12}: {mean:.4f} ± {std:.4f}\n")


#################################################################################################################################
# define a main function, including all args, which:
# 1) applies n-fold cross validation, save the Train, val loss and accuracy along with test accuracy
# 2) it will take the mean and standard_deviation of the different folds accuracy
# 3) it will captures the patches information for each fold such as name of wsi image, patch name, index and its coordinate
# 4) it will then use these information to compute majority voting accuracy for wsi image and compare the prediction with label of wsi image.    
# 5) it will generate a box plot on the wsi image using the prediction of wsi' patches 

   
def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", type=str, default="1st Batch of the Breast Cancer Data (EMBC)",
                    help="Name of this run. Used for monitoring.")
    # parser.add_argument("--dataset", choices=["tumor"], default="tumor",
    #                     help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/data/users4/pafshin1/Implementation/Vision Transformer/Original_VIT2/models/Pretrained_models/imagenet21k_ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="1st_batch_wsi-level_classification_automatic_threshold", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument('--wsi_img_path', type=str, default='/data/users4/pafshin1/My_Projects/Rands/DATA/PATCHES/INPUT_WSI',  # Default value 
                        help='Path to the directory containing WSI images')

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=256, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=50, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    
    parser.add_argument("--learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for. I considered the  5% total_steps which is 5000 = 250")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_epochs", default=60, type=int, help="number of epochs.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1: # non-distributed training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))


    # Set seed
    set_seed(args)
    
    ####################################################################################
    # function to generate n different predefined sets of training, validation and test sets used for cross validation
    # it ensures that all instances of the dataset appears once in the test set

    # Labels mapping from Pathologist to turn the labels to binary ones
    fancy_title = Figlet(font='standard')

    title_line_1 = "WSI-Level and Patch-Level Classification for"
    title_line_2 = args.name

    # Banner lines to be displayed and logged
    lines = [
        "WSI-Level and Patch-Level Classification for",
        args.name,
        "Multi-Fold Training and Evaluation",
        "Vision Transformer + Majority Voting",
        "Pouya Afshin - GSU"
    ]

    
    labelBenign, labelMalignant, labelExclude = '0', '1', '2'
    #Define labels used for normal/benign tissue
    #'a': normal adipose.
    #'s': normal stroma tissue excluding adipose.
    #'o': other normal tissue including parenchyma, adenosis, lobules, blood vessels, etc.
    labelsBenign = ['a', 's', 'o', 'normal']

    #Define labels used for malignant tissue
    #'d': IDC tumor
    #'l': ILC tumor
    #'ot': other tumor areas including DCIS, biopsy site, and slightly defocused tumor regions.
    labelsMalignant = ['d', 'l', 'ot', 'tumor']

    #Define labels used for tissues to be excluded
    #'ft': defocused but still visually tumor-like areas.
    #'f': severly out-of-focusing areas. 
    #'b': background. 
    #'e': bubbles.
    labelsExclude = ['ft', 'f', 'b', 'e', 'exclude']


    #Load/synchronize data labeling, drop excluded rows, and extract relevant metadata
    def loadMetadata_patches(filename):
        metadata = pd.read_csv(filename, header=0, names=['Sample', 'Index', 'Row', 'Column', 'Label'], converters={'Sample':str,'Index':str, 'Row':int, 'Column':int, 'Label':str})
        metadata['Label'] = metadata['Label'].replace(labelsBenign, labelBenign)
        metadata['Label'] = metadata['Label'].replace(labelsMalignant, labelMalignant)
        metadata['Label'] = metadata['Label'].replace(labelsExclude, labelExclude)
        metadata = metadata.loc[metadata['Label'] != labelExclude]
        return [np.squeeze(data) for data in np.split(np.asarray(metadata), [1, 2, 4], -1)]

    patchSampleNames_patches, indices_patches, locations_patches, patchLabels_patches = loadMetadata_patches('/data/users4/pafshin1/My_Projects copy/Rands/DATA/PATCHES/INPUT_PATCHES/metadata_patches.csv')
    patchLabels = patchLabels_patches.astype(int)
    patchNames_patches = np.asarray([patchSampleNames_patches[index] + '_' + indices_patches[index] for index in range(0, len(patchSampleNames_patches))]) # give an array contains all lists of the patches of wsi names 2_1_1
    sampleNames_patches = np.unique(patchSampleNames_patches) # give the array with the names of all wsi Images for example '10_3' 

    patchFilenames_patches = np.asarray(['/data/users4/pafshin1/My_Projects copy/Rands/DATA/PATCHES/INPUT_PATCHES/' + patchSampleNames_patches[index] + os.path.sep + 'PS'+patchSampleNames_patches[index]+'_'+str(indices_patches[index])+'_'+str(locations_patches[index, 0])+'_'+str(locations_patches[index, 1])+'.tif' for index in range(0, len(patchSampleNames_patches))])
    WSIFilenames_patches = np.asarray(['/data/users4/pafshin1/My_Projects copy/Rands/DATA/PATCHES/INPUT_WSI/' + sampleName + '.jpg' for sampleName in sampleNames_patches]) # give the location of each wsi images for exa My_Projects copy/Rands/DATA/PATCHES/INPUT_WSI/10_3.jpg
    patchNames = patchNames_patches # name of patches like 2_1_1
    patchFilenames = patchFilenames_patches # directory of patches
    patchSampleNames = patchSampleNames_patches # name of wsis
    patchLocations = locations_patches # coordinate of the patches

    sampleNames =  sampleNames_patches # name of wsi images
    WSIFilenames = WSIFilenames_patches # location of wsi images

    patchLabels = patchLabels_patches # contains all labels of the patches array([0, 0, 0, ..., 0, 0, 0]) for each patch wh



    #new Manual folds contain all 66 Samples (before 6 missing WSIs (13_1, 21_1, 32_2, 38_1, 3_1, 6_1))
    manualFolds = [
    ['2_1', '9_3', '11_3', '16_3', '34_1', '36_2', '40_2', '54_2', '57_2', '60_1', '62_1', '13_1', '21_1'],
    ['17_5', '20_3', '23_3', '24_2', '28_2', '30_2', '33_3', '51_2', '52_2', '59_2', '63_3', '66_2', '32_2'],
    ['12_1', '14_2', '22_3', '26_3', '35_4', '44_1', '45_1', '47_2', '49_1', '53_2', '56_2', '68_1', '38_1'],
    ['4_4', '5_3', '8_1', '10_3', '25_3', '27_1', '29_2', '37_1', '42_3', '48_3', '50_1', '69_1', '3_1'],
    ['7_2', '15_4', '19_2', '31_1', '43_1', '46_2', '55_2', '58_2', '61_1', '64_1', '65_1', '67_1', '70_1', '6_1']
    ]

    #Allocate samples to folds for cross validation
    if type(manualFolds) != list: 
        folds = [array.tolist() for array in np.array_split(np.random.permutation(sampleNames), manualFolds)]
    else: 
        folds = manualFolds

    numFolds = len(folds)

    #Split patch features into specified folds, keeping track of originating indices and matched labels
    patchIndices_, foldsFeatures, foldsLabels, foldsWeights, foldsPatchSampleNames, foldsPatchNames, foldsPatchLocations = [], [], [], [], [], [], []
    for fold in folds:
        patchIndices = np.concatenate([np.where(patchSampleNames == sampleName)[0] for sampleName in fold]) # combines the indices of all patches that belong to the samples in the current fold
        foldsPatchLocations.append(list(patchLocations[patchIndices])) # do the same thing for their locations
        foldsLabels.append(list(patchLabels[patchIndices]))       # do the same thing for their labels
        patchIndices_.append(patchIndices)
        
        foldsPatchSampleNames.append(list(patchSampleNames[patchIndices]))
        foldsPatchNames.append(list(patchNames[patchIndices]))            

    #Collapse data for later (correct/matched ordered) evaluation of the fold data
    foldsSampleNames = np.asarray(sum(folds, [])) #flattens the list of folds into a single list of all sample names.
    foldsPatchLocations = np.concatenate(foldsPatchLocations) # combines lists or arrays of patch information (such as locations, names, etc.) across all folds into single arrays.
    foldsPatchSampleNames = np.concatenate(foldsPatchSampleNames) # combine the list of all WSI images
    foldsPatchNames = np.concatenate(foldsPatchNames) # combine all Patch names
    foldsWSIFilenames = np.concatenate([WSIFilenames[np.where(sampleNames == sampleName)[0]] for sampleName in foldsSampleNames]) 


    #Perform training/testing among the folds, testing sequentially (to ensure the correct order) and storing results for later evaluation
    foldsPatchPredictions, foldsPatchPredictionsFusion = [], []
    
    
    train_epoch_losses_per_fold = []
    train_losses_per_fold = []
    val_losses_per_fold = []
    val_accuracies_per_fold = []
    test_accuracies_per_fold = []
    best_accuracies_per_fold = []

    all_test_indices = []


    # Initialize lists to store patch-level data across all folds
    
    all_patch_names = []
    all_patch_labels = []
    all_patch_predictions = []
    all_wsi_names = []
    all_patch_indices = []
    all_patch_coordinates = []
    
    # Initialize a list to store top probabilities for weighted voting
    all_top_patches_probabilities = []
    
    # initialize a list to store gradcam weights for each patches for weighted voting
    all_patch_densenet_gradcam_importance_weights = []  
    # Add these lists to store fold-specific metrics
    sensitivity_per_fold = []
    specificity_per_fold = []
    
    all_fold_thresholds = []
    
    all_slide_dfs = []

     ################WSI-LEVEL test evaluation######################

    # Majority voting metrics per fold
    maj_accs, maj_sens, maj_specs, maj_precs, maj_f1s = [], [], [], [], []

    # Softmax-weighted voting metrics per fold
    soft_accs, soft_sens, soft_specs, soft_precs, soft_f1s = [], [], [], [], []

    # Grad-CAM++-weighted voting metrics per fold

    gradcam_accs , gradcam_sens, gradcam_specs, gradcam_precs, gradcam_f1s = [], [], [], [] , []

    ####################################################################
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = os.path.join(args.output_dir, "Results")
    os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    results_file_path = os.path.join(results_dir, "cross_validation_results.txt")

  
    # Open a text file to save results for each fold
    with open(results_file_path, "w") as f:
        for line in lines:
            banner = fancy_title.renderText(line)
            print(banner)
            logger.info("\n" +banner)
            f.write(banner + "\n")
            f.flush()
            os.fsync(f.fileno())
        
        # WSI-Level Distribution 
        counts_wsi = wsidata_df['Binary_Label'].value_counts().sort_index()
        percentages = counts_wsi / counts_wsi.sum() *100 
        wsi_distribution_msg = (
            "\n WSI-level Distribution : \n"
            f"Benign (0):   {counts_wsi[0]} WSIs ({percentages[0]:.2f}%)\n"
            f"Malignant (1) :{counts_wsi[1]} WSIs ({percentages[1]:.2f}%)\n"
        )
        print(wsi_distribution_msg)
        logger.info(wsi_distribution_msg)
        f.write(wsi_distribution_msg + "\n")
        f.flush()
        os.fsync(f.fileno())
        #================save plot=============
        labels = ['Benign (0)', 'Malignant (0)']
        plt.figure(figsize = (5,6))
        plt.bar(labels, counts_wsi, color= ['green', 'red'])
        plt.title('WSI-Level Distribution')
        plt.ylabel('Number of WSIs')
        plt.xlabel('Categories')
        max_count = max(counts_wsi)
        plt.ylim(0, max_count*1.15) 

        for i, (count, pct) in enumerate(zip(counts_wsi, percentages)):
            plt.text(i, count/2, f"{count}", ha = 'center', va ='center', color ='white', fontsize =12, fontweight ='bold' )

            #percentage outside the bar
            plt.text(i, count + max_count * 0.02, f"{pct:.1f}%", ha= 'center', va='center', fontsize='12', fontweight ='bold')
        
        plot_path = os.path.join(results_dir, "wsi-level_distribution.png")
        plt.savefig(plot_path, bbox_inches='tight')

        logger.info(f"Saved WSI-Level distribution plot to: {plot_path}")
        f.write(f"Plot saved to: {plot_path}\n")

        f.flush()
        os.fsync(f.fileno())

        #============Patch-Level Distribution after filtering===========
        all_wsis = list(chain.from_iterable(manualFolds))
        filtered_meta = meta[meta['Sample'].isin(all_wsis)]
        patch_counts = filtered_meta['Binary_Label'].value_counts().sort_index()
        filtered_meta['Binary_Label'].value_counts().sort_index
        patch_percentages = patch_counts / patch_counts.sum() * 100 

        patch_distribution_msg = (
            "\n Patch-level Distribution : \n"
            f"Benign (0):   {patch_counts[0]} patches ({patch_percentages[0]:.2f}%)\n"
            f"Malignant (1) :{patch_counts[1]} patches ({patch_percentages[1]:.2f}%)\n"
        )
        print(patch_distribution_msg)
        
        logger.info(patch_distribution_msg)
        f.write(patch_distribution_msg + "\n")
        f.flush()
        os.fsync(f.fileno())

        #====================save Patch-level distribution bar plot=====================

        patch_labels = ['Benign (0)', 'Malignant (0)']
        plt.figure(figsize = (5,6))
        plt.bar(patch_labels, patch_counts, color= ['green', 'red'])
        plt.title('Patch-Level Distribution')
        plt.ylabel('Number of patches')
        plt.xlabel('Categories')
        max_count = max(patch_counts)
        plt.ylim(0, max_count*1.15) 

        f.write("=== Cross-Validation Results ===\n")
        f.flush()
        os.fsync(f.fileno())

        for i, (count, pct) in enumerate(zip(patch_counts, patch_percentages)):
            plt.text(i, count/2, f"{count}", ha = 'center', va ='center', color ='white', fontsize =12, fontweight ='bold' )

            #percentage outside the bar
            plt.text(i, count + max_count*0.02, f"{pct:.1f}%", ha= 'center', va='center', fontsize='12', fontweight ='bold')

        plot_path2 = os.path.join(results_dir, "patch-level_distribution.png")
        plt.savefig(plot_path2, bbox_inches='tight')

        logger.info(f"Saved Patch-Level distribution plot to: {plot_path2}")
        f.write(f"Plot saved to: {plot_path2}\n")

        f.flush()
        os.fsync(f.fileno())

        for foldNum in tqdm(range(0, numFolds), desc='Patch Classification for cross-validation', leave=True):
            print(" Starting Fold:", foldNum+1)
            # Train on all folds except the one specified (current fold is for testing)
            trainfolds = np.concatenate(folds[:foldNum] + folds[foldNum+1:])  # Select all patches except the current fold

            # # Select the current fold as the test fold
            test_fold = folds[foldNum]
            
            testfold_indices = patchIndices_[foldNum] # get the indices of the patches of wsi images for the remaining folds (training) in the original dataset
            train_val_fold_indices = np.concatenate(patchIndices_[:foldNum] + patchIndices_[foldNum+1:]) # # get the indices of the patches of wsi images for the excluded fold (evaluation) in the original dataset
            
            ##############Making dataset and dataloader for training, validation and testset using folds indices########
            # Split the train+val set into training and validation sets, alocate 20 percent of the indices for validation and the remaining for training
            
            val_split = 0.2
            train_size = int((1 - val_split) * len(train_val_fold_indices))
            all_trainin_validation_set = Subset(patchdataset, train_val_fold_indices) # will be used to obtain threshold for majority, softmax and gradcam++ for each fold
            len(all_trainin_validation_set)
            np.random.shuffle(train_val_fold_indices)  # Shuffle train+val to randomize training/validation split
            train_indices = train_val_fold_indices[:train_size]
            val_indices = train_val_fold_indices[train_size:]
            
            train_set = Subset(patchdataset, train_indices)

            len(train_set)
            val_set = Subset(patchdataset, val_indices)
            len(train_set)
            
            test_set = Subset(patchdataset, testfold_indices)
            len(test_set)
           
            # Debugging outputs
            print("train_folds:", trainfolds)
            print("test_fold:", test_fold)
            print("training set length:", len(train_set))
            print("training set length:", len(val_set))
            print("training set length:", len(test_set))
            print("all validation + training patches:", len(train_set)+len(val_set))
            print(" The total number of patches used for training and evaluation:", len(train_set)+len(val_set)+len(test_set))
            print("The total WSI images used for training and evaluation:", len(trainfolds) +len(test_fold))
            # print("list of test indices:", testfold_indices)
            
            f.write(f"train_folds: {trainfolds}\n")
            f.write(f"test_fold:   {test_fold}\n")
            f.write(f"WSIs train/val: {len(trainfolds)}\n")
            f.write(f"WSIs test:      {len(test_fold)}\n")
            f.write(f"total training patches:   {len(train_set)}\n")
            f.write(f"total validation patches: {len(val_set)}\n")
            f.write(f"total test patches:       {len(test_set)}\n")
            f.flush()
            os.fsync(f.fileno())
            
            # Model & Tokenizer Setup
            args, model = setup(args)  # Initialize the model for each fold
            
            # Get DataLoader for train, val, and test sets
            train_loader, val_loader, test_loader, tune_loader = get_loader(args, train_set, val_set, test_set, all_trainin_validation_set)
            
            # Train and evaluate
            train_epoch_losses, train_losses, val_losses, val_accuracies, best_acc, test_accuracy, patches_names, true_labels, predictions, top_patches_probabilities, wsi_names, patches_indices, patches_coordinates, patch_densenet_gradcam_importance_weights , fold_thresholds = train_test(
                args, model, train_loader, val_loader, test_loader, tune_loader, fold = foldNum + 1
            )
            
             # Store this fold’s thresholds
            all_fold_thresholds.append({
                "fold":     foldNum + 1,
                "majority": fold_thresholds["majority"],
                "softmax":  fold_thresholds["softmax"],
                "gradcam":  fold_thresholds["gradcam"]
            })

            # After all folds, save them to CSV
            # Save after each append (or once after the loop)
            th_df = pd.DataFrame(all_fold_thresholds)
            csv_path = os.path.join(results_dir, "all_fold_thresholds.csv")
            th_df.to_csv(csv_path, index=False)

            f.write(f"\nSaved all-fold thresholds to: {csv_path}\n")
            f.flush(); os.fsync(f.fileno())

            print("Per‑fold thresholds:\n", th_df)


             # 1) Build a patch‑level DataFrame for this fold’s test set
            # Build a patch‑level DataFrame for this fold’s TEST set, with everything
            test_df = pd.DataFrame({
                "Patch_Name":                        patches_names,
                "True_patch_Label":                  true_labels,
                "Prediction":                        predictions,
                "Top_probability":                   top_patches_probabilities,
                "GradCAM_Importance":                patch_densenet_gradcam_importance_weights,
                "WSI_Name":                          wsi_names,
                "Patch_Index":                       patches_indices,
                "Coordinate":                        patches_coordinates,
            })

            #  Persist it under results_dir
            csv_path = os.path.join(results_dir, f"fold{foldNum+1}_patch_level_test.csv")
            test_df.to_csv(csv_path, index=False)

            # Optional logging
            f.write(f"[Fold {foldNum+1}] Saved patch‑level test results to: {csv_path}\n")
            f.flush(); os.fsync(f.fileno())

            # Extract true labels and predictions for the test fold
            true_labels = np.array(true_labels)  # Ground truth for patches
            predictions = np.array(predictions)  # Predictions for patches
            
            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()
            
            # Calculate sensitivity and specificity
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Avoid division by zero
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero
            
            # Append to lists
            sensitivity_per_fold.append(sensitivity)
            specificity_per_fold.append(specificity)
            
            # Log fold-specific sensitivity and specificity
            f.write(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}\n")
            print(f"Fold {foldNum + 1} Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
            

            #Wsi-level classification based on the results obtained from patch-level and also tuned thresholds from training
            # Read in the thresholds just computed
            maj_th  = fold_thresholds["majority"]
            soft_th = fold_thresholds["softmax"]
            grad_th = fold_thresholds["gradcam"]

            # 2) Build slide‑level rows exactly as before, but *without* any grid search:
            rows = []
            for wsi, grp in test_df.groupby("WSI_Name"):
                # ground‑truth WSI label = 1 if any patch is malignant
                true_lbl   = int(grp["True_patch_Label"].any())

                # majority‐vote prediction
                maj_pred   = int(grp["Prediction"].mean() >= maj_th)

                # softmax‐weighted prediction (zero out 0.25<=p<=0.75)
                w_sm       = grp["Top_probability"].to_numpy().copy()
                w_sm[(w_sm >= 0.25) & (w_sm <= 0.75)] = 0
                soft_pred  = int((np.where(grp["Prediction"]==1, 1, -1) * w_sm).mean() > soft_th)

                # gradcam‐weighted prediction
                w_gc       = grp["GradCAM_Importance"].to_numpy()
                grad_pred  = int((np.where(grp["Prediction"]==1, 1, -1) * w_gc).mean() > grad_th)

                rows.append({
                    "WSI_Name":  wsi,
                    "True_Label": true_lbl,
                    "Maj_Pred":   maj_pred,
                    "Soft_Pred":  soft_pred,
                    "Grad_Pred":  grad_pred
                })

            # Turn it back into a DataFrame just like before
            slide_df = pd.DataFrame(rows)
            
            # --- Majority Voting ---
            y_true = slide_df["True_Label"].to_numpy()
            y_pred_maj = slide_df["Maj_Pred"].to_numpy()
            acc_maj, sens_maj, spec_maj, prec_maj, f1_maj = compute_wsi_metrics(y_true, y_pred_maj)

            # softmax voting Scheme
            y_pred_soft = slide_df["Soft_Pred"].to_numpy()
            acc_soft, sens_soft, spec_soft, prec_soft, f1_soft = compute_wsi_metrics(y_true, y_pred_soft)

            # Grad-CAM++ Voting scheme
            y_pred_gradcam = slide_df["Grad_Pred"].to_numpy()
            acc_gradcam, sens_gradcam, spec_gradcam, prec_gradcam, f1_gradcam = compute_wsi_metrics(y_true, y_pred_gradcam)

            
            f.write("\n=== WSI-Level Metrics ===\n")

            print("=== WSI-Level Metrics ===\n")
            print("--- Majority Voting ---")
            print(f"Accuracy:    {acc_maj:.4f}")
            print(f"Sensitivity: {sens_maj:.4f}")
            print(f"Specificity: {spec_maj:.4f}")
            print(f"Precision:   {prec_maj:.4f}")
            print(f"F1 Score:    {f1_maj:.4f}")

            print("\n--- Softmax Voting ---")
            print(f"Accuracy:    {acc_soft:.4f}")
            print(f"Sensitivity: {sens_soft:.4f}")
            print(f"Specificity: {spec_soft:.4f}")
            print(f"Precision:   {prec_soft:.4f}")
            print(f"F1 Score:    {f1_soft:.4f}")

            print("--- Grad-CAM++ Voting --- ")
            print(f"Accuracy:    {acc_gradcam:.4f}")
            print(f"Sensitivity: {sens_gradcam:.4f}")
            print(f"Specificity: {spec_gradcam:.4f}")
            print(f"Precision:   {prec_gradcam:.4f}")
            print(f"F1 Score:    {f1_gradcam:.4f}")

            

            f.write("\n-- Majority Voting --\n")
            f.write(f"Accuracy:    {acc_maj:.4f}\n")
            f.write(f"Sensitivity: {sens_maj:.4f}\n")
            f.write(f"Specificity: {spec_maj:.4f}\n")
            f.write(f"Precision:   {prec_maj:.4f}\n")
            f.write(f"F1 Score:    {f1_maj:.4f}\n")

            f.write("\n-- Softmax Voting --\n")
            f.write(f"Accuracy:    {acc_soft:.4f}\n")
            f.write(f"Sensitivity: {sens_soft:.4f}\n")
            f.write(f"Specificity: {spec_soft:.4f}\n")
            f.write(f"Precision:   {prec_soft:.4f}\n")
            f.write(f"F1 Score:    {f1_soft:.4f}\n")

            f.write("\n-- Grad-CAM++ Voting --\n")
            f.write(f"Accuracy:    {acc_gradcam:.4f}\n")
            f.write(f"Sensitivity: {sens_gradcam:.4f}\n")
            f.write(f"Specificity: {spec_gradcam:.4f}\n")
            f.write(f"Precision:   {prec_gradcam:.4f}\n")
            f.write(f"F1 Score:    {f1_gradcam:.4f}\n")

            # Ensure write is committed to disk
            f.flush()
            os.fsync(f.fileno())

            # Store fold-level metrics for aggregation later
            maj_accs.append(acc_maj)
            maj_sens.append(sens_maj)
            maj_specs.append(spec_maj)
            maj_precs.append(prec_maj)
            maj_f1s.append(f1_maj)

            soft_accs.append(acc_soft)
            soft_sens.append(sens_soft)
            soft_specs.append(spec_soft)
            soft_precs.append(prec_soft)
            soft_f1s.append(f1_soft)

            gradcam_accs.append(acc_gradcam)
            gradcam_sens.append(sens_gradcam)
            gradcam_specs.append(spec_gradcam)
            gradcam_precs.append(prec_gradcam)
            gradcam_f1s.append(f1_gradcam)

            # 4) (Optional) Save & log
            slide_csv = os.path.join(results_dir, f"fold{foldNum+1}_wsi_level_preds.csv")
            slide_df.to_csv(slide_csv, index=False)
            f.write(f"[Fold {foldNum+1}] Saved slide‑level predictions to: {slide_csv}\n")
            f.flush(); os.fsync(f.fileno())

            all_slide_dfs.append(slide_df)

            # Store results for this fold
            train_epoch_losses_per_fold.append(train_epoch_losses)  # List of losses per epoch
            train_losses_per_fold.append(train_losses)  # Final train loss per epoch
            val_losses_per_fold.append(val_losses)  # Validation losses per epoch
            val_accuracies_per_fold.append(val_accuracies)  # Validation accuracies per epoch
            test_accuracies_per_fold.append(test_accuracy)  # Single test accuracy value
            best_accuracies_per_fold.append(best_acc)  # Best validation accuracy
            
            
            # for wsi classification stores the information and predicition for each patch
            
             # Append patch-level data for this fold to the aggregated lists
            all_patch_names.extend(patches_names)
            all_patch_labels.extend(true_labels)
            all_patch_predictions.extend(predictions)
            all_top_patches_probabilities.extend(top_patches_probabilities)  # Store higher probabilities coming out of softmax for patches at each fold
            all_wsi_names.extend(wsi_names)
            all_patch_indices.extend(patches_indices)
            all_patch_coordinates.extend(patches_coordinates)
            all_patch_densenet_gradcam_importance_weights.extend(patch_densenet_gradcam_importance_weights)
            
            # Save fold-specific results to the text file
            f.write(f"\nFold {foldNum + 1} Results:\n")
            f.write(f"Train Epoch Losses: {train_epoch_losses}\n")
            f.write(f"Train Losses: {train_losses}\n")
            f.write(f"Validation Losses: {val_losses}\n")
            f.write(f"Validation Accuracies: {val_accuracies}\n")
            f.write(f"Best Validation Accuracy: {best_acc}\n")
            f.write(f"Test Accuracy: {test_accuracy}\n")
            f.flush()
            os.fsync(f.fileno())
            
            # Store test indices for this fold
            all_test_indices.extend(testfold_indices)
            
            save_plots_for_fold(foldNum, results_dir, train_losses, train_epoch_losses, val_losses, val_accuracies)
            
             # Clear references
            del train_loader, val_loader, test_loader  # Delete DataLoaders
            del model  # Delete model

            torch.cuda.empty_cache()  # Clear cache again
        
        ####################################################################################################################
        
        # Once all folds are done, calculate averages and standard deviations for each metric

        # Convert lists to numpy arrays for easier computation
        # Convert lists to numpy arrays for easier computation
        sensitivity_per_fold = np.array(sensitivity_per_fold)
        specificity_per_fold = np.array(specificity_per_fold)
        
        train_epoch_losses_per_fold = np.array(train_epoch_losses_per_fold, dtype=object)
        train_losses_per_fold = np.array(train_losses_per_fold, dtype=object)
        val_losses_per_fold = np.array(val_losses_per_fold, dtype=object)
        val_accuracies_per_fold = np.array(val_accuracies_per_fold, dtype=object)
        test_accuracies_per_fold = np.array(test_accuracies_per_fold)
        best_accuracies_per_fold = np.array(best_accuracies_per_fold)
        
        

        # Calculate averages and standard deviations for each metric
        logger.info("Calculating averages and standard deviations for each metric")
        
        sensitivity_avg = np.mean(sensitivity_per_fold)
        sensitivity_std = np.std(sensitivity_per_fold)
        
        specificity_avg = np.mean(specificity_per_fold)
        specificity_std = np.std(specificity_per_fold)
        
        train_epoch_avg = np.mean([np.mean(fold) for fold in train_epoch_losses_per_fold])
        train_epoch_std = np.std([np.mean(fold) for fold in train_epoch_losses_per_fold])

        train_loss_avg = np.mean([np.mean(fold) for fold in train_losses_per_fold])
        train_loss_std = np.std([np.mean(fold) for fold in train_losses_per_fold])

        val_loss_avg = np.mean([np.mean(fold) for fold in val_losses_per_fold])
        val_loss_std = np.std([np.mean(fold) for fold in val_losses_per_fold])

        val_acc_avg = np.mean([np.mean(fold) for fold in val_accuracies_per_fold])
        val_acc_std = np.std([np.mean(fold) for fold in val_accuracies_per_fold])

        test_acc_avg = np.mean(test_accuracies_per_fold)
        test_acc_std = np.std(test_accuracies_per_fold)

        best_acc_avg = np.mean(best_accuracies_per_fold)
        best_acc_std = np.std(best_accuracies_per_fold)

        # Save final averages and standard deviations to the text file
        f.write("\n=== Overall Cross-Validation Results ===\n")
        f.write(f"Train Epoch Loss Average: {train_epoch_avg:.4f} ± {train_epoch_std:.4f}\n")
        f.write(f"Train Loss Average: {train_loss_avg:.4f} ± {train_loss_std:.4f}\n")
        f.write(f"Validation Loss Average: {val_loss_avg:.4f} ± {val_loss_std:.4f}\n")
        f.write(f"Validation Accuracy Average: {val_acc_avg:.4f} ± {val_acc_std:.4f}\n")
        f.write(f"Best Validation Accuracy Average: {best_acc_avg:.4f} ± {best_acc_std:.4f}\n")
        f.write(f"Test Accuracy Average: {test_acc_avg:.4f} ± {test_acc_std:.4f}\n")
        f.write(f"Sensitivity Average: {sensitivity_avg:.4f} ± {sensitivity_std:.4f}\n")
        f.write(f"Specificity Average: {specificity_avg:.4f} ± {specificity_std:.4f}\n")
        f.flush()
        os.fsync(f.fileno())
        
        # Print the values to the console
        print("\n=== Overall Cross-Validation Results ===")
        print(f"Train Epoch Loss Average: {train_epoch_avg:.4f} ± {train_epoch_std:.4f}")
        print(f"Train Loss Average: {train_loss_avg:.4f} ± {train_loss_std:.4f}")
        print(f"Validation Loss Average: {val_loss_avg:.4f} ± {val_loss_std:.4f}")
        print(f"Validation Accuracy Average: {val_acc_avg:.4f} ± {val_acc_std:.4f}")
        print(f"Best Validation Accuracy Average: {best_acc_avg:.4f} ± {best_acc_std:.4f}")
        print(f"Test Accuracy Average: {test_acc_avg:.4f} ± {test_acc_std:.4f}")
        print(f"Sensitivity Average: {sensitivity_avg:.4f} ± {sensitivity_std:.4f}")
        print(f"Specificity Average: {specificity_avg:.4f} ± {specificity_std:.4f}")
        
        logger.info(" End of Cross-Validation")
        ###################################################################################################
        ###################################################################################################
        # Add a header for WSI-level results
        
        # WSI majority voting Classification using the patches information and predictions and WSI true labels
        # Save patch-level data to CSV
        logger.info(" Whole Slide Image (WSI) Classification using Majority voting and weighted average voting Using Grad-CAM.")
        
        
        # 2) Concatenate into one big DataFrame (60 rows total)
        all_slides = pd.concat(all_slide_dfs, ignore_index=True)

        all_csv = os.path.join(results_dir, "all_60_wsi_level_preds.csv")
        all_slides.to_csv(all_csv, index=False)
        logger.info(f"Saved all‑fold WSI‑level predictions to: {all_csv}")
        f.write(f"Saved concatenated slide‑level predictions to: {all_csv}\n")
        f.flush(); os.fsync(f.fileno())

        
        
        print(f"Length of Patch_Name: {len(all_patch_names)}")
        print(f"Length of True_patch_Label: {len(all_patch_labels)}")
        print(f"Length of Prediction: {len(all_patch_predictions)}")
        
        print(f"Length of Top_probability: {len(all_top_patches_probabilities)}")
        print(f"Length of WSI_Name: {len(all_wsi_names)}")
        print(f"Length of Index: {len(all_patch_indices)}")
        print(f"Length of Coordinate: {len(all_patch_coordinates)}")
        print(f"Length of Densenet Gradcam Patch Importance :{len(all_patch_densenet_gradcam_importance_weights)}")


        patch_data = pd.DataFrame({
            'Patch_Name': all_patch_names,
            'True_patch_Label': all_patch_labels,
            'Prediction': all_patch_predictions,
            'Top_probability' : all_top_patches_probabilities,
            'Densenet_Gradcam_Patch_Importance' : all_patch_densenet_gradcam_importance_weights,
            'WSI_Name': all_wsi_names,
            'Index': all_patch_indices,
            'Coordinate': all_patch_coordinates
        })
        
        
        # Save patch-level results to CSV in the output directory
        
        # 1) Save patch-level CSV
        patch_csv_path = os.path.join(results_dir, "patch_level_results.csv")
        patch_data.to_csv(patch_csv_path, index=False)
        f.write(f"\nSaved patch-level results to: {patch_csv_path}\n")
        f.flush(); os.fsync(f.fileno())

        #keep track of number of correct predictions/ wrong predicitons for each wsi
        wsi_stats = defaultdict(lambda: {'correct': 0, 'wrong': 0})

        # Count correct and wrong predictions per WSI
        for _, row in patch_data.iterrows():
            wsi = row['WSI_Name']
            true_label = int(row['True_patch_Label'])
            pred_label = int(row['Prediction'])

            if pred_label == true_label:
                wsi_stats[wsi]['correct'] += 1
            else:
                wsi_stats[wsi]['wrong'] += 1

        # Prepare summary list
        summary_data = []
        for wsi, stats in wsi_stats.items():
            correct = stats['correct']
            wrong = stats['wrong']
            total = correct + wrong
            correct_pct = round((correct / total) * 100, 2) if total > 0 else 0.0
            wrong_pct = round((wrong / total) * 100, 2) if total > 0 else 0.0

            summary_data.append({
                'Sample': wsi,
                'Correct_Predictions': correct,
                'Wrong_Predictions': wrong,
                'Correct_Percentage': correct_pct,
                'Wrong_Percentage': wrong_pct,
                'Total_Patches': total
            })

        # Save to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(results_dir, "wsi_level_accuracy_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)

        f.write(f"\n✅ Saved WSI-level accuracy summary to: {summary_csv_path}\n")
        f.flush(); os.fsync(f.fileno())

        # 4) Rename columns into your "final_results" schema
        final_results = all_slides.rename(columns={
            "True_Label": "True_WSI_Label_Normal",
            "Maj_Pred":   "Predicted_WSI_Label_Normal",
            "Soft_Pred":  "Predicted_WSI_Label_Weighted",
            "Grad_Pred":  "Predicted_WSI_Label_Weighted_Densenet_Gradcam"
        })

        # 5) Save the detailed WSI‑level CSV
        wsi_csv_path = os.path.join(results_dir, "wsi_level_results.csv")
        final_results.to_csv(wsi_csv_path, index=False)
        f.write(f"\nSaved WSI-level results to: {wsi_csv_path}\n")
        f.flush(); os.fsync(f.fileno())

        fold_metrics = pd.DataFrame({
            "Fold": list(range(1, len(maj_accs)+1)),
            "Maj_Acc": maj_accs,
            "Maj_Sens": maj_sens,
            "Maj_Spec": maj_specs,
            "Maj_Prec": maj_precs,
            "Maj_F1": maj_f1s,
            "Soft_Acc": soft_accs,
            "Soft_Sens": soft_sens,
            "Soft_Spec": soft_specs,
            "Soft_Prec": soft_precs,
            "Soft_F1": soft_f1s,
            "Grad_Cam_ACC": gradcam_accs,
            "Grad_Cam_Sens": gradcam_sens,
            "Grad_Cam_Spec": gradcam_specs,
            "Grad_Cam_Prec":gradcam_precs,
            "Grad_Cam_F1":gradcam_f1s,
        })
        fold_metrics_path = os.path.join(results_dir, "wsi_level_per_testfold_metrics.csv")
        fold_metrics.to_csv(fold_metrics_path, index=False)
        f.write(f"\nSaved per-fold WSI-level metrics to: {fold_metrics_path}\n")
        f.flush(); os.fsync(f.fileno())

        f.write("\n=== Final Cross-Fold WSI-Level Metrics ===\n")
        print("\n=== Final Cross-Fold WSI-Level Metrics ===")

        # 6) Compute per‑method WSI metrics
        
        for name, vals in zip(
            ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score"],
            [maj_accs, maj_sens, maj_specs, maj_precs, maj_f1s]
        ):
            summarize(vals, name, "Majority Voting", f)

        for name, vals in zip(
            ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score"],
            [soft_accs, soft_sens, soft_specs, soft_precs, soft_f1s]
        ):
            summarize(vals, name, "Softmax Voting", f)

        for name, vals, in zip(
            ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score"],
            [gradcam_accs, gradcam_sens, gradcam_specs, gradcam_precs, gradcam_f1s]
        ):
            summarize(vals, name, "Grad-CAM++ Voting", f)    

        # 7) Identify & save any mismatches across methods
        mismatches = final_results[
            (final_results["True_WSI_Label_Normal"] != final_results["Predicted_WSI_Label_Normal"]) |
            (final_results["True_WSI_Label_Normal"] != final_results["Predicted_WSI_Label_Weighted"]) |
            (final_results["True_WSI_Label_Normal"] != final_results["Predicted_WSI_Label_Weighted_Densenet_Gradcam"])]
        mismatch_csv_path = os.path.join(results_dir, "wsi_level_mismatches.csv")
        mismatches.to_csv(mismatch_csv_path, index=False)
        f.write(f"\nSaved all‑method mismatches to: {mismatch_csv_path} (count: {len(mismatches)})\n")
        f.flush(); os.fsync(f.fileno())

        logger.info("End of Whole Slide Image Classification")
        f.write("\nEnd of Whole Slide Image Classification\n")
        f.flush(); os.fsync(f.fileno())
        
    
       
        ########################################################################################################################################
        logger.info(" Box-Plot Visualization for patch level classification of Whole Slide Image")
        # visualize box-plots = drawing the box around the location of the patches in each wsi image normal with green and benign weith red color
        visualize_patch_locations(args, all_patch_names, all_patch_labels, all_patch_predictions, all_wsi_names, all_patch_coordinates)
    
        logger.info(" End of Box-Plot Visualization for patch level classification of Whole Slide Image")



if __name__ == "__main__":
    main()
            

