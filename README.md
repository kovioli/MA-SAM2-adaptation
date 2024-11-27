# SAM2 Particle Detection Pipeline for SHREC-2020
This repository contains a pipeline for detecting and localizing particles in cryo-electron tomograms using a fine-tuned SAM2 model. The folders `sam2` and `sam2_configs` are cloned from the [official repository](https://github.com/facebookresearch/sam2) of Meta. The pipeline works with the SHREC-2020 dataset, which should be downloaded and organized in a folder structure containing reconstruction tomograms and class masks.

## Pipeline Overview

Train segmentation models for each particle type using SAM2
Generate segmentation masks for tomograms
Process segmentations to detect particle positions
Evaluate results against ground truth

## Key Files

- train_classexploration.py - Fine-tunes SAM2 for particle segmentation
- eval_multi_particle_seg.py - Generates segmentation masks
- cluster_motl.py - Converts segmentations to particle coordinates
- particle_list_evaluation.py - Evaluates detection accuracy

## Setup

- Download SHREC-2020 dataset
- Update paths in config.py
- Install required dependencies
- Adapt paths in scripts to match your setup
- Run training and inference pipeline

The implementation includes data loaders, loss functions, evaluation metrics, and TensorBoard logging for tracking training progress.

## Files:
### _3D_HEAD
- `model_3D.py`: Contains the Head-Promptable SAM2 model with trainable image encoder and mask decoder with the attached U-Net
- `unet_check.py`: Contains the checkerboard-corrected U-Net implementation used as the head model

### FINETUNE
- `cluster_motl.py`: Contains utilities for processing and analyzing 3D tomographic data, with the main function cluster_and_clean performing particle detection in tomograms. The function takes prediction data from a neural network, applies thresholding and clustering to identify particle positions, and generates a motive list file in csv format containing particle coordinates and properties. Supporting functions handle MRC file I/O, cluster analysis, and coordinate transformations.
- `config.py`: Contains configuration settings for a deep learning model, specifically using SAM2 (Segment Anything Model 2). It defines key parameters including device settings, hyperparameters, model configurations for different sizes (tiny to large), logging settings, and dataset specifications with particle type mappings.
- `eval_multi_particle_seg.py`: Performs model evaluation on tomographic data using the HeadFinetuneModel with SAM2. It processes predictions for different particle types, calculates Dice and IoU metrics, and saves results as MRC files and a JSON summary. The evaluation uses a mapping of particle IDs to specific training timestamps. The segmentations are used as the basis for `cluster_motl.py` to generate motive lists.
- `metrics.py`: Contains the loss functions and evaluation metrics for image segmentation tasks, combining implementations from DeePiCt and SAM-adap projects. The main classes include DiceCoefficient for evaluation metrics, DiceCoefficientLoss (noted as not working), and helper functions for calculating IoU and Dice scores across different thresholds.
- `particle_list_evaluation.py`: This file contains evaluation code adapted from the [SHREC-2020 challenge](https://dataverse.nl/file.xhtml?fileId=296681&version=1.0) for assessing particle detection accuracy. The main function evaluate_particle calculates F1 scores by comparing predicted particle positions against ground truth locations, using an occupancy map to validate spatial predictions across different particle types.
- `shrec_dataset.py`: Contains the MRCDataset class for loading and preprocessing tomographic data, with functionality to create concatenated datasets for training and validation. It handles both reconstruction and grand model input types, implements 3D context by stacking adjacent slices, and performs resizing operations. Works for the SHREC-2020 dataset.
- `train_classexploration.py`: This script implements a training pipeline for fine-tuning the SAM2 model on tomographic particle detection. It handles model training, validation, early stopping, and logging using TensorBoard. The training loop processes each particle type sequentially, using mixed precision training and monitoring key metrics (loss, IoU, Dice) for model checkpointing.
