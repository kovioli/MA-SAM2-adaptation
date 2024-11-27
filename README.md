# SAM2 Particle Detection Pipeline for EMPIAR-10988
This repository contains a pipeline for detecting and localizing particles in cryo-electron tomograms using a fine-tuned SAM2 model. The folders `sam2` and `sam2_configs` are cloned from the [official repository](https://github.com/facebookresearch/sam2) of Meta. The pipeline works with the [EMPIAR-10988 dataset](https://ftp.ebi.ac.uk/empiar/world_availability/10988/data/VPP/), which should be downloaded and organized in a folder structure containing reconstruction tomograms and class masks.

## Pipeline Overview

- `train_dc.py`: Train segmentation models using SAM2
- `eval_dc.py`: Generate segmentation masks for tomograms
- `picking_pipeline.py`: Process segmentations to detect particle positions and evaluate results against ground truth

## Key Files

- train_classexploration.py - Fine-tunes SAM2 for particle segmentation
- eval_multi_particle_seg.py - Generates segmentation masks
- cluster_motl.py - Converts segmentations to particle coordinates
- particle_list_evaluation.py - Evaluates detection accuracy

## Setup

- Download EMPIAR-10988 dataset
- Update config.py
- Install required dependencies
- Adapt paths in scripts to match your setup
- Run training and inference pipeline

The implementation includes data loaders, loss functions, evaluation metrics, and TensorBoard logging for tracking training progress.

## Files

###  _3D_HEAD
- `model_3D.py`: Contains the Head-Promptable SAM2 model with trainable image encoder and mask decoder with the attached U-Net
- `unet_check.py`: Contains the checkerboard-corrected U-Net implementation used as the head model

###  DATA_POSTPROCESSOR
- `picking_pipeline`: Complete picking pipeline doing the segmentation and the picking in one go.

### FINETUNE
- `config.py`: Contains configuration settings for a deep learning model, specifically using SAM2 (Segment Anything Model 2). It defines key parameters including device settings, hyperparameters, model configurations for different sizes (tiny to large), logging settings, and dataset specifications with particle type mappings.
- `dataset_mrc.py`: Contains the MRCDataset class for loading and preprocessing tomographic data, with functionality to create concatenated datasets for training and validation. It handles both reconstruction and grand model input types, implements 3D context by stacking adjacent slices, and performs resizing operations. Works for the EMPIAR-10988 dataset.
- `eval_dc.py`: This script evaluates a 3D neural network model's performance on tomographic data. It loads pre-trained models, processes multiple tomogram datasets, and calculates segmentation metrics (Dice coefficient and IoU). The script saves predictions as MRC files and stores evaluation results in JSON format.
- `metrics.py`: Contains the loss functions and evaluation metrics for image segmentation tasks, combining implementations from DeePiCt and SAM-adap projects. The main classes include DiceCoefficient for evaluation metrics, DiceCoefficientLoss (noted as not working), and helper functions for calculating IoU and Dice scores across different thresholds.
- `train_dc.py`: Contains the TrainingPipeline class, which implements a complete training workflow for a segmentation model on tomographic data. It handles model initialization, training loops, validation, early stopping, and logging. The pipeline includes mixed-precision training, tensorboard logging, and tracks metrics like Dice score and IoU.
