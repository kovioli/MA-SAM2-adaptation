#  SAM2 Adaptation
The implementations for reproducing the experiments in the paper are under branches `SHREC` for the SHREC-2020 dataset and under `EMPIAR-10988` for the EMPIAR-10988 dataset.

## Minimal setup (training and picking evaluation)


1. Download checkpoints via
    ```bash
    bash sam2/download_ckpts.sh
    ```
2. Create, activate venv and install requirements
    ```
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3. Export python path
    ```
    export PYTHONPATH=''
    ```
3. Add dataset with the structure
    ```
    dataset
    ├── GT_particle_lists
    │   ├── TS_0001_cyto_ribosomes.csv
    │   ├── TS_0002_cyto_ribosomes.csv
    │   └── ...
    ├── tomograms
    │   ├── TS_0001.mrc
    │   ├── TS_0002.mrc
    │   └── ...
    └── {class_name}
        ├── TS_0001.mrc
        ├── TS_0002.mrc
        └── ...
    ```
    where `tomograms` is the folder containing the input tomograms and `{class_name}` is the folder containing the class samples.

4. Adapt config file (`FINETUNE/config.py`)
    - `DATA_DIR`: path to the dataset
    - `PREDICTION_CLASS`: the folder name from the dataset structure (same as `{class_name}`)
    - `SAVE_DIR`: path to the save directory (checkpoints, training logs, predictions)
    
5. Run the training script
    ```
    python FINETUNE/train.py
    ```
6. Run the prediction script to create segmentation maps and calculate voxel-wise metrics
    ```
    python FINETUNE/eval_segmentation.py --train_timestamp=...
    ```
    To be adapted:
    - `EVAL_DS_LIST`: list of datasets to evaluate (in `FINETUNE/config.py`)
7. Run coordinate-wise picking evaluation
    ```
    python FINETUNE/eval_picking.py --train_timestamp=...
    ```
    Example with all options:
    ```
    python eval_picking.py --train_timestamp "DDMMYYYY_HH:MM:SS" --threshold 0.75 --min_cluster_size 5 --max_cluster_size 100 --save_prediction false
    ```