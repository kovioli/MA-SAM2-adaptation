# Setup
1. Download checkpoints via
    ```bash
    bash sam2/download_ckpts.sh
    ```
2. Create, activate venv and install requirements
    ```
    python3 -m venv venv
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
    