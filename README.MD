# Scale-free and Unbiased Transformer for scRNA-seq Cell Type Annotation

## 1. Paper Reference
- **Title**: *Scale-free and unbiased transformer with tokenization for cell type annotation from single-cell RNA-seq data*  
- **Authors**: Zhang H, Jiang Z, Zhang S, et al.  
- **Journal**: Pattern Recognition  
- **Publication Year**: 2025  
- **DOI Link**: [https://doi.org/10.1016/j.patcog.2025.111724](https://doi.org/10.1016/j.patcog.2025.111724)  

### Citation Format
Zhang H, Jiang Z, Zhang S, et al. Scale-free and unbiased transformer with tokenization for cell type annotation from single-cell RNA-seq data[J]. Pattern Recognition, 2025: 111724.


## 2. Data Input Format
The model requires preprocessed scRNA-seq data in **AnnData (.h5ad)** format. Below are the detailed requirements for input data:

### 2.1 File Path
Place the .h5ad file in the following directory:  
`./cls_data/{DATASET_NAME}/preprocessed_{DATASET_NAME}.h5ad`  
- `{DATASET_NAME}`: Name of your dataset (e.g., "Kidney", "Baron")

### 2.2 Data Structure in .h5ad File
The .h5ad file must contain two core components:  
1. **Gene Expression Matrix (adata.X)**:  
   - Rows: Individual cells (each row represents one cell).  
   - Columns: Genes (each column represents one gene).  
   - The matrix can be stored as a sparse matrix (e.g., `scipy.sparse.csr_matrix`) or dense matrix; the script automatically handles both formats.  

2. **Cell Type Labels (adata.obs)**:  
   - The .h5ad file’s `obs` attribute must include a column that records the cell type label for each cell.  
   - The script supports common label column names by default (no manual modification required for these cases):  
     - "celltype" (for datasets like "Zheng68K").  
     - "cell_type" (for datasets like "Baron", "Segerstolpe").  
     - "cell_ontology_class" (for other datasets, e.g., "Kidney", "Mat").  
   - If your label column has a different name, modify the `y = adata.obs['your_label_column_name']` line in the `data_produce()` function.




## 3. Environment Dependencies
Install the following packages before running the script (compatible with Python 3.8+):  
```bash
# Basic data processing
pip install numpy pandas scipy
# Single-cell data tools
pip install scanpy anndata
# PyTorch (for model training)
pip install torch torchvision torchaudio  # Match CUDA version to your GPU if using GPU
# Scikit-learn (for evaluation and data splitting)
pip install scikit-learn
# Visualization and utilities
pip install matplotlib tqdm argparse
```


## 4. Script Running Commands
The script uses command-line arguments to configure hyperparameters. Below are **basic running commands** and **common parameter adjustments**:

### 4.1 Basic Command (Default Parameters)
Run the script with default settings (dataset: "Kidney", batch size: 64, epochs: 25, 5-fold cross-validation):  
```bash
python class.py  # Replace "your_script_name.py" with the actual name of your script
```

### 4.2 Custom Parameter Configuration
Modify key parameters via command-line arguments (examples below):  
```bash
# Example 1: Use "Baron" dataset, batch size=32, 10 epochs
python class.py --dataset Baron --batch_size 32 --num_epoch 10

# Example 2: Adjust learning rate, transformer heads, and disable cosine scheduler
python class.py --learning_rate 0.001 --num_mulhead 4 --scheduler_type False

# Example 3: Modify MLP hidden layer size and dropout rate
python class.py --hidden_size_1 1024 --hidden_size_2 128 --dropout 0.3
```

### 4.3 Key Parameter Explanations
| Parameter          | Type    | Default Value | Description                                                                 |
|---------------------|---------|---------------|-----------------------------------------------------------------------------|
| `--dataset`         | str     | "Kidney"      | Name of the dataset (matches the folder name in `./cls_data/`).             |
| `--batch_size`      | int     | 64            | Batch size for training (reduce if GPU memory is insufficient).             |
| `--num_epoch`       | int     | 25            | Number of training epochs.                                                  |
| `--num_fold`        | int     | 5             | Number of folds for cross-validation.                                       |
| `--learning_rate`   | float   | 0.0005        | Initial learning rate (Adam optimizer).                                     |
| `--scheduler_type`  | bool    | True          | Whether to use CosineAnnealingLR scheduler (True/False).                    |
| `--num_mulhead`     | int     | 8             | Number of heads in the transformer encoder.                                 |
| `--hidden_size_1`   | int     | 512           | Size of the first hidden layer in the MLP classifier.                       |
| `--dropout`         | float   | 0.2           | Dropout rate for the MLP classifier (prevents overfitting).                 |


## 5. Output Files
After running the script, the following output files will be generated:

### 5.1 Preprocessed Data
Stored in `./input_data/{DATASET_NAME}/`:  
- `data_x.npy`: 3D tensor of tokenized gene expression features (shape: [num_cells, num_tokens, token_dim]).  
- `data_y.npy`: Encoded cell type labels (0-based integers).  

### 5.2 Model Checkpoints
Stored in `./ckpts/{DATASET_NAME}/`:  
- Files named `{timestamp}_fold{fold_num}.pt`: Saved weights of the best model (by validation F1 score) for each cross-validation fold.  
- `timestamp`: Current time (e.g., "08-26_14-30-00") to avoid overwriting.  


