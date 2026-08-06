# Learning-Free Laplacian Kernel Modeling for Joint Imputation of Multi-Energy Load Data in Integrated Energy Systems

Official Python implementation for the paper:

> **Learning-Free Laplacian Kernel Modeling for Joint Imputation of Multi-Energy Load Data in Integrated Energy Systems**, accepted by **IEEE Transactions on Smart Grid**.

## Requirements

The code was tested with Python 3.x. The main required packages are:

```bash
numpy
scikit-learn
```

They can be installed by:

```bash
pip install numpy scikit-learn
```

## File Description

```text
data-imputation-main/
|-- main.py                         # Main implementation of the proposed imputation method
|-- evaluation.py                   # Evaluation using MAE, RMSE, and MAPE
|-- data/
|   |-- mask_point_missing.npy      # Mask for the point-wise missing scenario
|   `-- mask_contiguous_missing.npy # Mask for the contiguous missing scenario
`-- README.md
```

## Data Preparation

Please place the processed multi-energy load data in the `data/` folder:

```text
./data/data.npy
```

The input data should be a chronologically ordered two-dimensional NumPy array with the shape:

```text
[number of time steps, number of energy load types]
```

The data are divided chronologically into training, validation, and testing subsets with a ratio of 8:1:1. In this paper, the load types correspond to electricity, cooling, and heating loads. The missing masks have the same shape as the testing subset, where `1` denotes a missing entry to be imputed.

## Running the Code

Run the proposed method by:

```bash
python main.py
```

The point-wise missing scenario is used by default. The contiguous missing scenario can also be selected in `main.py`.

The imputed result will be saved in the `result/` folder.

## Evaluation

After running `main.py`, evaluate the imputation result by:

```bash
python evaluation.py
```

The evaluation script reports MAE, RMSE, and MAPE on the missing entries. Please make sure that the mask and result file paths in `evaluation.py` are consistent with the scenario and output file used in `main.py`.

## Main Hyperparameters

The main hyperparameters are set in `main.py`:

- `tau_t`: temporal neighborhood size of the Laplacian kernel
- `gamma_t`: weight of the temporal Laplacian regularizer
- `gamma_s`: weight of the cross-load Laplacian regularizer
- `lam`: ADMM penalty parameter
- `eta`: weight of the data fidelity term
- `admm_max_iter`: maximum number of ADMM iterations
- `inner_max_iter`: maximum number of inner proximal-gradient iterations
