"""Simple example which introduces the basic usage of the module."""

import numpy as np
import torch
from aeon.datasets import load_classification

import altx

# Import data from aeon
X, y = load_classification("GunPoint")
y = y.astype(np.int8)
# The data is shuffled
learn_set, transform_set = X[:10], X[10:]
learn_classes, transform_classes = y[:10], y[10:]

# Seting the parameters
R, L, K = 25, 4, 1
extr_methods = [["mean_all"], ["mean", 0.05]]
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transform the data
alt = altx.ALT(learn_set, learn_classes, R=R, L=L, K=K, device=device)
alt.train()
transformed_set = alt.transform_set(
    transform_set,
    extr_methods=extr_methods,  # type: ignore[arg-type]
    test_classes=transform_classes,
    save_file_name="GunPoint_results.csv",
    save_file_mode="New file",
)
