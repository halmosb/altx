# Usage

`altx` transforms time series into a fixed-length feature vector by discovering *linear laws* — conserved quantities of the data. These features can then be used with any downstream classifier or anomaly detector.

---

## Conceptual overview

The pipeline has two phases:

**Train** — For each configuration triplet `(R, L, K)`, every training instance is scanned with a sliding window of length `R` (step `K`). Each window is embedded into a symmetric `L×L` Hankel matrix `S`. The eigenvector corresponding to the smallest absolute eigenvalue of `S` is stored as a *law*. All laws are collected in `model.Ps`.

**Transform** — For a test instance, windows of length `L` are projected onto every stored law. The resulting score values are partitioned by training class and then summarised into scalar features by an *extraction method*. The final feature vector has length `len(RLK) × noc × n_methods × m`.

---

## Importing

```python
import torch
from altx import ALT as Altx
from altx import ExtractMethods
```

`ALT` is an alias for the main `Altx` class and is the recommended entry point.

---

## Input data format

`altx` accepts both `torch.Tensor` and `numpy.ndarray` inputs.

| Shape | Meaning |
|-------|---------|
| `(N, T)` | `N` univariate instances, each of length `T` |
| `(N, m, T)` | `N` multivariate instances, `m` channels each of length `T` |

- The first dimension always indexes instances.
- For univariate data the channel dimension `m` is optional and will be added automatically.

Class labels must be a 1-D array of integers (one per instance).

---

## Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | `int` or `list[int]` | `5` | Embedding dimension — size of the Hankel matrix. Larger values capture longer-range structure. |
| `R` | `int`, `list[int]`, or `None` | `2*L-1` | Window length. Must satisfy `(R-1) % (2*L-2) == 0`. `None` means use the default `2*L-1`. |
| `K` | `int` or `list[int]` | `1` | Step between consecutive windows during training. Larger values reduce training time. |
| `device` | `str` or `torch.device` | `"cpu"` | Computation device. Use `"cuda"` for GPU acceleration. |

`L`, `R`, and `K` can all be lists of equal length to run multiple configurations simultaneously. When one is a scalar and another is a list, the scalar is broadcast.

### The `R` constraint

`R` must satisfy:

$$
(R - 1) \bmod (2L - 2) = 0
$$

Equivalently, $R = s(2L-2) + 1$ for some positive integer step $s$.

For `L=3`: valid `R` values are `5` (s=1), `9` (s=2), `13` (s=3), …
For `L=4`: valid `R` values are `7` (s=1), `13` (s=2), `19` (s=3), …

---

## Step 1 — Initialise the model

```python
import torch
from altx import ALT as Altx

torch.manual_seed(42)

# 6 univariate training instances of length 50, two classes
train_data    = torch.randn(6, 50)
train_classes = torch.tensor([0, 0, 0, 1, 1, 1])

model = Altx(train_data, train_classes, L=3, K=1)
print(f"tau (instances): {model.tau}")
print(f"m (channels):    {model.m}")
print(f"noc (classes):   {model.noc}")
print(f"RLK triplets:    {model.RLK}")
print(f"class_labels:    {model.class_labels}")
```

Output:

```
tau (instances): 6
m (channels):    1
noc (classes):   2
RLK triplets:    ((5, 3, 1),)
class_labels:    tensor([0, 1])
```

`R` defaulted to `2*3-1 = 5`.

---

## Step 2 — Train

```python
model.train()
model.print_number_of_laws()
```

Output:

```
Number of laws for r = 5, l = 3, k = 1:
(tensor([0., 1.]), tensor([138, 138]))
```

Each class has 138 laws stored (one per sliding window across all training instances).

### Memory cleanup

To free training data from memory after training (useful for large datasets):

```python
model.train(cleanup=True)
# model.train_set, model.train_classes, etc. are now deleted
```

After cleanup, `train()` cannot be called again on the same model instance.

---

## Step 3 — Transform

### Single instance

`transform(z, extr_methods)` returns a 1-D feature tensor.

```python
test_instance = torch.randn(50)
features = model.transform(test_instance, [["mean", 0.05]])
print(f"feature shape:  {features.shape}")
print(f"feature values: {features}")
```

Output:

```
feature shape:  torch.Size([2])
feature values: tensor([0.0108, 0.0131])
```

The feature vector has length `len(RLK) × noc × n_methods × m = 1 × 2 × 1 × 1 = 2`.

### Batch of instances

`transform_set(test_set, extr_methods)` transforms an entire set at once and returns a 2-D tensor of shape `(n_instances, n_features)`.

```python
torch.manual_seed(42)
test_data    = torch.randn(4, 50)
test_classes = torch.tensor([0, 1, 0, 1])

features = model.transform_set(
    test_data,
    extr_methods=[["mean", 0.05], ["var", 0.1]],
    test_classes=test_classes,
)
print(f"output shape: {features.shape}")
print(features)
```

Output:

```
output shape: torch.Size([4, 4])
tensor([[0.0086, 0.0008, 0.0118, 0.0011],
        [0.0111, 0.0015, 0.0107, 0.0011],
        [0.0077, 0.0004, 0.0088, 0.0009],
        [0.0124, 0.0014, 0.0092, 0.0023]])
```

Feature length: `1 × 2 × 2 × 1 = 4` (`mean` and `var` are the two methods).

### Saving features to CSV

Pass `save_file_name` and `save_file_mode` to write features to disk:

```python
model.transform_set(
    test_data,
    extr_methods=[["mean", 0.05]],
    test_classes=test_classes,
    save_file_name="features.csv",
    save_file_mode="New file",
)
```

`save_file_mode` options:

| Mode | Behaviour |
|------|-----------|
| `"New file"` | Creates a new CSV (overwrites any existing file). |
| `"Append feature"` | Adds new feature columns to an existing CSV. |
| `"Append instance"` | Appends new rows to an existing CSV. |

---

## Step 4 — Save and load the model

```python
model.save("my_model.pkl")

loaded = Altx.load("my_model.pkl")
loaded.device = model.device  # device is not stored; restore it manually
```

**Important:** `load()` does not restore `device`. Always set `loaded.device` before calling `transform`.

The saved file contains the extracted laws (`Ps`), `RLK`, `m`, `noc`, and `class_labels`. The training data is **not** saved.

---

## Multivariate time series

Pass a 3-D array `(N, m, T)`:

```python
torch.manual_seed(42)
train_data    = torch.randn(6, 3, 50)   # 3 channels
train_classes = torch.tensor([0, 0, 0, 1, 1, 1])

model = Altx(train_data, train_classes, L=3, K=1)
print(f"m (channels): {model.m}")       # → 3
model.train()

test_instance = torch.randn(3, 50)
features = model.transform(test_instance, [["mean", 0.05]])
print(f"feature shape: {features.shape}")
print(f"feature values: {features}")
```

Output:

```
m (channels): 3
feature shape: torch.Size([6])
feature values: tensor([0.0064, 0.0179, 0.0146, 0.0104, 0.0102, 0.0156])
```

Feature length: `1 × 2 × 1 × 3 = 6` (one feature per class per channel).

---

## Multiple `(R, L, K)` configurations

Providing lists enables training with multiple window scales simultaneously:

```python
torch.manual_seed(42)
train_data    = torch.randn(6, 50)
train_classes = torch.tensor([0, 0, 0, 1, 1, 1])

model = Altx(train_data, train_classes, L=[3, 4], K=1)
print(f"RLK triplets: {model.RLK}")
model.train()

test = torch.randn(50)
features = model.transform(test, [["mean", 0.05]])
print(f"feature shape: {features.shape}")
```

Output:

```
RLK triplets: ((5, 3, 1), (7, 4, 1))
feature shape: torch.Size([4])
```

Feature length: `2 × 2 × 1 × 1 = 4`.

---

## Variable-length instances

If training instances have different lengths, pass a `train_length` tensor:

```python
torch.manual_seed(42)
train_data    = torch.randn(4, 100)    # padded to length 100
train_classes = torch.tensor([0, 0, 1, 1])
train_length  = torch.tensor([60, 70, 55, 80], dtype=torch.int)

model = Altx(train_data, train_classes, train_length=train_length, L=3, K=1)
```

During training, each instance will only be scanned up to its stated length.

---

## GPU acceleration

```python
torch.manual_seed(42)
train_data    = torch.randn(6, 50)
train_classes = torch.tensor([0, 0, 0, 1, 1, 1])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Altx(train_data, train_classes, L=3, K=1, device=device)
model.train()

test_instance = torch.randn(50)
features = model.transform(test_instance, [["mean", 0.05]])
# Move back to CPU for downstream use
features_cpu = features.cpu()
```

---

## Extraction methods

Extraction methods summarise the law-projection scores into scalar features. They are passed as a list of `[method, percentile]` pairs to `transform` and `transform_set`.

### `mean`

```python
extr_methods = [["mean", 0.05]]
```

Computes the mean over the rows that fall at or near the `q`-th percentile of the squared projection scores. The percentile acts as a soft filter that focuses on small values — those projections that are most nearly *orthogonal* to the law, indicating the law is most active.

**When to use:** general-purpose baseline; captures the average alignment strength.

```python
import torch
from altx import ExtractMethods

torch.manual_seed(42)
F = torch.randn(5, 20, 2).abs() + 0.5
result = ExtractMethods.extract(F, [["mean", 0.05]])
print(f"mean (q=0.05): {result}")
# → tensor([[0.3538, 0.3744]])

result = ExtractMethods.extract(F, [["mean", 0.5]])
print(f"mean (q=0.50): {result}")
# → tensor([[1.6190, 1.5376]])
```

### `var`

```python
extr_methods = [["var", 0.05]]
```

Computes the variance of the projection scores at the given percentile. Measures how consistently the law is expressed across the windows of the instance.

**When to use:** when you want to distinguish stable patterns from noisy ones.

```python
result = ExtractMethods.extract(F, [["var", 0.05]])
print(f"var (q=0.05): {result}")
# → tensor([[0.0103, 0.0059]])
```

### `excess_kurtosis`

```python
extr_methods = [["excess_kurtosis", 0.1]]
```

Computes the excess kurtosis (Fisher's definition: kurtosis − 3) of the projection scores at the given percentile. Positive values indicate a heavy-tailed distribution of scores; negative values indicate a flat distribution. Returns zeros when the input variance is zero (e.g. constant signal).

**When to use:** detecting whether a law is activated in sharp, isolated bursts versus spread uniformly.

```python
result = ExtractMethods.extract(F, [["excess_kurtosis", 0.1]])
print(f"excess_kurtosis (q=0.1): {result}")
# → tensor([[-0.7103, -1.7724]])
```

### `nth_moment`

```python
extr_methods = [["4th_moment", 0.1]]
```

Computes the `n`-th central moment of the projection scores at the given percentile. Replace `n` with any positive integer (e.g. `"2nd_moment"`, `"3rd_moment"`, `"4th_moment"`).

- 1st moment: always zero (it is the central moment).
- 2nd moment: equivalent to biased variance.
- 4th moment: sensitivity to extremes, similar to kurtosis.

**When to use:** when you need fine-grained statistical control over which distributional property to capture.

```python
result = ExtractMethods.extract(F, [["4th_moment", 0.1]])
print(f"4th_moment (q=0.1): {result}")
# → tensor([[4.2127e-05, 2.2910e-03]])

result = ExtractMethods.extract(F, [["2nd_moment", 0.1]])
print(f"2nd_moment (q=0.1): {result}")
# → tensor([[0.0043, 0.0432]])
```

### `mean_all`

```python
extr_methods = [["mean_all"]]
```

Computes the mean of **all** squared projection scores across the entire instance, without any percentile selection. No second parameter is needed; `None` is filled in automatically.

**When to use:** a fast global summary when you do not want to tune a percentile.

```python
result = ExtractMethods.extract(F, [["mean_all", None]])
print(f"mean_all: {result}")
# → tensor([[2.1676, 1.8516]])
```

### Combining multiple methods

All methods can be combined freely. The output rows are stacked in the order you specify them:

```python
extr_methods = [["mean", 0.05], ["var", 0.1], ["mean_all", None]]
result = ExtractMethods.extract(F, extr_methods)
print(f"output shape: {result.shape}   # (n_methods=3, m=2)")
print(result)
```

Output:

```
output shape: torch.Size([3, 2])   # (n_methods=3, m=2)
tensor([[0.3538, 0.3744],
        [0.0054, 0.0540],
        [2.1676, 1.8516]])
```

Using multiple extraction methods increases the feature vector length proportionally.

---

## How the feature vector is structured

The final feature vector from `transform` is a flattened tensor with one value per combination of:

```
(RLK triplet, class, extraction method, channel)
```

Concretely, for a model with `L=[3, 4]`, `noc=2`, methods `["mean", "var"]`, and `m=1`:

```
feature index 0 → RLK=(5,3,1), class=0, method=mean,  channel=0
feature index 1 → RLK=(5,3,1), class=0, method=var,   channel=0
feature index 2 → RLK=(5,3,1), class=1, method=mean,  channel=0
feature index 3 → RLK=(5,3,1), class=1, method=var,   channel=0
feature index 4 → RLK=(7,4,1), class=0, method=mean,  channel=0
...
```

The column names in the saved CSV encode the same structure:
`RLK(5, 3, 1)|C0|Fmean(q0.05)|m0`

---

## End-to-end example

Below is a complete workflow on synthetic data. Class 0 contains sinusoidal signals; class 1 contains linear-trend signals.

```python
import torch
import numpy as np
from altx import ALT as Altx

torch.manual_seed(42)
np.random.seed(42)

T = 100
t = torch.linspace(0, 4 * 3.14159, T)

# Class 0: sinusoidal signals with slight frequency variation
class0 = torch.stack([
    torch.sin(t * (1 + 0.1 * i)) + 0.3 * torch.randn(T)
    for i in range(10)
])

# Class 1: linear-trend signals
class1 = torch.stack([
    torch.linspace(0, 1, T) + 0.3 * torch.randn(T)
    for _ in range(10)
])

# Split into train / test
train_data    = torch.cat([class0[:7], class1[:7]], dim=0)
train_classes = torch.tensor([0] * 7 + [1] * 7)
test_data     = torch.cat([class0[7:], class1[7:]], dim=0)
test_classes  = torch.tensor([0] * 3 + [1] * 3)

print(f"train: {train_data.shape},  test: {test_data.shape}")

# Create and train a model with two window scales
model = Altx(train_data, train_classes, L=[3, 4], K=1)
print(f"RLK triplets: {model.RLK}")
model.train()

# Transform using two extraction methods
extr_methods = [["mean", 0.05], ["var", 0.1]]
features = model.transform_set(test_data, extr_methods)

print(f"\nFeature matrix shape: {features.shape}")
print(f"  = len(RLK) × noc × n_methods × m = "
      f"{len(model.RLK)} × {model.noc} × 2 × {model.m}")
print("\nFeatures per test instance:")
for i in range(len(test_data)):
    label = "sine" if test_classes[i] == 0 else "linear"
    print(f"  instance {i} ({label:6s}): {features[i].numpy().round(4)}")
```

Output:

```
train: torch.Size([14, 100]),  test: torch.Size([6, 100])
RLK triplets: ((5, 3, 1), (7, 4, 1))

Feature matrix shape: torch.Size([6, 8])
  = len(RLK) × noc × n_methods × m = 2 × 2 × 2 × 1

Features per test instance:
  instance 0 (sine  ): [0.0016 0.     0.0018 0.     0.0013 0.     0.0016 0.    ]
  instance 1 (sine  ): [0.0012 0.     0.0015 0.     0.001  0.     0.0013 0.    ]
  instance 2 (sine  ): [0.0014 0.     0.0014 0.     0.0012 0.     0.0014 0.    ]
  instance 3 (linear): [0.0009 0.     0.0011 0.     0.0007 0.     0.0008 0.    ]
  instance 4 (linear): [0.0014 0.     0.0014 0.     0.001  0.     0.001  0.    ]
  instance 5 (linear): [0.0013 0.     0.0014 0.     0.0011 0.     0.0009 0.    ]
```

The resulting feature matrix can be passed directly to any scikit-learn classifier or to any other Python, MATLAB, or R-based analysis pipeline.
