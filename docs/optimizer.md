# **Class Documentation: `Optimizer`**

The `Optimizer` class provides a pipeline for text embedding, dimensionality reduction, clustering, and hyperparameter optimization. It integrates various machine learning techniques to facilitate unsupervised learning workflows, including UMAP for dimensionality reduction and HDBSCAN for clustering.

---

## **Class: Optimizer**

### **Initialization**
```python
Optimizer(embed_batch_size=64, embed_device='cpu', embed_model_name='all-MiniLM-L6-v2', embed_max_seq_length=512)
```

### **Description**
Initializes the `Optimizer` class with configurations for text embedding.

### **Parameters**
- **`embed_batch_size` (int)**:  
  Number of texts processed in a batch during embedding.  
  *Default*: 64  

- **`embed_device` (str)**:  
  Device for embedding computation. Options: `'cpu'` or `'cuda'`.  
  *Default*: `'cpu'`  

- **`embed_model_name` (str)**:  
  Pre-trained model name from `SentenceTransformers` to use for embedding.  
  *Default*: `'all-MiniLM-L6-v2'`  

- **`embed_max_seq_length` (int)**:  
  Maximum sequence length for embeddings. Texts longer than this will be truncated.  
  *Default*: 512  

---

## **Methods**

### **`fit()`**
```python
fit(embeddings, optimization_trials=100, sample_size=None)
```

#### **Description**
Performs hyperparameter optimization for UMAP (dimensionality reduction) and HDBSCAN (clustering) using Optuna. It evaluates clustering performance and stores the best parameters.

#### **Parameters**
- **`embeddings` (list or numpy.ndarray)**:  
  Pre-computed text embeddings or raw texts (strings). If raw texts are provided, embeddings are computed automatically.  

- **`optimization_trials` (int)**:  
  Number of trials for Optuna optimization.  
  *Default*: 100  

- **`sample_size` (int, optional)**:  
  Number of data points sampled for optimization. If `None`, uses the entire dataset.  
  *Default*: None  

#### **Returns**
- **`projection_args` (dict)**:  
  Best UMAP hyperparameters:  
  - `'n_neighbors'`: Number of neighbors.  
  - `'min_dist'`: Minimum distance between points in the reduced space.  
  - `'metric'`: Metric used for UMAP distance calculation.  

- **`clustering_args` (dict)**:  
  Best HDBSCAN hyperparameters:  
  - `'min_cluster_size'`: Minimum size of clusters.  
  - `'metric'`: Metric used for HDBSCAN distance calculation.  
  - `'cluster_selection_epsilon'`: Epsilon parameter for HDBSCAN.  

---

### **`normalize()`**
```python
normalize(value, min_val, max_val)
```

#### **Description**
Normalizes a value to the range `[0, 1]`.

#### **Parameters**
- **`value` (float)**:  
  Value to be normalized.  

- **`min_val` (float)**:  
  Minimum value of the original range.  

- **`max_val` (float)**:  
  Maximum value of the original range.  

#### **Returns**
- **`normalized_value` (float)**:  
  Normalized value in the range `[0, 1]`.  

---

### **`compute_score()`**
```python
compute_score(data, cluster_labels, weights=(1, 1, 1))
```

#### **Description**
Calculates a composite score for clustering performance by combining:
- Silhouette score (weighted between -1 and 1).  
- Calinski-Harabasz index (higher values indicate better clustering).  
- Davies-Bouldin index (lower values indicate better clustering).  

#### **Parameters**
- **`data` (numpy.ndarray)**:  
  Data points in reduced dimensions (output from UMAP).  

- **`cluster_labels` (numpy.ndarray)**:  
  Cluster assignments for each data point.  

- **`weights` (tuple of 3 floats)**:  
  Weights for combining the three scores:  
  - Silhouette score weight.  
  - Calinski-Harabasz index weight.  
  - Davies-Bouldin index weight.  
  *Default*: `(1, 1, 1)`  

#### **Returns**
- **`composite_score` (float)**:  
  Weighted composite clustering score.  

---

### **`embed()`**
```python
embed(texts)
```

#### **Description**
Generates text embeddings using a pre-trained model from `SentenceTransformers`. The embeddings are normalized to unit length for consistent usage in downstream tasks.

#### **Parameters**
- **`texts` (list of str)**:  
  Input texts for embedding generation.  

#### **Returns**
- **`embeddings` (numpy.ndarray)**:  
  Generated text embeddings as a NumPy array.  

#### **Additional Features**
- Embedding computation uses GPU (`'cuda'`) if available, falling back to CPU if not.  
- Displays a progress bar for embedding generation.  

---

## **Usage Example**

```python
from optimizer import Optimizer

# Initialize the optimizer
optimizer = Optimizer(embed_batch_size=32, embed_device='cuda')

# Input texts
texts = ["This is the first text.", "Another example sentence.", "More text data."]

# Generate embeddings and optimize
embeddings = optimizer.embed(texts)
projection_args, clustering_args = optimizer.fit(embeddings, optimization_trials=50)

# Print the best hyperparameters
print("Best UMAP parameters:", projection_args)
print("Best HDBSCAN parameters:", clustering_args)
```