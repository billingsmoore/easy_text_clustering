# Text Clustering

This repository contains tools to easily embed and cluster texts as well as label clusters semantically and produce visualizations of those labeled clusters. 

<center><img src="https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/jMKGaE_UnEfH3j8iZYXVN.png"></center>
<center>Clustering of texts in the <a href="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia">Cosmopedia dataset</a>.</center>

## How it works
The pipeline consists of several distinct blocks that can be customized. Each block uses existing standard methods and works quite robustly. The default pipeline is shown in the graphic below.

<center><img src="https://huggingface.co/datasets/lvwerra/admin/resolve/main/text-clustering.png"></center>
<center>Text clustering pipeline.</center>

As was true in the original repo, users can choose alternative models for Embeddings and labeling. Additionally, in this version, users can choose alternative algorithms for projection and clustering, and customize all hyperparameters for those algorithms.

## Install 

Install the library to get started:
```bash
pip install --upgrade easy_text_clustering
```

## Optimized Usage

The suggested way to use this module is to follow the code block below.

```python
from easy_text_clustering.clusterer import ClusterClassifier
from datasets import load_dataset

SAMPLE = 900 # select the number of samples from the data that you'd like to use

texts = load_dataset("billingsmoore/text-clustering-example-data", split="train").select(range(SAMPLE))["text"]

cc = ClusterClassifier()

# Perform optimization and fitting
cc.optimize_fit(texts=texts)

## View the results interactively
cc.show(interactive=True)

## Save the result
cc.save('./clustering-results')
```

## Basic Usage

Run pipeline and visualize results:

```python
from easy_text_clustering.clusterer import ClusterClassifier
from datasets import load_dataset

SAMPLE = 900

texts = load_dataset("billingsmoore/text-clustering-example-data", split="train").select(range(SAMPLE))["text"]

cc = ClusterClassifier()

# run the pipeline:
cc.fit(texts)

# show the results
cc.show()

# save 
cc.save("./clustering-results")
```

Load classifier and run inference:
```python
from easy_text_clustering.clusterer import ClusterClassifier

cc = ClusterClassifier()

# load state
cc.load("./clustering-results")

# visualize
cc.show()

# classify new texts with k-nearest neighbour search
cluster_labels, embeddings = cc.infer(some_texts, top_k=1)
```

If you want to customize the color scheme in the plot you can add (some version of) the following code before you run `cc.show()`:
```python
from cycler import cycler
import matplotlib.pyplot as plt

default_cycler = (cycler(color=[
    "0F0A0A",
    "FF6600",
    "FFBE00",
    "496767",
    "87A19E",
    "FF9200",
    "0F3538",
    "F8E08E",
    "0F2021",
    "FAFAF0"])
    )
plt.rc('axes', prop_cycle=default_cycler)
```
If you would like to customize the plotting further the easiest way is to customize or overwrite the `_show_mpl` and `_show_plotly` methods.

## Advanced Usage

You can fit and refit with your preferred algorithms and hyperparameters as shown below.

```python
from easy_text_clustering.clusterer import ClusterClassifier
from datasets import load_dataset

SAMPLE = 900

texts = load_dataset("billingsmoore/text-clustering-example-data", split="train").select(range(SAMPLE))["text"]

# initialize the ClusterClassifier to use TruncatedSVD with appropriate params
# also set the clustering to use KMeans clustering with appropriate params
cc = ClusterClassifier(
    projection_algorithm='tsvd', 
    projection_args={'n_components': 5, 'n_iter': 7, 'random_state': 42},
    clustering_algorithm='kmeans',
    clustering_args={'n_clusters': 2, 'random_state': 0, 'n_init': "auto"})

# run the pipeline:
cc.fit(texts)

# show the results
cc.show()

# if results are unsatisfactory, refit with new selections
cc.fit(
    projection_algorithm='pca', 
    projection_args={'n_components': 3},
    clustering_algorithm='hdbscan',
    clustering_args={'min_cluster_size': 10})

cc.show()


# still unsatisfied? you can keep projections, but change clustering params
cc.fit(clustering_args={'min_cluster_size': 25})

cc.show()

# save when done
cc.save("./clustering-results")
```

## Credits

This is project was created, and is maintained, by [@billingsmoore](https://github.com/billingsmoore).

This project is a fork of ['huggingface/text-clustering'](https://github.com/huggingface/text-clustering). All images in this README come from their repo. The following changes have been made to the codebase:

1. Projection and clustering algorithms can now be selected by the user as appropriate for their use-case.
2. Each algorithm's relevant hyperparamaters can be provided by the user as a dictionary, without having to store all possible hyperparameters.
3. Visualizations can now be done interactively in 3 dimensions.
4. The pipeline can be run and re-run with new hyperparameters, or even new algorithm selections for projections and/or clustering without having to re-perform computationally expensive embedding or projections unnecessarily. 
5. Texts can be batched into groups prior to clustering.
6. A simple automated test suite has been added to the repo.
7. An optimization method has been added to allow for easy optimization of hyperparameters

Additionally, a substantial amount of documentation has been added to this repository for both the new functionality and the original functionality, improving readability and usability. This documentation is available as comments in the code and below in this README.

## Extensive Documentation for ClusterClassifier

Object with parameters for embedding generation, dimensionality reduction, clustering, and summarization of text data.

### Args:

- **batch_size** (`int`, default=1):  
  The number of samples to process in each batch. Setting a larger batch size can speed up the embedding generation but may require more memory.

- **embed_model_name** (`str`, default="all-MiniLM-L6-v2"):  
  The name of the pre-trained embedding model to use for generating embeddings from input text.

- **embed_device** (`str`, default='cpu'):  
  The device to use for embedding generation. Options are:
  - `'cpu'`: Use CPU for embeddings (default).
  - `'cuda'`: Use GPU for embeddings (if available).

- **embed_batch_size** (`int`, default=64):  
  The number of samples per batch during the embedding generation process. A larger batch size can improve efficiency but requires more memory.

- **embed_max_seq_length** (`int`, default=512):  
  The maximum sequence length for the embedding model. Texts longer than this will be truncated.

- **embed_agg_strategy** (`str`, optional):  
  Aggregation strategy for embeddings when the model supports multiple tokens (e.g., 'mean', 'sum', or `None`). Default is `None`.

- **projection_algorithm** (`str`, default='umap'):  
  Algorithm for dimensionality reduction of embeddings. Options include:
  - `'pca'`: Principal Component Analysis.
  - `'tsvd'`: Truncated Singular Value Decomposition.
  - `'umap'`: Uniform Manifold Approximation and Projection (default).

- **projection_args** (`dict`, default={}):  
  Additional arguments passed to the dimensionality reduction algorithm. For example, `n_neighbors` for UMAP.

- **clustering_algorithm** (`str`, default='dbscan'):  
  Clustering algorithm to apply to the projections. Options include:
  - `'dbscan'`: Density-Based Spatial Clustering of Applications with Noise.
  - `'hdbscan'`: Hierarchical Density-Based Spatial Clustering.
  - `'optics'`: Ordering Points to Identify Clustering Structure.
  - `'kmeans'`: K-means clustering.

- **clustering_args** (`dict`, default={}):  
  Additional arguments for the clustering algorithm, such as `eps` for DBSCAN or `n_clusters` for KMeans.

- **summary_create** (`bool`, default=True):  
  Whether to create summaries for each cluster. If set to `True`, summaries will be generated for the cluster centers.

- **summary_model** (`str`, default="mistralai/Mixtral-8x7B-Instruct-v0.1"):  
  The model to use for creating cluster summaries. This model generates textual summaries for each cluster based on its center.

- **topic_mode** (`str`, default='multiple_topics'):  
  Mode for topic extraction in summaries. Options include:
  - `'multiple_topics'`: Extract multiple topics per cluster.
  - `'single_topic'`: Extract a single topic per cluster (e.g., educational score).

- **summary_n_examples** (`int`, default=10):  
  The number of examples (texts) to send to the summary model for each cluster when creating summaries.

- **summary_chunk_size** (`int`, default=420):  
  The maximum number of tokens per chunk when sending text to the summary model. Large clusters may require splitting the text into smaller chunks.

- **summary_model_token** (`bool`, default=True):  
  Whether to use a token to authenticate with the summary model. If `True`, the `summary_model_token` is used to authenticate API requests.

- **summary_template** (`str`, optional, default=DEFAULT_TEMPLATE):  
  The template used for formatting the summary request to the summary model.

- **summary_instruction** (`str`, optional, default=DEFAULT_INSTRUCTION):  
  The instruction given to the summary model when generating summaries for the clusters.

### Attributes:

- **embeddings** (`numpy.ndarray`):  
  The embeddings for the input texts.

- **faiss_index** (`faiss.Index`):  
  The FAISS index for fast retrieval of nearest neighbors for clustering.

- **cluster_labels** (`numpy.ndarray`):  
  The cluster labels assigned to each document in the input dataset.

- **texts** (`list`):  
  The input texts that were passed to the classifier.

- **projections** (`numpy.ndarray`):  
  The 2D or 3D projections of the input texts for visualization after dimensionality reduction.

- **mapper** (`object`):  
  The dimensionality reduction mapper object used, such as a UMAP or PCA model.

- **id2label** (`dict`):  
  Mapping from document ID to cluster label.

- **label2docs** (`dict`):  
  Mapping from cluster label to list of document indices.

- **embed_model** (`SentenceTransformer`):  
  The SentenceTransformer model used for embedding generation.

##### Raises:

- **ValueError**:  
  If an invalid `projection_algorithm` or `clustering_algorithm` is provided.

##### Example:

```python
# Example usage
my_clusterer = ClusterClassifier(
    batch_size=16,
    embed_model_name="all-MiniLM-L6-v2",
    clustering_algorithm="kmeans",
    clustering_args={'n_clusters': 5},
)

# Fit the model to the texts and get the embeddings, labels, and summaries
my_clusterer.fit(texts)

# Visualize the clustering results
my_clusterer.show()

# Save the model
my_clusterer.save("./cluster_classifier_5_clusters")
```

### Methods

#### optimize_fit ( texts=None, optimization_trials=None )

Combines hyperparameter optimization and model fitting in a single method. It first tunes hyperparameters for dimensionality reduction and clustering using Optuna and then fits the model to the provided or existing texts.

##### Parameters:
- **texts** (*list, optional*):  
  A list of input texts to process. If provided, it overrides the current `self.texts`.  
  Defaults to `None` (uses `self.texts`).

- **optimization_trials** (*int, optional*):  
  The number of optimization trials for hyperparameter tuning.  
  Defaults to `None` (uses `self.optimization_trials`).

##### Returns:
- **None**

##### Notes:
- The `optimize` method is called to identify the best hyperparameters for:
  - **Dimensionality reduction** (e.g., UMAP).
  - **Clustering** (e.g., HDBSCAN).
- The `fit` method is subsequently invoked to apply these parameters and process the text data.
- This method streamlines the workflow by combining hyperparameter tuning and model fitting in one step.

##### Example:
```python
# Initialize the clustering object
my_clusterer = ClusterClassifier()

# Perform optimization and fitting
my_clusterer.optimize_fit(texts=["Text 1", "Text 2", "Text 3"], optimization_trials=20)

# The model now has optimized parameters and is fitted to the provided texts
```

--- 


#### optimize ( texts=None, optimization_trials=None )

    Optimizes hyperparameters for dimensionality reduction (UMAP) and clustering (HDBSCAN) using Optuna. This method performs hyperparameter tuning by maximizing the silhouette score, a measure of clustering quality.

##### Parameters:

- **texts** (`list`, optional):  
  A list of input texts to embed and optimize. If provided and different from `self.texts`, it replaces the current `self.texts`, and embeddings are recalculated. Defaults to `None` (uses `self.texts`).

- **optimization_trials** (`int`, optional):  
  The number of optimization trials to perform. If not provided, the value of `self.optimization_trials` is used. Defaults to `None`.

##### Returns:

- **None**  

##### Notes:

1. **Objective Function:**  
   - **UMAP Parameters:**  
     - `umap_n_neighbors` (`int`): Number of neighbors to consider for UMAP. Suggested range: 5–50.  
     - `umap_min_dist` (`float`): Minimum distance between points in the low-dimensional space. Suggested range: 0.0–1.0.  
     - `umap_metric` (`str`): Metric for UMAP distance calculations. Options: `['euclidean', 'cosine']`.  
   - **HDBSCAN Parameters:**  
     - `hdbscan_min_cluster_size` (`int`): Minimum cluster size. Suggested range: 5–100.  
     - `hdbscan_min_samples` (`int`): Number of samples in a neighborhood for a point to be a core point. Suggested range: 1–10.  
     - `hdbscan_metric` (`str`): Metric for HDBSCAN distance calculations. Options: `['euclidean', 'cosine']`.  
   - Evaluates clustering using the **silhouette score**, which requires at least 2 clusters. Assigns a score of `-1` for single-cluster results.

2. **Optimization Process:**  
   - An **Optuna study** is created to maximize the silhouette score over `optimization_trials`.  
   - The best UMAP and HDBSCAN parameters are stored in `self.projection_args` and `self.clustering_args`, respectively.

3. **Outputs:**  
   - Prints the best parameters and the corresponding silhouette score.

4. **Updates Model Configuration:**  
   - Sets `self.projection_algorithm` to `'umap'` and updates its arguments.  
   - Sets `self.clustering_algorithm` to `'hdbscan'` and updates its arguments.

##### Example:

```python
# Example usage of optimize
cluster_classifier.optimize(
    texts=["Sample text 1", "Sample text 2", "Sample text 3"],
    optimization_trials=20
)

# Prints:
# Best Parameters: {'umap_n_neighbors': 15, 'umap_min_dist': 0.1, ...}
# Best Score: 0.75
```

---

#### fit ( texts=None, batch_size=None, projection_algorithm=None, projection_args=None, clustering_algorithm=None, clustering_args=None )

    Perform the complete process of fitting the model, which includes embedding the texts, projecting the embeddings into a lower-dimensional space, clustering the projections, and optionally summarizing the clusters.

##### Parameters:

- **texts** (`list`, optional):  
  List of input texts to process. If not provided, the method will use the existing `self.texts`. This parameter is required for the first time fitting or when new texts need to be processed.

- **batch_size** (`int`, optional):  
  The number of texts to process in a single batch. If provided, this will override the default `self.batch_size`. Setting a larger batch size can speed up processing, but may require more memory.

- **projection_algorithm** (`str`, optional):  
  The dimensionality reduction technique to apply to the embeddings. Options include:
  - `'pca'`: Principal Component Analysis.
  - `'tsvd'`: Truncated Singular Value Decomposition.
  - `'umap'`: Uniform Manifold Approximation and Projection (default is `self.projection_algorithm`).

- **projection_args** (`dict`, optional):  
  Additional parameters to pass to the projection algorithm. For example, `n_neighbors` for UMAP or `n_components` for PCA.

- **clustering_algorithm** (`str`, optional):  
  The clustering algorithm to apply to the projected embeddings. Options include:
  - `'dbscan'`: Density-Based Spatial Clustering of Applications with Noise.
  - `'kmeans'`: K-means clustering (default is `self.clustering_algorithm`).

- **clustering_args** (`dict`, optional):  
  Additional parameters to pass to the clustering algorithm, such as `eps` for DBSCAN or `n_clusters` for KMeans.

##### Returns:

- **tuple**:  
  A tuple containing:
  - **embeddings** (`numpy.ndarray`):  
    The embeddings for the input texts generated by the embedding model.
  - **cluster_labels** (`numpy.ndarray`):  
    The cluster labels assigned to each input text after clustering.
  - **cluster_summaries** (`dict`, optional):  
    The summaries of each cluster, if `self.summary_create` is `True`. This field will contain the generated summaries for each cluster.

##### Raises:

- **ValueError**:  
  If the provided `batch_size` or `projection_algorithm` is invalid.

---

#### infer ( texts, top_k=1 )

    Infers the cluster labels for a given list of text inputs by finding the most common cluster label among the nearest neighbors of each text in the FAISS index.

##### Parameters:

- **texts** (`list`):  
  List of text data to be classified into clusters. These texts will be embedded and classified into one of the existing clusters based on their nearest neighbors in the FAISS index.

- **top_k** (`int`, optional):  
  The number of nearest neighbors to consider for each input text when predicting the cluster label. The default value is 1, meaning only the nearest neighbor will be considered.

##### Returns:

- **inferred_labels** (`list`):  
  A list of predicted cluster labels for each input text. Each element corresponds to the cluster label of a text in the input list.

- **embeddings** (`numpy.ndarray`):  
  The computed embeddings for each input text, which are generated using the same model as during training.

##### Example:

```python
inferred_labels, embeddings = cluster_classifier.infer(texts, top_k=3)
```

##### Notes:

- This method relies on the FAISS index for fast nearest neighbor search and uses the `top_k` nearest neighbors to determine the most likely cluster for each text.
- The embeddings for the input texts are calculated as part of the inference process and can be used for further analysis or visualization.

---

#### save ( folder )

    Saves various components of the model and related data to the specified folder. If the folder doesn't exist, it is created. This function saves embeddings, projections, cluster labels, texts, and optional cluster summaries to disk in a structured format.

##### Parameters:

- **folder** (`str`):  
  The path to the folder where the model data will be saved. If the folder doesn't exist, it will be created.

##### Returns:

- **None**

##### Notes:

- The function saves the following files in the specified folder:
    - `embeddings.npy`: The model's embeddings as a NumPy binary file.
    - `faiss.index`: The FAISS index object for nearest neighbor search.
    - `projections.npy`: The projections of the data points as a NumPy binary file.
    - `cluster_labels.npy`: The cluster labels associated with the data points.
    - `texts.json`: The raw input texts associated with the embeddings.
    - `mistral_prompt.txt`: A text file containing the default instruction prompt for the model.
    - `cluster_summaries.json` (optional): Summaries of the clusters, saved if available.

- The function uses NumPy and FAISS libraries to save arrays and indexes efficiently.

##### Example:

```python
cluster_classifier.save('./model_data')
```

---

#### load ( folder )

    Loads model data and related information from the specified folder. If the folder doesn't exist, an error is raised. This function restores embeddings, projections, cluster labels, texts, and optional cluster summaries. It also infers additional information based on the loaded data.

##### Parameters:

- **folder** (`str`):  
  The path to the folder from which the model data will be loaded. The folder must contain the necessary files.

##### Raises:

- **ValueError**:  
  If the specified folder does not exist.

##### Returns:

- **None**

##### Notes:

- The function loads the following files from the specified folder:
    - `embeddings.npy`: The model's embeddings as a NumPy binary file.
    - `faiss.index`: The FAISS index object for nearest neighbor search.
    - `projections.npy`: The projections of the data points as a NumPy binary file.
    - `cluster_labels.npy`: The cluster labels associated with the data points.
    - `texts.json`: The raw input texts associated with the embeddings.
    - `cluster_summaries.json` (optional): Summaries of the clusters, loaded if available.

- The function also infers the following based on the loaded data:
    - `id2cluster`: A mapping from document index to cluster label.
    - `label2docs`: A mapping from cluster label to a list of document indices belonging to that cluster.
    - `cluster_centers`: A dictionary of cluster centers, computed as the mean of the projections for each cluster.

##### Example:

```python
cluster_classifier.load('./model_data')
```

---

Here's the documentation for the `show` method:

---

#### show ( interactive=False ) 

    Visualizes the projections of the data points, optionally in 2D or 3D, with cluster labels and associated text content. The method displays the projections using either Matplotlib or Plotly for interactive or static plotting.

##### Parameters:

- **interactive** (`bool`, optional):  
  If `True`, the plot is displayed using Plotly for interactivity (zoom, hover, exploration). If `False`, a static plot is shown using Matplotlib. Default is `False`.

##### Returns:

- **None**

##### Notes:

- If the number of componenets in the projections is 3 or more, a 3D plot is created, where `X`, `Y`, and `Z` represent the projections in 3-dimensional space.
- If  number of componenets in the projections is 2, a 2D plot is created, with `X` and `Y` representing the projections in 2-dimensional space.
- The content of each data point (up to 1024 characters) is displayed in the plot, with long text wrapped to fit within the plot's space.
- The `labels` represent the cluster labels for each data point.
- The function relies on the `projections` (data points' projections), `cluster_labels` (assigned clusters), and `texts` (the content for each data point).

##### Visualization Methods:
- For interactive plotting, **Plotly** is used, allowing zoom, hover, and exploration.
- For static plotting, **Matplotlib** is used for a simpler, non-interactive visualization.

##### Example:

```python
# Static plot (using Matplotlib)
cluster_classifier.show(interactive=False)

# Interactive plot (using Plotly)
cluster_classifier.show(interactive=True)
```

---
