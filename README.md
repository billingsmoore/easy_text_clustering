# Text Clustering

This repository contains tools to easily embed and cluster texts as well as label clusters semantically and produce visualizations of those labeled clusters. 

<center><img src="https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/jMKGaE_UnEfH3j8iZYXVN.png"></center>
<center>Clustering of texts in the <a href="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia">Cosmopedia dataset</a>.</center>

This project is a fork of ['huggingface/text-clustering'](https://github.com/huggingface/text-clustering). All images in this README come from their repo. The following changes have been made to the codebase:

1. Projection and clustering algorithms can now be selected by the user as appropriate for their use-case.
2. Each algorithm's relevant hyperparamaters can be provided by the user as a dictionary, without having to store all possible hyperparameters.
3. Visualizations can now be done interactively in 3 dimensions.
4. The pipeline can be run and re-run with new hyperparameters, or even new algorithm selections for projections and/or clustering without having to re-perform computationally expensive embedding or projections unnecessarily. 
5. Texts can be batched into groups prior to clustering.
6. A simple automated test suite has been added to the repo.

Additionally, a substantial amount of documentation has been added to this repository for both the new functionality and the original functionality, improving readability and usability. This documentation is available as comments in the code and in a standalone document.

[Documentation can be found here](/docs/ClusterClassifier.md)

## How it works
The pipeline consists of several distinct blocks that can be customized and the whole pipeline can run in a few minutes on a consumer laptop. Each block uses existing standard methods and works quite robustly. The default pipeline is shown in the graphic below.

<center><img src="https://huggingface.co/datasets/lvwerra/admin/resolve/main/text-clustering.png"></center>
<center>Text clustering pipeline.</center>

As was true in the original repo, users can choose alternative models for Embeddings and labeling. Additionally, in this version, users can choose alternative algorithms for projection and clustering, and customize all hyperparameters for those algorithms.

## Install 

Install the library to get started:
```bash
pip install --upgrade easy_text_clustering
```

## Basic Usage

Run pipeline and visualize results:

```python
from easy_text_clustering.src.clusterer import ClusterClassifier
from datasets import load_dataset

SAMPLE = 100_000

texts = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train").select(range(SAMPLE))["text"]

cc = ClusterClassifier()

# run the pipeline:
embs, labels, summaries = cc.fit(texts)

# show the results
cc.show()

# save 
cc.save("./cc_100k")
```

Load classifier and run inference:
```python
from easy_text_clustering.src.clusterer import ClusterClassifier

cc = ClusterClassifier()

# load state
cc.load("./cc_100k")

# visualize
cc.show()

# classify new texts with k-nearest neighbour search
cluster_labels, embeddings = cc.infer(some_texts, top_k=1)
```

If you want to reproduce the color scheme in the plot above you can add the following code before you run `cc.show()`:
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

```python
from easy_text_clustering.src.clusterer import ClusterClassifier
from datasets import load_dataset

SAMPLE = 100_000

texts = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train").select(range(SAMPLE))["text"]

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
cc.save("./cc_100k")
```
