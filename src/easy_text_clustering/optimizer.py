# Import necessary libraries for text embedding, dimensionality reduction, and clustering
from sentence_transformers import SentenceTransformer  # Embedding model
from umap import UMAP  # Dimensionality reduction
from sklearn.cluster import HDBSCAN  # Clustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # Evaluation metrics
import optuna  # Hyperparameter optimization
from optuna.pruners import MedianPruner  # Early stopping for optimization
import random  # Sampling for optimization
import numpy as np  # Array operations
import torch  # GPU availability check

class Optimizer():
    """
    A class to perform text embedding, dimensionality reduction, clustering, and hyperparameter optimization.
    The workflow includes generating embeddings, tuning parameters for UMAP and HDBSCAN, 
    and evaluating clustering performance using multiple metrics.
    """

    def __init__(
        self,
        embed_batch_size=64,
        embed_device='cpu',
        embed_model_name='all-MiniLM-L6-v2',
        embed_max_seq_length=512,
    ):
        """
        Initializes the Optimizer class with embedding configuration.

        Args:
            embed_batch_size (int): Batch size for generating embeddings. Default is 64.
            embed_device (str): Device for embedding computation ('cpu' or 'cuda'). Default is 'cpu'.
            embed_model_name (str): Name of the SentenceTransformer model. Default is 'all-MiniLM-L6-v2'.
            embed_max_seq_length (int): Maximum sequence length for embedding generation. Default is 512.
        """
        self.embed_batch_size = embed_batch_size
        self.embed_device = embed_device
        self.embed_model_name = embed_model_name
        self.embed_max_seq_length = embed_max_seq_length

        # Store hyperparameters for dimensionality reduction and clustering
        self.umap_args = None
        self.hdbscan_args = None

    def fit(self, embeddings, optimization_trials=100, sample_size=None):
        """
        Optimizes UMAP and HDBSCAN hyperparameters using Optuna, evaluates clustering performance, 
        and stores the best parameters.

        Args:
            embeddings (list or numpy.ndarray): Input embeddings or raw text for optimization.
            optimization_trials (int): Number of trials for hyperparameter optimization. Default is 100.
            sample_size (int): Number of data points to sample for optimization. Defaults to full dataset.

        Returns:
            projection_args (dict): Best UMAP parameters.
            clustering_args (dict): Best HDBSCAN parameters.
        """
        if type(embeddings[0]) is str:
            print('Embedding texts...')
            embeddings = self.embed(embeddings)

        if sample_size is None:
            sample_size = len(embeddings)

        # Random sampling of data points for optimization
        if sample_size < len(embeddings):
            data = random.sample(list(embeddings), sample_size)
        else:
            data = embeddings

        def objective(trial):
            """
            Objective function for Optuna to suggest and evaluate UMAP and HDBSCAN parameters.
            """
            # Suggest UMAP hyperparameters
            n_neighbors = trial.suggest_int('umap_n_neighbors', 5, 50)
            min_dist = trial.suggest_float('umap_min_dist', 0.0, 1.0)
            metric = trial.suggest_categorical('umap_metric', ['euclidean', 'cosine'])

            # Suggest HDBSCAN hyperparameters
            min_cluster_size = trial.suggest_int('hdbscan_min_cluster_size', 5, 100)
            hdbscan_metric = trial.suggest_categorical('hdbscan_metric', ['euclidean', 'cosine'])
            cluster_selection_epsilon = trial.suggest_float('cluster_selection_epsilon', 0, 1.0)

            try:
                # Dimensionality reduction using UMAP
                umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
                umap_embedding = umap_model.fit_transform(data)

                # Clustering using HDBSCAN
                hdbscan_model = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    metric=hdbscan_metric,
                    cluster_selection_epsilon=cluster_selection_epsilon
                )
                cluster_labels = hdbscan_model.fit_predict(umap_embedding)

                # Evaluate clustering performance using silhouette score
                if len(np.unique(cluster_labels)) > 1:
                    score = self.compute_score(umap_embedding, cluster_labels)
                else:
                    score = -1  # Assign a low score for single-cluster results
            except Exception as e:
                print(e)
                score = -1  # Assign a low score to failed trials

            return score

        # Optimize using Optuna
        best = -1
        while best == -1:
            study = optuna.create_study(direction='maximize', pruner=MedianPruner())
            study.optimize(objective, n_trials=optimization_trials)
            best = study.best_value

            if best == -1:
                print(f"Study failed with sample size: {sample_size}")
                sample_size = int(sample_size // 1.25)
                if sample_size > 0:
                    data = random.sample(list(embeddings), sample_size)
                else:
                    print("No optimal hyperparameters found.")
                    return {}, {}

        # Store the best parameters
        projection_args = {
            'n_neighbors': study.best_params['umap_n_neighbors'],
            'min_dist': study.best_params['umap_min_dist'],
            'metric': study.best_params['umap_metric']
        }
        clustering_args = {
            'min_cluster_size': study.best_params['hdbscan_min_cluster_size'],
            'metric': study.best_params['hdbscan_metric'],
            'cluster_selection_epsilon': study.best_params['cluster_selection_epsilon']
        }

        self.umap_args = projection_args
        self.hdbscan_args = clustering_args
        return projection_args, clustering_args

    def normalize(self, value, min_val, max_val):
        """
        Normalize a value to the range [0, 1].
        """
        return (value - min_val) / (max_val - min_val)

    def compute_score(self, data, cluster_labels, weights=(1, 1, 1)):
        """
        Compute a composite clustering score using silhouette, CH index, and DB index.
        """
        silhouette_weight, ch_weight, db_weight = weights

        # Silhouette Score
        silhouette = silhouette_score(data, cluster_labels)
        silhouette = (silhouette + 1) / 2

        # Calinski-Harabasz Index
        ch_index = calinski_harabasz_score(data, cluster_labels)
        ch_normalized = self.normalize(ch_index, 0, 1000)

        # Davies-Bouldin Index
        db_index = davies_bouldin_score(data, cluster_labels)
        db_normalized = 1 - self.normalize(db_index, 0, 10)

        return (
            silhouette_weight * silhouette +
            ch_weight * ch_normalized +
            db_weight * db_normalized
        )

    def embed(self, texts):
        """
        Generate text embeddings using SentenceTransformer.
        """
        try:
            device = 'cuda' if torch.cuda.is_available() else self.embed_device
            model = SentenceTransformer(self.embed_model_name, device=device)
            model.max_seq_length = self.embed_max_seq_length

            return model.encode(
                texts,
                batch_size=self.embed_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as e:
            print(f"Embedding failed: {e}")
            raise
