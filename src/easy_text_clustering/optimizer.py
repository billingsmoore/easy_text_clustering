from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import optuna
from optuna.pruners import MedianPruner
import random
import numpy as np
import torch

class Optimizer():
    def __init__(
            self,
            embed_batch_size=64,
            embed_device='cpu',
            embed_model_name='all-MiniLM-L6-v2',
            embed_max_seq_length=512):
        
        self.embed_batch_size = embed_batch_size
        self.embed_device = embed_device
        self.embed_model_name = embed_model_name
        self.embed_max_seq_length = embed_max_seq_length

        self.umap_args = None
        self.hdbscan_args = None

    def fit(self, embeddings, optimization_trials=100, sample_size=None):

        if type(embeddings[0]) is str:
            embeddings = self.embed(embeddings)

        if sample_size is None:
            sample_size = len(embeddings)

        if sample_size < len(embeddings):
            data = random.sample(list(embeddings), sample_size)
        else:
            data = embeddings


        # Define the objective function for Optuna optimization
        def objective(trial):

            # Suggest UMAP hyperparameters
            n_neighbors = trial.suggest_int('umap_n_neighbors', 5, 50)  # Number of neighbors for UMAP
            min_dist = trial.suggest_float('umap_min_dist', 0.0, 1.0)  # Minimum distance for UMAP
            metric = trial.suggest_categorical('umap_metric', ['euclidean', 'cosine'])  # Metric for UMAP

            # Suggest HDBSCAN hyperparameters
            min_cluster_size = trial.suggest_int('hdbscan_min_cluster_size', 5, 100)  # Minimum cluster size
            hdbscan_metric = trial.suggest_categorical('hdbscan_metric', ['euclidean', 'cosine'])  # Metric for HDBSCAN
            cluster_selection_epsilon = trial.suggest_float('cluster_selection_epsilon', 0, 1.0)    #Cluster selection epsilon for hdbscan

            try:
                # Apply UMAP for dimensionality reduction
                umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
                umap_embedding = umap_model.fit_transform(data)

                # Apply HDBSCAN for clustering
                hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, 
                                        metric=hdbscan_metric,
                                        cluster_selection_epsilon=cluster_selection_epsilon)
                cluster_labels = hdbscan_model.fit_predict(umap_embedding)

                # Evaluate clustering performance using the silhouette score
                # Silhouette score requires at least 2 clusters; handle single-cluster cases
                if len(np.unique(cluster_labels)) > 1:
                    score = self.compute_score(umap_embedding, cluster_labels)
                else:
                    score = -1  # Assign a low score for poor clustering results (e.g., single cluster)
            except Exception as e:
                print(e)
                score = -1 # Assign a low score to failed runs

            return score

        # Create and optimize an Optuna study
        best = -1
        while best==-1:
            study = optuna.create_study(direction='maximize', pruner=MedianPruner())  # Maximize the silhouette score
            study.optimize(objective, n_trials=optimization_trials)
            best = study.best_value
            if best == -1:
                # if study fails, retry with 80% of sample size
                print(f'Study failed to optimize with sample size: {sample_size}')
                if sample_size > 0:
                    sample_size = int(sample_size // 1.25)
                    print(f'Re-trying with sample size: {sample_size}')
                    data = random.sample(list(embeddings), sample_size)
                else:
                    print(f'No optimal hyperparameters found.')
                    projection_args = {}
                    clustering_args = {}
                    return projection_args, clustering_args

        # Print the best parameters and corresponding score
        print("Best Parameters:", study.best_params)
        print("Best Score:", study.best_value)

        # Update projection and clustering args with the optimized parameters
        projection_args = {'n_neighbors': study.best_params['umap_n_neighbors'], 
                            'min_dist': study.best_params['umap_min_dist'], 
                            'metric': study.best_params['umap_metric']}

        clustering_args = {'min_cluster_size': study.best_params['hdbscan_min_cluster_size'], 
                            'metric': study.best_params['hdbscan_metric'],
                            'cluster_selection_epsilon': study.best_params['cluster_selection_epsilon']}
        
        self.umap_args = projection_args
        self.hdbscan_args = clustering_args
        
        return projection_args, clustering_args

    def normalize(self, value, min_val, max_val):
        """Normalize a value to range [0, 1]."""
        return (value - min_val) / (max_val - min_val)

    def compute_score(self, data, cluster_labels, weights=(1,1,1)):
        """Calculate the composite clustering score using given weights."""
        silhouette_weight, ch_weight, db_weight = weights
        
        # Silhouette Score
        silhouette = silhouette_score(data, cluster_labels)  # Range: [-1, 1]
        silhouette = (silhouette + 1) / 2  # Normalize to [0, 1]
        
        # Calinski-Harabasz Index
        ch_index = calinski_harabasz_score(data, cluster_labels)  # Positive unbounded
        # Example normalization for CH Index
        ch_min, ch_max = 0, 1000
        ch_normalized = self.normalize(ch_index, ch_min, ch_max)
        
        # Davies-Bouldin Index
        db_index = davies_bouldin_score(data, cluster_labels)  # Lower is better
        # Example normalization for DB Index
        db_min, db_max = 0, 10
        db_normalized = 1 - self.normalize(db_index, db_min, db_max)  # Invert and normalize
        
        # Composite Score
        composite_score = (
            silhouette_weight * silhouette +
            ch_weight * ch_normalized +
            db_weight * db_normalized
        )
        
        return composite_score

    def embed(self, texts):

        try:
            if torch.cuda.is_available():
                embed_device='cuda'

            embed_model = SentenceTransformer(
                    self.embed_model_name, device=self.embed_device
                )
            embed_model.max_seq_length = self.embed_max_seq_length

            # Generate embeddings for the input texts with specified parameters
            embeddings = embed_model.encode(
                texts,
                batch_size=self.embed_batch_size,      # Process texts in batches to optimize performance
                show_progress_bar=True,                # Display a progress bar for embedding generation
                convert_to_numpy=True,                 # Convert embeddings to a NumPy array format
                normalize_embeddings=True,             # Normalize embeddings to unit length
            )
        
        except:

            embed_model = SentenceTransformer(
                    self.embed_model_name, device=embed_device
                )
            embed_model.max_seq_length = self.embed_max_seq_length

            # Generate embeddings for the input texts with specified parameters
            embeddings = embed_model.encode(
                texts,
                batch_size=self.embed_batch_size,      # Process texts in batches to optimize performance
                show_progress_bar=True,                # Display a progress bar for embedding generation
                convert_to_numpy=True,                 # Convert embeddings to a NumPy array format
                normalize_embeddings=True,             # Normalize embeddings to unit length
            )

        return embeddings