import json
import logging
import os
import random
import textwrap
from collections import Counter, defaultdict
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import plotly.express as px

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from umap import UMAP
from sklearn.decomposition import TruncatedSVD, PCA

from sklearn.cluster import DBSCAN, OPTICS, KMeans, HDBSCAN

from sklearn.metrics import silhouette_score
import optuna

logging.basicConfig(level=logging.INFO)


DEFAULT_INSTRUCTION = (
    instruction
) = "Use three words total (comma separated)\
to describe general topics in above texts. Under no circumstances use enumeration. \
Example format: Tree, Cat, Fireman"

DEFAULT_TEMPLATE = "<s>[INST]{examples}\n\n{instruction}[/INST]"


class ClusterClassifier:
    def __init__(
        self,
        batch_size = 1,
        sample_size = 100_000,
        embed_model_name="all-MiniLM-L6-v2",
        embed_device="cpu",
        embed_batch_size=64,
        embed_max_seq_length=512,
        embed_agg_strategy=None,
        optimization_trials=20,
        projection_algorithm='umap',
        projection_args = {},
        clustering_algorithm='hdbscan',
        clustering_args = {},
        summary_create=True,
        summary_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        topic_mode="multiple_topics",
        summary_n_examples=10,
        summary_chunk_size=420,
        summary_model_token=True,
        summary_template=None,
        summary_instruction=None,
    ):
        """
        Initializes the object with parameters for embedding generation, dimensionality reduction,
        clustering, and summarization of text data.

        Args:
            batch_size (int): Number of samples to process in each batch (default is 1).
            embed_model_name (str): Name of the pre-trained embedding model to use (default is "all-MiniLM-L6-v2").
            embed_device (str): The device to use for embedding generation. Options are 'cpu' or 'cuda' (default is 'cpu').
            embed_batch_size (int): Number of samples per batch during embedding generation (default is 64).
            embed_max_seq_length (int): Maximum sequence length for the embedding model (default is 512).
            embed_agg_strategy (str, optional): Aggregation strategy for embeddings (e.g., 'mean', 'sum', or None).
            optimization_trials (int): Number of trials to run during optimization
            projection_algorithm (str): Algorithm for dimensionality reduction. Options are 'pca', 'tsvd', or 'umap' (default is 'umap').
            projection_args (dict): Additional arguments for the projection algorithm (default is an empty dictionary).
            clustering_algorithm (str): Clustering algorithm to use. Options are 'dbscan', 'hdbscan', 'optics', 'kmeans' (default is 'dbscan').
            clustering_args (dict): Additional arguments for the clustering algorithm (default is an empty dictionary).
            summary_create (bool): Whether to create summaries for each cluster (default is True).
            summary_model (str): The model to use for summary creation (default is "mistralai/Mixtral-8x7B-Instruct-v0.1").
            topic_mode (str): Mode for topic extraction in summaries. Options are 'multiple_topics' or 'single_topic' (default is 'multiple_topics').
            summary_n_examples (int): Number of examples per cluster to send to the summary model (default is 10).
            summary_chunk_size (int): Maximum number of tokens per chunk in summary creation (default is 420).
            summary_model_token (bool): Whether to use a token to authenticate with the summary model (default is True).
            summary_template (str, optional): The template used for formatting the summary request (default is `DEFAULT_TEMPLATE`).
            summary_instruction (str, optional): The instruction for the summary generation (default is `DEFAULT_INSTRUCTION`).

        Attributes:
            embeddings (numpy.ndarray): The embeddings for the input texts.
            faiss_index (faiss.Index): The FAISS index for fast retrieval.
            cluster_labels (numpy.ndarray): The cluster labels for each document.
            texts (list): The input texts.
            projections (numpy.ndarray): The 2D or 3D projections for visualization.
            mapper (object): The mapper for dimensionality reduction (e.g., UMAP, PCA).
            id2label (dict): Mapping from document ID to cluster label.
            label2docs (dict): Mapping from cluster label to list of document indices.
            embed_model (SentenceTransformer): The SentenceTransformer model used for embedding generation.

        Raises:
            ValueError: If an invalid projection_algorithm or clustering_algorithm is provided.

        Example:
            # Example usage
            my_clusterer = MyClusteringClass(
                batch_size=32,
                embed_model_name="all-MiniLM-L6-v2",
                clustering_algorithm="kmeans",
                clustering_args={'n_clusters': 5},
            )
        """
        
        self.batch_size = batch_size
        self.sample_size = sample_size
        
        # Embedding model parameters
        self.embed_model_name = embed_model_name
        self.embed_device = embed_device
        self.embed_batch_size = embed_batch_size
        self.embed_max_seq_length = embed_max_seq_length
        self.embed_agg_strategy = embed_agg_strategy

        # Optimization parameters
        self.optimization_trials = optimization_trials

        # Projection algorithm parameters (e.g., UMAP, PCA)
        self.projection_algorithm = projection_algorithm
        if self.projection_algorithm not in ['pca', 'tsvd', 'umap']:
            raise ValueError("projection_algorithm must be one of ['pca', 'tsvd', 'umap']")
        self.projection_args = projection_args

        # Clustering algorithm parameters (e.g., DBSCAN, KMeans)
        self.clustering_algorithm = clustering_algorithm
        if self.clustering_algorithm not in ['dbscan', 'hdbscan', 'optics', 'kmeans']:
            raise ValueError("clustering_algorithm must be one of ['dbscan', 'hdbscan', 'optics', 'kmeans']")
        self.clustering_args = clustering_args

        # Summary creation parameters
        self.summary_create = summary_create
        self.summary_model = summary_model
        self.topic_mode = topic_mode
        self.summary_n_examples = summary_n_examples
        self.summary_chunk_size = summary_chunk_size
        self.summary_model_token = summary_model_token

        # Template and instructions for summary generation
        if summary_template is None:
            self.summary_template = DEFAULT_TEMPLATE
        else:
            self.summary_template = summary_template

        if summary_instruction is None:
            self.summary_instruction = DEFAULT_INSTRUCTION
        else:
            self.summary_instruction = summary_instruction

        # Initialize attributes for embeddings, projections, and clustering
        self.embeddings = None
        self.faiss_index = None
        self.cluster_labels = None
        self.texts = None
        self.projections = None
        self.mapper = None
        self.id2label = None
        self.label2docs = None

        # Initialize the embedding model
        self.embed_model = SentenceTransformer(
            self.embed_model_name, device=self.embed_device
        )
        self.embed_model.max_seq_length = self.embed_max_seq_length

    def fit(self, 
            texts=None,
            batch_size=None, 
            projection_algorithm=None,
            projection_args=None,
            clustering_algorithm=None,
            clustering_args=None
            ):
        """
        This method performs the complete process of fitting the model, including embedding the texts, projecting the embeddings into a lower-dimensional space,
        clustering the projections, and optionally summarizing the clusters.

        Args:
            texts (list): List of input texts to process. If not provided, the existing `self.texts` is used.
            batch_size (int, optional): Number of texts to process in a batch. If provided, it overrides the default `self.batch_size`.
            projection_algorithm (str, optional): The dimensionality reduction technique to use. Options include 'pca', 'tsvd', or 'umap'. Defaults to `self.projection_algorithm`.
            projection_args (dict, optional): Additional parameters for the projection algorithm (e.g., UMAP settings).
            clustering_algorithm (str, optional): Clustering algorithm to apply. Options include 'dbscan', 'kmeans', etc. Defaults to `self.clustering_algorithm`.
            clustering_args (dict, optional): Additional parameters for the clustering algorithm (e.g., DBSCAN settings).

        Returns:
            tuple: A tuple containing:
                - embeddings (numpy.ndarray): The embeddings for the input texts.
                - cluster_labels (numpy.ndarray): The cluster labels assigned to each document.
                - cluster_summaries (dict, optional): The summaries of each cluster, if `self.summary_create` is True.
        
        Raises:
            ValueError: If the provided `batch_size` or `projection_algorithm` is invalid.
        """
        
        # If batch size has changed, reset embeddings and projections
        if (batch_size is not None) and (batch_size != self.batch_size):
            self.embeddings = None
            self.projections = None

        # Update internal settings with new or default parameters
        self.batch_size = batch_size or self.batch_size
        self.texts = texts or self.texts
        self.projection_algorithm = projection_algorithm or self.projection_algorithm
        self.projection_args = projection_args or self.projection_args
        self.clustering_algorithm = clustering_algorithm or self.clustering_algorithm
        self.clustering_args = clustering_args or self.clustering_args

        # Preprocess the texts if batch size > 1
        if self.batch_size > 1:
            logging.info("Batching texts...")
            self.texts = self.batch_and_join(self.texts, self.batch_size)

        # Embedding generation: either from scratch or using precomputed embeddings
        if self.embeddings is None:
            logging.info("Embedding texts...")
            self.embeddings = self.embed(self.texts)
        else:
            logging.info("Using precomputed embeddings...")

        # FAISS index construction
        logging.info("Building FAISS index...")
        self.faiss_index = self.faiss_index or self.build_faiss_index(self.embeddings)

        # Projection: Apply dimensionality reduction (e.g., UMAP)
        if self.projections is None:
            logging.info(f"Projecting with {self.projection_algorithm}...")
            self.projections, self.mapper = self.project(self.embeddings, self.projection_algorithm, self.projection_args)
        else:
            logging.info("Using precomputed projections...")

        # Clustering: Apply clustering to the projections
        logging.info("Clustering...")
        self.cluster(self.projections, self.clustering_algorithm, self.clustering_args)

        # Summarization: Optionally create summaries for each cluster
        if self.summary_create:
            logging.info("Summarizing cluster centers...")
            self.cluster_summaries = self.summarize(self.texts, self.cluster_labels)
        else:
            self.cluster_summaries = None

    def infer(self, texts, top_k=1):

        """
        Infers the cluster labels for a given list of text inputs by finding the most common cluster label among the 
        nearest neighbors of each text in the FAISS index.

        Args:
            texts (list): List of text data to be classified into clusters.
            top_k (int): The number of nearest neighbors to consider for each input text. Defaults to 1.

        Returns:
            inferred_labels (list): List of predicted cluster labels for each input text.
            embeddings (numpy array): The computed embeddings for each input text.
        """

        embeddings = self.embed(texts)

        dist, neighbours = self.faiss_index.search(embeddings, top_k)
        inferred_labels = []
        for i in tqdm(range(embeddings.shape[0])):
            labels = [self.cluster_labels[doc] for doc in neighbours[i]]
            inferred_labels.append(Counter(labels).most_common(1)[0][0])

        return inferred_labels, embeddings

    def batch_and_join(self, texts, n):

        """
        Groups a list of text strings into batches of 'n' items, with each batch containing 'n' strings joined by newline characters.

        Args:
            texts (list): List of text strings to be grouped.
            n (int): The number of strings to include in each batch.

        Returns:
            batched_texts (list): List of strings, where each string is a batch of 'n' joined by newline characters.
        """

        # Create batches of 'n' strings joined with new lines
        batched_texts = ["\n".join(texts[i:i + n]) for i in range(0, len(texts), n)]
        return batched_texts

    def embed(self, texts):
        """
        Generates embeddings for a list of text strings using the specified embedding model.

        Args:
            texts (list): List of text strings to embed.

        Returns:
            embeddings (numpy.ndarray): Array of embeddings generated for each text string.
        """

        # Generate embeddings for the input texts with specified parameters
        embeddings = self.embed_model.encode(
            texts,
            batch_size=self.embed_batch_size,      # Process texts in batches to optimize performance
            show_progress_bar=True,                # Display a progress bar for embedding generation
            convert_to_numpy=True,                 # Convert embeddings to a NumPy array format
            normalize_embeddings=True,             # Normalize embeddings to unit length
        )

        return embeddings

    def optimize(self, texts=None, optimization_trials=None, sample_size=None):
        """
        Optimizes hyperparameters for dimensionality reduction (UMAP) and clustering (HDBSCAN) using Optuna.
        This method maximizes the silhouette score, which evaluates clustering quality.
        
        Args:
            texts (list, optional): A list of input texts to embed and optimize. If provided and different
                                    from `self.texts`, it replaces the current `self.texts`, and embeddings 
                                    are recalculated. Defaults to None (uses `self.texts`).
            optimization_trials (int, optional): The number of optimization trials to perform. If not provided,
                                                the value of `self.optimization_trials` is used. Defaults to None.
        
        Returns:
            None
        """
        # Use provided optimization trials, or default to the instance's optimization_trials value
        self.optimization_trials = optimization_trials or self.optimization_trials
        self.sample_size = sample_size or self.sample_size

        # If new texts are provided and differ from the current texts, update and reset embeddings
        if texts is not None and texts is not self.texts:
            self.texts = texts
            self.embeddings = None  # Reset embeddings since the input texts have changed

        # If embeddings are not already computed, generate them
        if self.embeddings is None:
            self.embeddings = self.embed(self.texts)

        if len(self.embeddings) > self.sample_size:
            data = random.sample(list(self.embeddings), self.sample_size)
        else:
            data = self.embeddings

        # Define the objective function for Optuna optimization
        def objective(trial):
            # Suggest UMAP hyperparameters
            n_neighbors = trial.suggest_int('umap_n_neighbors', 5, 50)  # Number of neighbors for UMAP
            min_dist = trial.suggest_float('umap_min_dist', 0.0, 1.0)  # Minimum distance for UMAP
            metric = trial.suggest_categorical('umap_metric', ['euclidean', 'cosine'])  # Metric for UMAP

            # Suggest HDBSCAN hyperparameters
            min_cluster_size = trial.suggest_int('hdbscan_min_cluster_size', 5, 100)  # Minimum cluster size
            min_samples = trial.suggest_int('hdbscan_min_samples', 1, 10)  # Minimum samples for a core point
            hdbscan_metric = trial.suggest_categorical('hdbscan_metric', ['euclidean', 'cosine'])  # Metric for HDBSCAN

            # Apply UMAP for dimensionality reduction
            umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
            umap_embedding = umap_model.fit_transform(data)

            # Apply HDBSCAN for clustering
            hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, 
                                    min_samples=min_samples, 
                                    metric=hdbscan_metric)
            cluster_labels = hdbscan_model.fit_predict(umap_embedding)

            # Evaluate clustering performance using the silhouette score
            # Silhouette score requires at least 2 clusters; handle single-cluster cases
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(umap_embedding, cluster_labels)
            else:
                score = -1  # Assign a low score for poor clustering results (e.g., single cluster)

            return score

        # Create and optimize an Optuna study
        while True:
            try:
                study = optuna.create_study(direction='maximize')  # Maximize the silhouette score
                study.optimize(objective, n_trials=self.optimization_trials)
                break
            except:
                # if study fails, retry with 80% of sample size
                print(f'Study failed with sample size: {self.sample_size}')
                self.sample_size = self.sample_size // (self.sample_size * 1.25)
                print(f'Re-trying with sample size: {self.sample_size}')

        # Print the best parameters and corresponding score
        print("Best Parameters:", study.best_params)
        print("Best Score:", study.best_value)

        # Update projection and clustering algorithms with the optimized parameters
        self.projection_algorithm = 'umap'
        self.projection_args = {'n_neighbors': study.best_params['umap_n_neighbors'], 
                                'min_dist': study.best_params['umap_min_dist'], 
                                'metric': study.best_params['umap_metric']}
        self.clustering_algorithm = 'hdbscan'
        self.clustering_args = {'min_cluster_size': study.best_params['hdbscan_min_cluster_size'], 
                                'min_samples': study.best_params['hdbscan_min_samples'], 
                                'metric': study.best_params['hdbscan_metric']}

    def optimize_fit(self, texts=None, optimization_trials=None, sample_size=None):
        """
        Combines optimization and fitting in a single method. 
        First, it optimizes hyperparameters for dimensionality reduction and clustering using Optuna.
        Then, it fits the model with the optimized parameters on the provided or existing texts.

        Args:
            texts (list, optional): A list of input texts to process. If provided, it overrides the 
                                    current `self.texts`. Defaults to None (uses `self.texts`).
            optimization_trials (int, optional): The number of optimization trials for hyperparameter 
                                                tuning. Defaults to None (uses `self.optimization_trials`).

        Returns:
            None
        """
        # Step 1: Perform optimization to find the best hyperparameters
        self.optimize(texts, optimization_trials, sample_size)

        # Step 2: Fit the model using the optimized parameters
        self.fit(texts)

    def project(self, embeddings, projection_algorithm, projection_args, sample_size=None):
        """
        Projects embeddings into a lower-dimensional space using a specified dimensionality reduction algorithm.

        Args:
            embeddings (numpy.ndarray): Array of embeddings to project.
            projection_algorithm (str): The algorithm to use for projection. Supported values are 'pca' (Principal Component Analysis),
                                        'umap' (Uniform Manifold Approximation and Projection), and 'tsvd' (Truncated SVD).
            projection_args (dict): Additional arguments for the projection algorithm, such as the number of components.

        Returns:
            tuple: A tuple containing:
                - projections (numpy.ndarray): The lower-dimensional representations of the embeddings.
                - mapper (object): The trained projection model instance.

        Notes:
            - 'pca': Reduces dimensionality using Principal Component Analysis.
            - 'umap': Projects data using Uniform Manifold Approximation and Projection.
            - 'tsvd': Uses Truncated SVD for dimensionality reduction, typically suitable for sparse matrices.
        """

        # Set or update the projection algorithm to be used
        self.projection_algorithm = projection_algorithm or self.projection_algorithm
        self.sample_size = sample_size or self.sample_size

        if projection_algorithm == 'pca':
            mapper = PCA(**projection_args)            # Initialize PCA with specified arguments
            projections = mapper.fit_transform(embeddings)  # Fit and transform embeddings to lower dimensions
            return projections, mapper

        elif projection_algorithm == 'umap':
            if len(embeddings) <= self.sample_size:
                mapper = UMAP(**projection_args).fit(embeddings)  # Fit UMAP model to embeddings
                return mapper.embedding_, mapper                  # Return UMAP projections and the model instance
            else:
                # Fit UMAP model on a random sample
                mapper = UMAP(**projection_args).fit(random.sample(list(embeddings), self.sample_size))

                num_embeddings = len(embeddings)
                embedding_dim = mapper.embedding_.shape[1]  # Get the dimensionality of the projections
                num_batches = (num_embeddings - self.sample_size) // self.sample_size  # Calculate number of batches

                # Initialize an empty NumPy array for projections
                projections = np.zeros((num_embeddings, embedding_dim))

                start = 0
                end = self.sample_size

                # Use tqdm to show progress over batches
                for batch_idx in tqdm(range(num_batches), desc="Projecting embeddings"):
                    batch_projection = mapper.transform(embeddings[start:end])
                    projections[start:end] = batch_projection
                    start = end
                    end += self.sample_size

                # Handle remaining embeddings if any
                if start < num_embeddings:
                    projections[start:] = mapper.transform(embeddings[start:])

                return projections, mapper

        elif projection_algorithm == 'tsvd':
            mapper = TruncatedSVD(**projection_args)    # Initialize Truncated SVD with specified arguments
            projections = mapper.fit_transform(embeddings)  # Fit and transform embeddings with TSVD
            return projections, mapper

    def cluster(self, embeddings, clustering_algorithm, clustering_args):

        """
        Applies a specified clustering algorithm to the given embeddings and stores the resulting cluster labels.

        Args:
            embeddings (np.ndarray): Array of embeddings to cluster, with shape (num_samples, embedding_dim).
            clustering_algorithm (str): The name of the clustering algorithm to use. Options include:
                                        - 'dbscan': Density-based spatial clustering.
                                        - 'hdbscan': Hierarchical density-based spatial clustering.
                                        - 'kmeans': K-Means clustering.
                                        - 'optics': Ordering points to identify the clustering structure.
            clustering_args (dict): Dictionary of arguments specific to the chosen clustering algorithm.

        Returns:
            None

        Notes:
            - Depending on the selected algorithm, different clustering behaviors and outputs may result.
            - The resulting cluster labels are stored for further analysis or downstream tasks.
        """

        # Apply DBSCAN clustering if specified
        if clustering_algorithm == 'dbscan':
            print(f"Using DBSCAN params={clustering_args}")
            clustering = DBSCAN(**clustering_args).fit(embeddings)

        # Apply HDBSCAN clustering if specified
        if clustering_algorithm == 'hdbscan':
            print(f"Using HDBSCAN params={clustering_args}")
            clustering = HDBSCAN(**clustering_args).fit(embeddings)

        # Apply K-Means clustering if specified
        elif clustering_algorithm == 'kmeans':
            print(f"Using K-Means params={clustering_args}")
            clustering = KMeans(**clustering_args).fit(embeddings)

        # Apply OPTICS clustering if specified
        elif clustering_algorithm == 'optics':
            print(f"Using OPTICS params={clustering_args}")
            clustering = OPTICS(**clustering_args).fit(embeddings)

        # Store the resulting cluster labels
        self.store_cluster_info(clustering.labels_)

    def store_cluster_info(self, cluster_labels):
        """
        Stores information about clustering results, including cluster labels, document-to-cluster mappings,
        and calculated cluster centers in a 2D projection space.

        Args:
            cluster_labels (list): List of cluster labels assigned to each document.

        Returns:
            None

        Attributes Updated:
            self.cluster_labels (list): Stores the cluster labels for each document.
            self.id2cluster (dict): Maps document indices to their assigned cluster labels.
            self.label2docs (defaultdict): Maps each cluster label to a list of document indices that belong to that cluster.
            self.cluster_centers (dict): Stores the calculated center coordinates (x, y) of each cluster in the 2D projection space.

        Notes:
            - `self.cluster_centers` is computed based on the mean coordinates of documents belonging to each cluster label.
            - Assumes `self.projections` contains 2D coordinates for each document for use in calculating cluster centers.
        """

        # Store the cluster labels provided as input
        self.cluster_labels = cluster_labels

        # Create a mapping from document index to cluster label
        self.id2cluster = {
            index: label for index, label in enumerate(self.cluster_labels)
        }

        # Create a mapping from each cluster label to the list of document indices within that cluster
        self.label2docs = defaultdict(list)
        for i, label in enumerate(self.cluster_labels):
            self.label2docs[label].append(i)

        # Calculate the center coordinates for each cluster in the 2D projection space
        self.cluster_centers = {}
        for label in self.label2docs.keys():
            x = np.mean([self.projections[doc, 0] for doc in self.label2docs[label]])
            y = np.mean([self.projections[doc, 1] for doc in self.label2docs[label]])
            self.cluster_centers[label] = (x, y)

    def build_faiss_index(self, embeddings):

        """
        Constructs a FAISS index for efficient similarity search using the provided embeddings.

        Args:
            embeddings (np.ndarray): Array of embeddings with shape (num_documents, embedding_dim),
                                    where each row represents a document's embedding vector.

        Returns:
            faiss.IndexFlatL2: A FAISS index using L2 distance for similarity search, containing the added embeddings.

        Notes:
            - `faiss.IndexFlatL2` is a simple FAISS index type that calculates L2 (Euclidean) distances.
            - Assumes embeddings are in a 2D numpy array format with consistent dimensionality across rows.
        """

        # Create a FAISS index for L2 (Euclidean) distance with embedding dimensionality
        index = faiss.IndexFlatL2(embeddings.shape[1])

        # Add the embeddings to the index for future similarity searches
        index.add(embeddings)

        return index

    def summarize(self, texts, labels):

        """
        Generates summaries for each unique cluster in the provided labels using an inference client.

        Args:
            texts (list of str): List of text data to be summarized, where each text corresponds to a document.
            labels (list of int): List of cluster labels corresponding to each text in `texts`.
                                The label "-1" is excluded from summarization as it typically represents noise.

        Returns:
            dict: A dictionary where keys are cluster labels and values are the generated summaries.

        Notes:
            - The function selects a random sample of texts for each cluster and generates a summary based on
            these examples.
            - The summary model is accessed via `InferenceClient`, and text generation requests are structured
            with a template and an instruction.
            - The response is processed using `_postprocess_response` for final formatting.
        """

        unique_labels = len(set(labels)) - 1  # Number of unique clusters excluding "-1" (noise cluster)
        client = InferenceClient(self.summary_model, token=self.summary_model_token)
        cluster_summaries = {-1: "None"}  # Initialize with label "-1" as "None"

        # Generate summaries for each valid cluster label
        for label in range(unique_labels):
            # Select random text samples from the current cluster
            ids = np.random.choice(self.label2docs[label], self.summary_n_examples)
            examples = "\n\n".join(
                [
                    f"Example {i+1}:\n{texts[_id][:self.summary_chunk_size]}"
                    for i, _id in enumerate(ids)
                ]
            )

            # Format the request for the summary model
            request = self.summary_template.format(
                examples=examples, instruction=self.summary_instruction
            )
            
            # Obtain the summary from the inference client
            response = client.text_generation(request)
            
            # Display the first request as a sample for debugging
            if label == 0:
                print(f"Request:\n{request}")

            # Process and store the response summary
            cluster_summaries[label] = self._postprocess_response(response)

        print(f"Number of clusters is {len(cluster_summaries)}")
        return cluster_summaries

    def _postprocess_response(self, response):
        """
        Processes the raw response from the summary model to extract and format the relevant topic information
        based on the specified topic mode.

        Args:
            response (str): The raw text response from the model, expected to contain topic details and possibly 
                            an educational value rating.

        Returns:
            str: A processed summary string.
                - If `topic_mode` is "multiple_topics", returns a comma-separated list of topics from the first line.
                - If `topic_mode` is "single_topic", returns the topic and educational score in a formatted string.

        Raises:
            ValueError: If `self.topic_mode` is not "single_topic" or "multiple_topics".

        Notes:
            - In "multiple_topics" mode, only topics from the first line are retained, and any parenthetical information 
            is removed.
            - In "single_topic" mode, both topic and educational score are extracted from the first line. If either is 
            not found, an error message is printed.
        """

        if self.topic_mode == "multiple_topics":
            # Extract and format the first line to get multiple topics, removing any parenthesis or empty segments
            summary = response.split("\n")[0].split(".")[0].split("(")[0]
            summary = ",".join(
                [txt for txt in summary.strip().split(",") if len(txt) > 0]
            )
            return summary

        elif self.topic_mode == "single_topic":
            # Extract topic and educational score, if available
            first_line = response.split("\n")[0]
            topic, score = None, None

            try:
                # Extract topic name
                topic = first_line.split("Topic:")[1].split("(")[0].split(",")[0].strip()
            except IndexError:
                print("No topic found")

            try:
                # Extract educational score
                score = first_line.split("Educational value rating:")[1].strip().split(".")[0].strip()
            except IndexError:
                print("No educational score found")

            # Format output as "<Topic>. Educational score: <Score>"
            full_output = f"{topic}. Educational score: {score}"
            return full_output

        else:
            # Raise an error if the topic mode is unsupported
            raise ValueError(
                f"Topic labeling mode {self.topic_mode} is not supported, use single_topic or multiple_topics instead."
            )

    def save(self, folder):

        """
        Saves various components of the model and related data to the specified folder. If the folder doesn't exist, 
        it is created. This function saves embeddings, projections, cluster labels, texts, and optional cluster summaries 
        to disk in a structured format.

        Args:
            folder (str): The path to the folder where the model data will be saved. If the folder doesn't exist, 
                        it will be created.

        Returns:
            None

        Notes:
            - The function saves the following files in the specified folder:
                - `embeddings.npy`: The model's embeddings as a NumPy binary file.
                - `faiss.index`: The FAISS index object for nearest neighbor search.
                - `projections.npy`: The projections of the data points as a NumPy binary file.
                - `cluster_labels.npy`: The cluster labels associated with the data points.
                - `texts.json`: The raw input texts associated with the embeddings.
                - `mistral_prompt.txt`: A text file containing the default instruction prompt for the model.
                - `cluster_summaries.json` (optional): Summaries of the clusters, saved if available.
            - The function uses NumPy and FAISS libraries to save arrays and indexes efficiently.
        """

        # Ensure the folder exists or create it
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save embeddings as a binary NumPy file
        with open(f"{folder}/embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)

        # Save FAISS index to disk
        faiss.write_index(self.faiss_index, f"{folder}/faiss.index")

        # Save projections as a binary NumPy file
        with open(f"{folder}/projections.npy", "wb") as f:
            np.save(f, self.projections)

        # Save cluster labels as a binary NumPy file
        with open(f"{folder}/cluster_labels.npy", "wb") as f:
            np.save(f, self.cluster_labels)

        # Save the raw texts as a JSON file
        with open(f"{folder}/texts.json", "w") as f:
            json.dump(self.texts, f)

        # Save the default instruction prompt in a text file
        with open(f"{folder}/mistral_prompt.txt", "w") as f:
            f.write(DEFAULT_INSTRUCTION)

        # Optionally, save the cluster summaries if available
        if self.cluster_summaries is not None:
            with open(f"{folder}/cluster_summaries.json", "w") as f:
                json.dump(self.cluster_summaries, f)

    def load(self, folder):
        """
        Loads model data and related information from the specified folder. If the folder doesn't exist, an error is raised.
        This function restores embeddings, projections, cluster labels, texts, and optional cluster summaries. It also 
        infers additional information based on the loaded data.

        Args:
            folder (str): The path to the folder from which the model data will be loaded. The folder must contain the necessary files.

        Raises:
            ValueError: If the specified folder does not exist.

        Returns:
            None

        Notes:
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
        """

        # Check if the folder exists
        if not os.path.exists(folder):
            raise ValueError(f"The folder '{folder}' does not exist.")

        # Load embeddings from file
        with open(f"{folder}/embeddings.npy", "rb") as f:
            self.embeddings = np.load(f)

        # Load FAISS index
        self.faiss_index = faiss.read_index(f"{folder}/faiss.index")

        # Load projections from file
        with open(f"{folder}/projections.npy", "rb") as f:
            self.projections = np.load(f)

        # Load cluster labels from file
        with open(f"{folder}/cluster_labels.npy", "rb") as f:
            self.cluster_labels = np.load(f)

        # Load raw texts from file
        with open(f"{folder}/texts.json", "r") as f:
            self.texts = json.load(f)

        # Optionally load cluster summaries if available
        if os.path.exists(f"{folder}/cluster_summaries.json"):
            with open(f"{folder}/cluster_summaries.json", "r") as f:
                self.cluster_summaries = json.load(f)
                keys = list(self.cluster_summaries.keys())
                for key in keys:
                    self.cluster_summaries[int(key)] = self.cluster_summaries.pop(key)

        # Infer additional objects based on loaded data
        self.id2cluster = {
            index: label for index, label in enumerate(self.cluster_labels)
        }
        self.label2docs = defaultdict(list)
        for i, label in enumerate(self.cluster_labels):
            self.label2docs[label].append(i)

        # Compute cluster centers based on the projections
        self.cluster_centers = {}
        for label in self.label2docs.keys():
            x = np.mean([self.projections[doc, 0] for doc in self.label2docs[label]])
            y = np.mean([self.projections[doc, 1] for doc in self.label2docs[label]])
            self.cluster_centers[label] = (x, y)

    def show(self, interactive=False):
        """
        Visualizes the projections of the data points, optionally in 2D or 3D, with cluster labels and associated text content.
        The method displays the projections using either Matplotlib or Plotly for interactive or static plotting.

        Args:
            interactive (bool): If True, the plot is displayed using Plotly for interactivity; 
                                if False, a static plot is shown using Matplotlib. Default is False.

        Returns:
            None

        Notes:
            - If the number of components in self.projections is 3 or more, a 3D plot is created, where `X`, `Y`, and `Z` represent the projections in 3-dimensional space.
            - If the number of components in self.projections is 2, a 2D plot is created, with `X` and `Y` representing the projections in 2-dimensional space.
            - The content of each data point (up to 1024 characters) is displayed in the plot, with long text wrapped to fit within the plot's space.
            - The `labels` represent the cluster labels for each data point.
            - The function relies on the `projections` (data points' projections), `cluster_labels` (assigned clusters), and `texts` (the content for each data point).

        Visualization Methods:
            - For interactive plotting, Plotly is used, allowing zoom, hover, and exploration.
            - For static plotting, Matplotlib is used for a simpler, non-interactive visualization.
        """

        # Prepare the DataFrame based on the number of UMAP components (2D or 3D projection)
        if len(self.projections[0]) == 2:
            df = pd.DataFrame(
                data={
                    "X": self.projections[:, 0],
                    "Y": self.projections[:, 1],
                    "labels": self.cluster_labels,
                    "content_display": [
                        textwrap.fill(txt[:1024], 64) for txt in self.texts
                    ],
                }
            )
        else:
            df = pd.DataFrame(
                data={
                    "X": self.projections[:, 0],
                    "Y": self.projections[:, 1],
                    "Z": self.projections[:, 2],
                    "labels": self.cluster_labels,
                    "content_display": [
                        textwrap.fill(txt[:1024], 64) for txt in self.texts
                    ],
                }
            )

        # Choose the plotting method based on the 'interactive' flag
        if interactive:
            self._show_plotly(df)
        else:
            self._show_mpl(df)

    def _show_mpl(self, df):
        """
        Displays a static 2D scatter plot using Matplotlib, visualizing the cluster projections with text annotations
        for each cluster summary. The plot uses different colors for each cluster and optionally displays a summary 
        of each cluster at its center.

        Args:
            df (pd.DataFrame): A DataFrame containing the projection data (`X`, `Y`), cluster labels (`labels`), 
                                and the content to display (`content_display`). The DataFrame should also have the 
                                cluster center positions and a column indicating the color for each cluster.

        Returns:
            None

        Notes:
            - The plot will use Matplotlib to display the scatter plot with `X` and `Y` as the coordinates and 
            color-coding the points based on their assigned `labels`.
            - The function handles clusters by assigning a color to each label, where label `-1` is considered as outliers 
            and is assigned a specific color (`C0`).
            - If available, cluster summaries will be shown at the center of each cluster, with each cluster summary 
            being displayed as a small annotation on the plot.
            - The plot will have no axes or grid, providing a clean visualization.

        Visualization:
            - Each point is represented in the scatter plot with a size of `0.75` and an alpha value of `0.8` for transparency.
            - The cluster centers are annotated with the cluster summaries, which are displayed at the center of each cluster.

        Exception Handling:
            - If an error occurs while plotting the scatter plot with `labels`, the function attempts to plot the scatter plot 
            without the `c="labels"` argument (i.e., without color mapping based on labels).
        """
        
        # Create a Matplotlib figure and axis for the plot
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        # Assign colors to clusters (outliers are assigned the color 'C0')
        df["color"] = df["labels"].apply(lambda x: "C0" if x == -1 else f"C{(x % 9) + 1}")

        # Attempt to plot with color based on labels
        try:
            df.plot(
                kind="scatter",
                x="X",
                y="Y",
                c="labels",
                s=0.75,
                alpha=0.8,
                linewidth=0,
                color=df["color"],
                ax=ax,
                colorbar=False,
            )
        except:
            # Fallback to plotting without coloring by labels if the previous attempt fails
            df.plot(
                kind="scatter",
                x="X",
                y="Y",
                s=0.75,
                alpha=0.8,
                linewidth=0,
                color=df["color"],
                ax=ax,
                colorbar=False,
            )

        # Annotate each cluster with its summary text at the cluster center
        for label in self.cluster_summaries.keys():
            if label == -1:
                continue  # Skip the outlier cluster
            summary = self.cluster_summaries[label]
            position = self.cluster_centers[label]
            t = ax.text(
                position[0],
                position[1],
                summary,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=4,
            )
            # Set the background for the text annotation for better readability
            t.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=0, boxstyle='square,pad=0.1'))

        # Turn off the axis for a cleaner plot
        ax.set_axis_off()

    def _show_plotly(self, df):
        """
        Displays an interactive 2D or 3D scatter plot using Plotly, visualizing the cluster projections with hoverable data 
        and text annotations for each cluster summary. The plot uses different colors for each cluster and optionally 
        displays a summary of each cluster at its center.

        Args:
            df (pd.DataFrame): A DataFrame containing the projection data (`X`, `Y`, and optionally `Z`), cluster labels (`labels`),
                            and the content to display (`content_display`). The DataFrame should also have the cluster center positions
                            and a column indicating the color for each cluster.

        Returns:
            None

        Notes:
            - If the projection has two components, a 2D scatter plot will be generated.
            - If the projection has three components, a 3D scatter plot will be generated.
            - Each point in the plot represents a data point and is color-coded by its assigned cluster label.
            - Hovering over the points displays the content associated with that point.
            - If available, cluster summaries will be displayed at the center of each cluster as text annotations.
            - The plot is rendered in a dark-themed layout using Plotly.

        Visualization:
            - Points in the scatter plot are sized and styled with a fixed size (`1`) and opacity (`0.8`).
            - The hover data includes the `content_display` for each data point, which is a portion of the text data associated with it.
            - The cluster summaries are displayed at the cluster center with `text` annotations.

        Exception Handling:
            - If the projection has three components, the plot will be rendered in 3D.
            - Otherwise, the plot will be 2D, using `X` and `Y` for the axes.

        Example Usage:
            - To use this function, pass a DataFrame containing the projected data (`X`, `Y`, and optionally `Z`) and the corresponding cluster labels.
        """
        
        # Check if the projection has 3 components or less and plot accordingly
        if len(self.projections[0]) == 2:
            fig = px.scatter(
                df,
                x="X",
                y="Y",
                color="labels",
                hover_data={"content_display": True, "X": False, "Y": False},
                width=1600,
                height=800,
                color_continuous_scale="HSV",
            )
        else:
            print('showing 3d')
            fig = px.scatter_3d(
                df,
                x="X",
                y="Y",
                z="Z",
                color="labels",
                hover_data={"content_display": True, "X": False, "Y": False},
                width=1600,
                height=800,
                color_continuous_scale="HSV",
            )

        # Update hover template to display content data on hover
        fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

        # Update marker properties for better visibility
        fig.update_traces(
            marker=dict(size=1, opacity=0.8),  # Customize marker size and opacity
            selector=dict(mode="markers"),
        )

        # Update the layout to apply a dark theme
        fig.update_layout(
            template="plotly_dark",
        )

        # Add cluster summaries as annotations at the cluster centers
        for label in self.cluster_summaries.keys():
            if label == -1:
                continue  # Skip outlier clusters
            summary = self.cluster_summaries[label]
            position = self.cluster_centers[label]

            if len(self.projections[0]) == 2:
                fig.add_annotation(
                    x=position[0],
                    y=position[1],
                    text=summary,
                    showarrow=False,
                    yshift=0,
                )
            else:
                fig.add_annotation(
                    x=position[0],
                    y=position[1],
                    z=position[2],
                    text=summary,
                    showarrow=False,
                    yshift=0,
                )

        # Show the plot
        fig.show()
