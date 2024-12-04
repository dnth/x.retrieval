import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from rich.console import Console
from rich.table import Table

from .datasets_registry import DatasetRegistry
from .metrics import calculate_retrieval_metrics
from .models.bm25 import BM25sModel
from .models_registry import ModelRegistry


def list_datasets(search: str = ""):
    # Convert wildcard pattern to simple regex-like matching
    search = search.replace("*", "").lower()
    datasets_dict = DatasetRegistry.list()
    filtered_datasets = {
        name: desc for name, desc in datasets_dict.items() if search in name.lower()
    }

    # Create and print table
    table = Table(title="Available Datasets")
    table.add_column("Dataset Name", style="cyan")
    table.add_column("Description", style="magenta")

    for name, description in filtered_datasets.items():
        table.add_row(name, description or "No description available")

    console = Console()
    console.print(table)

    # return datasets


def list_models(search: str = "") -> dict:
    # Convert wildcard pattern to simple regex-like matching
    search = search.replace("*", "").lower()
    # Get filtered models
    models = {
        model_id: model_input
        for model_id, model_input in ModelRegistry.list().items()
        if search in model_id.lower()
    }

    # Create and print table
    table = Table(title="Available Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Model Input", style="magenta")

    for model_id, input_type in models.items():
        table.add_row(model_id, input_type)

    console = Console()
    console.print(table)

    # return models


def load_dataset(name: str | pd.DataFrame):
    if isinstance(name, pd.DataFrame):
        return name
    dataset_class = DatasetRegistry.get(name)
    return dataset_class.get_dataset()


def load_model(model_id: str):
    model_class = ModelRegistry.get(model_id)
    return model_class(model_id=model_id)


def run_benchmark_bm25(dataset: str, top_k: int = 10):
    logger.info("Running BM25 retrieval benchmark")
    bm25_model = BM25sModel()
    dataset = load_dataset(dataset)

    logger.info("Tokenizing corpus")
    corpus = dataset["caption"].tolist()
    bm25_model.tokenize_text(corpus)

    # Get labels for evaluation
    image_ids = dataset.image_id.tolist()
    image_ids = np.array(image_ids)
    labels = dataset.loc[(dataset.image_id.isin(image_ids))].name.to_numpy()

    logger.info("Performing retrieval")
    retrieved_ids = bm25_model.retrieve(corpus, top_k=top_k)

    logger.info("Calculating metrics")
    eval_metrics_results = calculate_retrieval_metrics(labels, retrieved_ids, top_k)

    # Create results DataFrame for visualization
    results_data = []
    for idx, retrieved in enumerate(retrieved_ids):
        query_name = dataset.iloc[idx]["name"]
        ground_truth_matches = dataset[
            (dataset["name"] == query_name)
            & (dataset["image_id"] != dataset.iloc[idx]["image_id"])
        ]

        query_row = {
            "query_id": dataset.iloc[idx]["image_id"],
            "query_path": dataset.iloc[idx]["image_path"],
            "query_caption": dataset.iloc[idx]["caption"],
            "query_name": dataset.iloc[idx]["name"],
            "retrieved_ids": [dataset.iloc[i]["image_id"] for i in retrieved],
            "retrieved_paths": [dataset.iloc[i]["image_path"] for i in retrieved],
            "retrieved_captions": [dataset.iloc[i]["caption"] for i in retrieved],
            "retrieved_names": [dataset.iloc[i]["name"] for i in retrieved],
            "is_correct": [labels[i] == labels[idx] for i in retrieved],
            "ground_truth_ids": ground_truth_matches["image_id"].tolist(),
            "ground_truth_paths": ground_truth_matches["image_path"].tolist(),
            "ground_truth_captions": ground_truth_matches["caption"].tolist(),
        }
        results_data.append(query_row)

    results_df = pd.DataFrame(results_data)

    return eval_metrics_results, results_df


def run_benchmark(
    dataset: str | pd.DataFrame,
    model_id: str,
    mode: str = "image-to-image",  # Can be "image-to-image", "text-to-text", "text-to-image", or "image-to-text"
    top_k: int = 10,
):
    """
    Run retrieval benchmark on a dataset

    Args:
        dataset_name: Name of the dataset to use or a pandas DataFrame containing the dataset
        model_id: ID of the model to use
        mode: Type of retrieval ("image-to-image", "text-to-text", "text-to-image", or "image-to-text")
        top_k: Number of top results to retrieve (will retrieve top_k + 1 to account for self-matches)
    """
    dataset = load_dataset(dataset)

    # TODO: Dataset should contain columns ['image_id', 'file_name', 'image_path', 'caption', 'name']
    model = load_model(model_id)
    model_info = ModelRegistry.get_model_info(model_id)

    logger.info(f"Evaluating embeddings with parameter top_k: {top_k}")
    image_ids = dataset.image_id.tolist()
    image_ids = np.array(image_ids)
    labels = dataset.loc[(dataset.image_id.isin(image_ids))].name.to_numpy()

    # Encode embeddings based on mode
    if mode.endswith("image"):
        logger.info(f"Encoding database images for {model_id}")
        embeddings = model.encode_image(dataset["image_path"].tolist())
    else:
        logger.info(f"Encoding database text for {model_id}")
        embeddings = model.encode_text(dataset["caption"].tolist())

    if (
        mode.startswith("text")
        and mode.endswith("image")
        or mode.startswith("image")
        and mode.endswith("text")
    ):
        if mode.startswith("image"):
            logger.info(f"Encoding query images for {model_id}")
            query_embeddings = model.encode_image(dataset["image_path"].tolist())
        else:
            logger.info(f"Encoding query text for {model_id}")
            query_embeddings = model.encode_text(dataset["caption"].tolist())
    else:
        query_embeddings = embeddings

    # Create FAISS index
    logger.info("Creating indices for vector search...")
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dimension))
    faiss.normalize_L2(embeddings)
    index.add_with_ids(embeddings, np.arange(len(embeddings)))

    # Search
    logger.info("Performing vector search...")
    faiss.normalize_L2(query_embeddings)
    _, retrieved_ids = index.search(
        query_embeddings, k=top_k + 1
    )  # +1 to account for self-matches

    # Filter self matches
    filtered_retrieved_ids = []
    for idx, row in enumerate(retrieved_ids):
        filtered_row = [x for x in row if x != idx]
        if len(filtered_row) != top_k:
            filtered_row = filtered_row[:top_k]
        filtered_retrieved_ids.append(filtered_row)
    filtered_retrieved_ids = np.array(filtered_retrieved_ids)

    # Calculate metrics
    eval_metrics_results = calculate_retrieval_metrics(
        labels, filtered_retrieved_ids, top_k
    )

    # Create results DataFrame (keeping this for visualization purposes)
    results_data = []
    for idx, retrieved in enumerate(filtered_retrieved_ids):
        # Get all ground truth matches for this query (excluding self)
        query_name = dataset.iloc[idx]["name"]
        ground_truth_matches = dataset[
            (dataset["name"] == query_name)
            & (dataset["image_id"] != dataset.iloc[idx]["image_id"])
        ]

        query_row = {
            "query_id": dataset.iloc[idx]["image_id"],
            "query_path": dataset.iloc[idx]["image_path"],
            "query_caption": dataset.iloc[idx]["caption"],
            "query_name": dataset.iloc[idx]["name"],
            "retrieved_ids": [dataset.iloc[i]["image_id"] for i in retrieved],
            "retrieved_paths": [dataset.iloc[i]["image_path"] for i in retrieved],
            "retrieved_captions": [dataset.iloc[i]["caption"] for i in retrieved],
            "retrieved_names": [dataset.iloc[i]["name"] for i in retrieved],
            "is_correct": [labels[i] == labels[idx] for i in retrieved],
            # Add ground truth information
            "ground_truth_ids": ground_truth_matches["image_id"].tolist(),
            "ground_truth_paths": ground_truth_matches["image_path"].tolist(),
            "ground_truth_captions": ground_truth_matches["caption"].tolist(),
        }
        results_data.append(query_row)

    results_df = pd.DataFrame(results_data)

    return eval_metrics_results, results_df


def visualize_retrieval(
    results_df: pd.DataFrame,
    mode: str | None = None,
    num_queries: int = 5,
    seed: int = 42,
):
    """
    Visualize retrieval results from the benchmark results DataFrame

    Args:
        results_df: DataFrame containing retrieval results from run_benchmark
        mode: Type of retrieval ("image-to-image", "text-to-text", "text-to-image", "image-to-text")
              If None, shows both image and caption for queries and results
        num_queries: Number of random queries to visualize
    """
    np.random.seed(seed)
    # Select random queries
    query_indices = np.random.choice(len(results_df), num_queries, replace=False)

    for query_idx in query_indices:
        query_row = results_df.iloc[query_idx]
        retrieved_paths = query_row["retrieved_paths"]
        retrieved_captions = query_row["retrieved_captions"]
        retrieved_ids = query_row["retrieved_ids"]

        # Always filter out self-matches
        mask = [rid != query_row["query_id"] for rid in retrieved_ids]
        retrieved_paths = [p for p, m in zip(retrieved_paths, mask) if m]
        retrieved_captions = [c for c, m in zip(retrieved_captions, mask) if m]
        retrieved_ids = [i for i, m in zip(retrieved_ids, mask) if m]

        # Limit to 10 results (2 rows of 5)
        max_results = 10
        retrieved_paths = retrieved_paths[:max_results]
        retrieved_captions = retrieved_captions[:max_results]
        retrieved_ids = retrieved_ids[:max_results]
        top_k = len(retrieved_paths)

        plt.figure(figsize=(20, 12))

        plt.subplot(3, 1, 1)
        if mode is None or mode.startswith("image"):
            query_img = Image.open(query_row["query_path"])
            plt.imshow(query_img)
            if mode is None:
                plt.title(
                    f'Query ID: {query_row["query_id"]}\n{query_row["query_caption"][:50]}...',
                    fontsize=10,
                )
            else:
                plt.title(f'Query ID: {query_row["query_id"]}', fontsize=10)
        else:  # text-only mode
            plt.text(
                0.5,
                0.5,
                query_row["query_caption"],
                horizontalalignment="center",
                verticalalignment="center",
                wrap=True,
                fontsize=12,
            )
            plt.title(f'Query ID: {query_row["query_id"]}', fontsize=10)
        plt.axis("off")

        # Retrieved results visualization in 2 rows
        for i in range(top_k):
            row = 1 if i < 5 else 2  # First 5 in row 1, next 5 in row 2
            col = i % 5  # Column position within row
            plt.subplot(3, 5, 5 * row + col + 1)  # Adjusted subplot positioning
            if mode is None or mode.endswith("image"):
                retrieved_img = Image.open(retrieved_paths[i])
                plt.imshow(retrieved_img)
                if mode is None:
                    plt.title(
                        f"Match {i+1} (ID: {retrieved_ids[i]})\n{retrieved_captions[i][:50]}...",
                        fontsize=8,
                    )
                else:
                    plt.title(f"Match {i+1} (ID: {retrieved_ids[i]})", fontsize=8)
            else:  # text-only mode
                plt.text(
                    0.5,
                    0.5,
                    retrieved_captions[i],
                    horizontalalignment="center",
                    verticalalignment="center",
                    wrap=True,
                    fontsize=8,
                )
                plt.title(f"Match {i+1} (ID: {retrieved_ids[i]})", fontsize=8)
            plt.axis("off")

        plt.tight_layout(h_pad=2, w_pad=1)
        plt.show()


def visualize_ground_truth(
    dataset: str,
    mode: str | None = None,
    num_queries: int = 5,
    seed: int = 42,
):
    """
    Visualize ground truth matches from the dataset

    Args:
        dataset: Dataset name or DataFrame containing the dataset
        mode: Type of visualization ("image-to-image", "text-to-text", "text-to-image", "image-to-text")
              If None, shows both image and caption for queries and results
        num_queries: Number of random queries to visualize
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    dataset = load_dataset(dataset)

    # Select random queries
    query_indices = np.random.choice(len(dataset), num_queries, replace=False)

    for query_idx in query_indices:
        query_row = dataset.iloc[query_idx]
        query_name = query_row["name"]

        # Get all matches (excluding self)
        matches = dataset[
            (dataset["name"] == query_name)
            & (dataset["image_id"] != query_row["image_id"])
        ]

        # Limit to 10 results (2 rows of 5)
        max_results = 10
        matches = matches.head(max_results)
        top_k = len(matches)

        plt.figure(figsize=(20, 12))

        # Query visualization
        plt.subplot(3, 1, 1)
        if mode is None or mode.startswith("image"):
            query_img = Image.open(query_row["image_path"])
            plt.imshow(query_img)
            if mode is None:
                plt.title(
                    f'Query ID: {query_row["image_id"]}\n{query_row["caption"][:50]}...',
                    fontsize=10,
                )
            else:
                plt.title(f'Query ID: {query_row["image_id"]}', fontsize=10)
        else:  # text-only mode
            plt.text(
                0.5,
                0.5,
                query_row["caption"],
                horizontalalignment="center",
                verticalalignment="center",
                wrap=True,
                fontsize=12,
            )
            plt.title(f'Query ID: {query_row["image_id"]}', fontsize=10)
        plt.axis("off")

        # Ground truth matches visualization in 2 rows
        for i, (_, match) in enumerate(matches.iterrows()):
            row = 1 if i < 5 else 2  # First 5 in row 1, next 5 in row 2
            col = i % 5  # Column position within row
            plt.subplot(3, 5, 5 * row + col + 1)  # Adjusted subplot positioning

            if mode is None or mode.endswith("image"):
                match_img = Image.open(match["image_path"])
                plt.imshow(match_img)
                if mode is None:
                    plt.title(
                        f"Match {i+1} (ID: {match['image_id']})\n{match['caption'][:50]}...",
                        fontsize=8,
                    )
                else:
                    plt.title(f"Match {i+1} (ID: {match['image_id']})", fontsize=8)
            else:  # text-only mode
                plt.text(
                    0.5,
                    0.5,
                    match["caption"],
                    horizontalalignment="center",
                    verticalalignment="center",
                    wrap=True,
                    fontsize=8,
                )
                plt.title(f"Match {i+1} (ID: {match['image_id']})", fontsize=8)
            plt.axis("off")

        plt.tight_layout(h_pad=2, w_pad=1)
        plt.show()
