import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
from loguru import logger
from PIL import Image
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from .datasets_registry import DatasetRegistry
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
        top_k: Number of top results to retrieve
    """
    dataset = load_dataset(dataset)
    # TODO: Dataset should contain columns ['image_id', 'file_name', 'image_path', 'caption', 'name']
    model = load_model(model_id)
    model_info = ModelRegistry.get_model_info(model_id)

    image_ids = dataset.image_id.tolist()
    image_ids = np.array(image_ids)
    labels = dataset.loc[(dataset.image_id.isin(image_ids))].name.to_numpy()

    # Encode database items (what we're searching through)
    if mode.endswith("image"):  # text-to-image or image-to-image
        logger.info(f"Encoding database images for {model_id}")
        db_embeddings = model.encode_image(dataset["image_path"].tolist())
    else:  # text-to-text or image-to-text
        logger.info(f"Encoding database text for {model_id}")
        db_embeddings = model.encode_text(dataset["caption"].tolist())

    # Encode queries
    if mode.startswith("image"):  # image-to-image or image-to-text
        logger.info(f"Encoding query images for {model_id}")
        query_embeddings = model.encode_image(dataset["image_path"].tolist())
    else:  # text-to-text or text-to-image
        logger.info(f"Encoding query text for {model_id}")
        query_embeddings = model.encode_text(dataset["caption"].tolist())

    # Create FAISS index
    index = faiss.IndexIDMap(faiss.IndexFlatIP(db_embeddings.shape[1]))
    faiss.normalize_L2(db_embeddings)
    index.add_with_ids(db_embeddings, np.arange(len(db_embeddings)))

    # Search
    faiss.normalize_L2(query_embeddings)
    _, retrieved_ids = index.search(query_embeddings, k=top_k)

    # Remove self matches for same-modality retrieval
    if mode in ["image-to-image", "text-to-text"]:
        filtered_retrieved_ids = []
        for idx, row in enumerate(tqdm(retrieved_ids)):
            filtered_row = [x for x in row if x != idx]
            if len(filtered_row) != top_k - 1:
                filtered_row = filtered_row[: top_k - 1]
            filtered_retrieved_ids.append(filtered_row)
        retrieved_ids = np.array(filtered_retrieved_ids)

    # Create results DataFrame
    results_data = []
    for idx, retrieved in enumerate(retrieved_ids):
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
        }
        results_data.append(query_row)

    results_df = pd.DataFrame(results_data)

    # Calculate metrics
    matches = np.expand_dims(labels, axis=1) == labels[retrieved_ids]
    matches = torch.tensor(np.array(matches), dtype=torch.float16)
    targets = torch.ones(matches.shape)
    indexes = (
        torch.arange(matches.shape[0]).view(-1, 1)
        * torch.ones(1, matches.shape[1]).long()
    )

    metrics = [
        torchmetrics.retrieval.RetrievalMRR(top_k=top_k),
        torchmetrics.retrieval.RetrievalNormalizedDCG(top_k=top_k),
        torchmetrics.retrieval.RetrievalPrecision(top_k=top_k),
        torchmetrics.retrieval.RetrievalRecall(top_k=top_k),
        torchmetrics.retrieval.RetrievalHitRate(top_k=top_k),
        torchmetrics.retrieval.RetrievalMAP(top_k=top_k),
    ]
    eval_metrics_results = {}

    for metr in metrics:
        score = round(metr(targets, matches, indexes).item(), 4)
        metr_name = metr.__class__.__name__.replace("Retrieval", "")
        eval_metrics_results[metr_name] = score

    return eval_metrics_results, results_df


def visualize_retrieval(
    results_df: pd.DataFrame,
    mode: str | None = None,  # Changed default to None
    num_queries: int = 5,
):
    """
    Visualize retrieval results from the benchmark results DataFrame

    Args:
        results_df: DataFrame containing retrieval results from run_benchmark
        mode: Type of retrieval ("image-to-image", "text-to-text", "text-to-image", "image-to-text")
              If None, shows both image and caption for queries and results
        num_queries: Number of random queries to visualize
    """
    # Select random queries
    query_indices = np.random.choice(len(results_df), num_queries, replace=False)

    for query_idx in query_indices:
        query_row = results_df.iloc[query_idx]
        retrieved_paths = query_row["retrieved_paths"]
        retrieved_captions = query_row["retrieved_captions"]
        top_k = len(retrieved_paths)

        plt.figure(figsize=(20, 8))

        # Query visualization
        plt.subplot(2, 1, 1)
        if mode is None or mode.startswith("image"):
            query_img = Image.open(query_row["query_path"])
            plt.imshow(query_img)
            # Show caption below image if mode is None
            if mode is None:
                plt.title(f'Query\n{query_row["query_caption"][:100]}...', fontsize=10)
            else:
                plt.title("Query Image", fontsize=10)
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
            plt.title("Query Text", fontsize=10)
        plt.axis("off")

        # Retrieved results visualization
        for i in range(top_k):
            plt.subplot(2, top_k, top_k + i + 1)
            if mode is None or mode.endswith("image"):
                retrieved_img = Image.open(retrieved_paths[i])
                plt.imshow(retrieved_img)
                # Show caption below image if mode is None
                if mode is None:
                    plt.title(
                        f"Match {i+1}\n{retrieved_captions[i][:50]}...", fontsize=8
                    )
                else:
                    plt.title(f"Match {i+1}", fontsize=8)
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
                plt.title(f"Match {i+1}", fontsize=8)
            plt.axis("off")

        plt.tight_layout(h_pad=2, w_pad=1)
        plt.show()
