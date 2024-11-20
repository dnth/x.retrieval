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
    results = {}

    for metr in metrics:
        score = round(metr(targets, matches, indexes).item(), 4)
        metr_name = metr.__class__.__name__.replace("Retrieval", "")
        results[metr_name] = score

    return results


def visualize_retrieval(
    dataset_name: str | pd.DataFrame,
    model_id: str,
    mode: str = "image-to-image",  # Can be "image-to-image", "text-to-text", "text-to-image", or "image-to-text"
    num_queries: int = 5,
    top_k: int = 5,
):
    """
    Visualize retrieval results for random queries from the dataset

    Args:
        dataset_name: Name of the dataset to use or a pandas DataFrame containing the dataset
        model_id: ID of the model to use
        mode: Type of retrieval to perform
        num_queries: Number of random queries to visualize
        top_k: Number of top results to show for each query
    """
    dataset = load_dataset(dataset_name)
    model = load_model(model_id)
    model_info = ModelRegistry.get_model_info(model_id)

    # Encode database items (what we're searching through)
    if mode.endswith("image"):  # text-to-image or image-to-image
        db_embeddings = model.encode_image(dataset["image_path"].tolist())
    else:  # text-to-text or image-to-text
        db_embeddings = model.encode_text(dataset["caption"].tolist())

    # Create FAISS index for database embeddings
    index = faiss.IndexIDMap(faiss.IndexFlatIP(db_embeddings.shape[1]))
    faiss.normalize_L2(db_embeddings)
    index.add_with_ids(db_embeddings, np.arange(len(db_embeddings)))

    # Select random queries
    query_indices = np.random.choice(len(dataset), num_queries, replace=False)

    for query_idx in query_indices:
        # Encode query based on mode
        if mode.startswith("image"):  # image-to-image or image-to-text
            query_embedding = model.encode_image(
                [dataset.iloc[query_idx]["image_path"]]
            )
        else:  # text-to-text or text-to-image
            query_embedding = model.encode_text([dataset.iloc[query_idx]["caption"]])

        # Search
        _, retrieved_ids = index.search(query_embedding, k=top_k + 1)

        # Remove self match if same modality
        retrieved_ids = retrieved_ids[0]
        if mode == "image-to-image":
            retrieved_ids = [id for id in retrieved_ids if id != query_idx][:top_k]
        else:
            retrieved_ids = retrieved_ids[:top_k]

        # Visualization
        plt.figure(figsize=(15, 3))

        # Plot query
        plt.subplot(1, top_k + 1, 1)
        if mode.startswith("image"):
            query_img = Image.open(dataset.iloc[query_idx]["image_path"])
            plt.imshow(query_img)
            plt.title(
                f'Query Image\n{dataset.iloc[query_idx]["caption"][:50]}...', fontsize=8
            )
        else:  # text-to-text or text-to-image
            plt.text(
                0.5,
                0.5,
                dataset.iloc[query_idx]["caption"],
                ha="center",
                va="center",
                wrap=True,
                fontsize=8,
            )
            plt.title("Query Text", fontsize=8)
        plt.axis("off")

        # Plot retrieved results
        for i, retrieved_id in enumerate(retrieved_ids):
            plt.subplot(1, top_k + 1, i + 2)
            if mode.endswith("image"):  # retrieving images
                retrieved_img = Image.open(dataset.iloc[retrieved_id]["image_path"])
                plt.imshow(retrieved_img)
                plt.title(
                    f'Match {i+1}\n{dataset.iloc[retrieved_id]["caption"][:50]}...',
                    fontsize=8,
                )
            else:  # retrieving text
                plt.text(
                    0.5,
                    0.5,
                    dataset.iloc[retrieved_id]["caption"],
                    ha="center",
                    va="center",
                    wrap=True,
                    fontsize=8,
                )
                plt.title(f"Match {i+1}", fontsize=8)
            plt.axis("off")

        plt.tight_layout()
        plt.show()
