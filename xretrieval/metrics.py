import numpy as np
import torch
import torchmetrics
from rich.console import Console
from rich.table import Table


def calculate_retrieval_metrics(
    labels: np.ndarray, retrieved_ids: np.ndarray, top_k: int
) -> dict:
    """
    Calculate retrieval metrics given labels and retrieved IDs

    Args:
        labels: Ground truth labels for each query
        retrieved_ids: Retrieved IDs for each query
        top_k: Number of top results retrieved

    Returns:
        dict: Dictionary containing metric names and scores
    """
    matches = np.expand_dims(labels, axis=1) == labels[retrieved_ids]
    matches = torch.tensor(np.array(matches), dtype=torch.float16)
    targets = torch.ones(matches.shape)
    indexes = (
        torch.arange(matches.shape[0]).view(-1, 1)
        * torch.ones(1, matches.shape[1]).long()
    )

    metrics = [
        torchmetrics.retrieval.RetrievalMRR(),
        torchmetrics.retrieval.RetrievalNormalizedDCG(),
        torchmetrics.retrieval.RetrievalPrecision(),
        torchmetrics.retrieval.RetrievalRecall(),
        torchmetrics.retrieval.RetrievalHitRate(),
        torchmetrics.retrieval.RetrievalMAP(),
    ]
    eval_metrics_results = {}

    for metr in metrics:
        score = round(metr(targets, matches, indexes).item(), 4)
        metr_name = metr.__class__.__name__.replace("Retrieval", "")
        eval_metrics_results[metr_name] = score

    # Print metrics in a rich table
    table = Table(title=f"Retrieval Metrics @ k={top_k}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta")

    for metric_name, score in eval_metrics_results.items():
        table.add_row(metric_name, f"{score:.4f}")

    console = Console()
    console.print(table)

    return eval_metrics_results
