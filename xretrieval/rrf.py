import numpy as np
import pandas as pd
import torch
import torchmetrics
from rich.console import Console
from rich.table import Table

from .core import load_dataset


def run_rrf(
    results_list: list[pd.DataFrame], dataset: str, top_k: int = 10
) -> pd.DataFrame:
    """
    Combines multiple retrieval results using Reciprocal Rank Fusion algorithm.

    Args:
        results_list: List of DataFrames containing retrieval results
        dataset: Dataset supported by xretrieval. Run `xretrieval.list_datasets()` to see available datasets.
        top_k: Number of top results to retrieve

    Returns:
        DataFrame with combined retrieval results
    """

    dataset = load_dataset(dataset)

    # Initialize lists for all columns
    new_retrieved_ids = []
    new_retrieved_paths = []
    new_retrieved_captions = []
    new_retrieved_names = []
    new_is_correct = []

    # Get retrieved IDs from all results
    retrieved_ids_lists = [df["retrieved_ids"].tolist() for df in results_list]

    # Iterate through each query
    for idx in range(len(results_list[0])):
        # Get rankings for current query from all results
        rankings = [results[idx] for results in retrieved_ids_lists]

        # Apply RRF to get sorted doc IDs and limit to top_k
        rrf_scores = reciprocal_rank_fusion(rankings)
        sorted_docs = [
            doc_id
            for doc_id, _ in sorted(
                rrf_scores.items(), key=lambda x: x[1], reverse=True
            )
        ][:top_k]  # Limit to top_k results

        # Get corresponding values from dataset
        paths = [
            dataset[dataset["image_id"] == doc_id]["image_path"].iloc[0]
            for doc_id in sorted_docs
        ]
        captions = [
            dataset[dataset["image_id"] == doc_id]["caption"].iloc[0]
            for doc_id in sorted_docs
        ]
        names = [
            dataset[dataset["image_id"] == doc_id]["name"].iloc[0]
            for doc_id in sorted_docs
        ]

        # Check if retrieved IDs contain the query ID
        query_id = results_list[0].iloc[idx]["query_id"]
        is_correct = [doc_id == query_id for doc_id in sorted_docs]

        # Append to lists
        new_retrieved_ids.append(sorted_docs)
        new_retrieved_paths.append(paths)
        new_retrieved_captions.append(captions)
        new_retrieved_names.append(names)
        new_is_correct.append(is_correct)

    # Create new dataframe with updated columns
    new_df = results_list[0].copy()
    new_df["retrieved_ids"] = new_retrieved_ids
    new_df["retrieved_paths"] = new_retrieved_paths
    new_df["retrieved_captions"] = new_retrieved_captions
    new_df["retrieved_names"] = new_retrieved_names

    # Recalculate is_correct based on query_name matching retrieved_names
    new_df["is_correct"] = new_df.apply(
        lambda row: [name == row["query_name"] for name in row["retrieved_names"]],
        axis=1,
    )

    matches = np.array(new_df["is_correct"].tolist())
    matches = torch.tensor(matches, dtype=torch.float16)
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

    return eval_metrics_results, new_df


def reciprocal_rank_fusion(ranked_lists: list[list], bias: int = 60) -> dict:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion algorithm.

    Args:
        ranked_lists: List of lists, where each sublist contains document IDs in ranked order
        bias: Constant that smooths the impact of high rankings (default: 60)

    Returns:
        Dictionary mapping document IDs to their combined RRF scores, sorted by score
    """
    fusion_scores = {}

    # Calculate RRF score for each document in each ranking
    for ranked_list in ranked_lists:
        for position, document_id in enumerate(ranked_list, start=1):
            if document_id not in fusion_scores:
                fusion_scores[document_id] = 0

            # RRF formula: 1 / (rank + bias)
            rrf_score = 1 / (position + bias)
            fusion_scores[document_id] += rrf_score

    # Sort documents by their fusion scores in descending order
    sorted_results = dict(
        sorted(fusion_scores.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_results
