import pandas as pd

from .core import load_dataset


def run_rrf(results_list: list, dataset: str) -> pd.DataFrame:
    """
    Combines multiple retrieval results using Reciprocal Rank Fusion algorithm.

    Args:
        results_list: List of DataFrames containing retrieval results
        dataset: Dataset containing image mappings
        bias: RRF bias parameter (default: 60)

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

        # Apply RRF to get sorted doc IDs
        rrf_scores = reciprocal_rank_fusion(rankings)
        sorted_docs = [
            doc_id
            for doc_id, _ in sorted(
                rrf_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

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
    new_df["is_correct"] = new_is_correct

    return new_df


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
