from .datasets_registry import DatasetRegistry
from .models_registry import ModelRegistry


def list_datasets(search: str = ""):
    # Convert wildcard pattern to simple regex-like matching
    search = search.replace("*", "").lower()
    return [ds for ds in DatasetRegistry.list() if search in ds.lower()]


def list_models(search: str = ""):
    # Convert wildcard pattern to simple regex-like matching
    search = search.replace("*", "").lower()
    return [model for model in ModelRegistry.list() if search in model.lower()]


def load_dataset(name: str):
    dataset_class = DatasetRegistry.get(name)
    return dataset_class.get_dataset()


def load_model(model_id: str):
    model_class = ModelRegistry.get(model_id)
    return model_class(model_id=model_id).load_model()


def run_benchmark(dataset_name: str, model_id: str):
    dataset = load_dataset(dataset_name)
    model = load_model(model_id)

    image_ids = dataset.image_id.tolist()

    import numpy as np

    image_ids = np.array(image_ids)
    labels = dataset.loc[(dataset.image_id.isin(image_ids))].name.to_numpy()

    embeddings = model.encode(dataset.caption)

    import faiss

    index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
    faiss.normalize_L2(embeddings)
    index.add_with_ids(embeddings, np.arange(len(embeddings)))

    _, retrieved_ids = index.search(embeddings, k=10)

    # Remove self matches
    from tqdm.auto import tqdm

    top_k = 10

    filtered_retrieved_ids = []

    for idx, row in enumerate(tqdm(retrieved_ids)):
        filtered_row = [x for x in row if x != idx]

        if len(filtered_row) != top_k - 1:
            filtered_row = filtered_row[: top_k - 1]

        filtered_retrieved_ids.append(filtered_row)

    filtered_retrieved_ids = np.array(filtered_retrieved_ids)

    matches = np.expand_dims(labels, axis=1) == labels[filtered_retrieved_ids]

    import torch

    matches = torch.tensor(np.array(matches), dtype=torch.float16)
    targets = torch.ones(matches.shape)

    indexes = (
        torch.arange(matches.shape[0]).view(-1, 1)
        * torch.ones(1, matches.shape[1]).long()
    )

    import torchmetrics

    metrics = [
        torchmetrics.retrieval.RetrievalMRR(),
        torchmetrics.retrieval.RetrievalNormalizedDCG(),
        torchmetrics.retrieval.RetrievalPrecision(),
        torchmetrics.retrieval.RetrievalRecall(),
        torchmetrics.retrieval.RetrievalHitRate(),
        torchmetrics.retrieval.RetrievalMAP(),
    ]
    results = {}

    for metr in metrics:
        score = round(metr(targets, matches, indexes).item(), 4)
        metr_name = metr.__class__.__name__.replace("Retrieval", "")
        results[metr_name] = score
        print(f"{metr_name}: {score}")
