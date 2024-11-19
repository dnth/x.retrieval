import numpy as np
import timm
import torch
from PIL import Image
from tqdm.auto import tqdm

from xretrieval.models.base import ImageModel
from xretrieval.models_registry import ModelRegistry


@ModelRegistry.register("timm/resnet18.a1_in1k", model_input="image")
class TimmModel(ImageModel):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = self.load_model()
        self.model.eval()
        self.transform = timm.data.create_transform(
            **timm.data.resolve_model_data_config(self.model)
        )

    def load_model(self):
        return timm.create_model(self.model_id.replace("timm/", ""), pretrained=True)

    def encode_image(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        all_features = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            preprocessed = torch.stack([self.transform(img) for img in images])
            with torch.inference_mode():
                features = self.model(preprocessed.to(device))
            all_features.append(features.cpu())

        return torch.cat(all_features, dim=0).numpy()
