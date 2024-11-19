import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    Blip2TextModelWithProjection,
    Blip2VisionModelWithProjection,
)

from xretrieval.models.base import ImageModel, MultiModalModel, TextModel
from xretrieval.models_registry import ModelRegistry


@ModelRegistry.register(
    "transformers/Salesforce/blip2-itm-vit-g",
    model_input="text-image",
)
class BLIP2Model(MultiModalModel):
    def __init__(self, model_id: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id.replace("transformers/", "")
        self.text_model, self.vision_model, self.processor = self.load_models()
        self.text_model.to(self.device)
        self.vision_model.to(self.device)
        self.text_model.eval()
        self.vision_model.eval()

    def load_models(self):
        text_model = Blip2TextModelWithProjection.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto"
        )
        vision_model = Blip2VisionModelWithProjection.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_id)
        return text_model, vision_model, processor

    def encode_text(self, captions: list[str], batch_size: int = 32) -> np.ndarray:
        all_features = []
        for i in tqdm(range(0, len(captions), batch_size), desc="Encoding captions"):
            batch_captions = captions[i : i + batch_size]
            inputs = self.processor(
                text=batch_captions, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.inference_mode():
                text_features = (
                    self.text_model(**inputs)
                    .text_embeds.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            all_features.append(text_features[:, 0, :])
        return np.concatenate(all_features, axis=0)

    def encode_image(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        all_features = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = self.processor(
                images=images, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.inference_mode():
                image_features = (
                    self.vision_model(**inputs)
                    .image_embeds.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            all_features.append(image_features[:, 0, :])
        return np.concatenate(all_features, axis=0)


@ModelRegistry.register(
    "transformers/Salesforce/blip2-itm-vit-g-text",
    model_input="text",
)
class BLIP2TextModel(TextModel):
    def __init__(self, model_id: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id.replace("transformers/", "")
        self.model_id = self.model_id.replace("-text", "")
        self.model, self.processor = self.load_model()
        self.model.to(self.device)
        self.model.eval()

    def load_model(self):
        model = Blip2TextModelWithProjection.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_id)
        return model, processor

    def encode_text(self, captions: list[str], batch_size: int = 32) -> np.ndarray:
        all_features = []

        for i in tqdm(range(0, len(captions), batch_size), desc="Encoding captions"):
            batch_captions = captions[i : i + batch_size]
            inputs = self.processor(
                text=batch_captions, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.inference_mode():
                text_features = (
                    self.model(**inputs)
                    .text_embeds.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            all_features.append(text_features[:, 0, :])

        return np.concatenate(all_features, axis=0)


@ModelRegistry.register(
    "transformers/Salesforce/blip2-itm-vit-g-image",
    model_input="image",
)
class BLIP2ImageModel(ImageModel):
    def __init__(self, model_id: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id.replace("transformers/", "")
        self.model_id = self.model_id.replace("-image", "")
        self.model, self.processor = self.load_model()
        self.model.to(self.device)
        self.model.eval()

    def load_model(self):
        model = Blip2VisionModelWithProjection.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_id)
        return model, processor

    def encode_image(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        all_features = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = self.processor(
                images=images, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.inference_mode():
                image_features = (
                    self.model(**inputs)
                    .image_embeds.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            all_features.append(image_features[:, 0, :])

        return np.concatenate(all_features, axis=0)
