import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, Blip2TextModelWithProjection

from xretrieval.models.base import TextModel
from xretrieval.models_registry import ModelRegistry


@ModelRegistry.register(
    "transformers/Salesforce/blip2-itm-vit-g",
    model_input="text",
)
class BLIP2TextModel(TextModel):
    def __init__(self, model_id: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id.replace("transformers/", "")
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
