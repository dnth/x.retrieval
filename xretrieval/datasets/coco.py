import json
from pathlib import Path
from typing import Literal

import pandas as pd


class COCODataset:
    def __init__(self, data_dir: str = "data/coco"):
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir / "annotations"
        self.images_dir = self.data_dir / "val2017"

    def load_annotations(self) -> pd.DataFrame:
        """Load and process COCO annotations."""
        # Load caption annotations
        captions_file = self.annotations_dir / "captions_val2017.json"
        with open(captions_file, "r") as f:
            coco_captions = json.load(f)

        # Create DataFrames
        df_images = pd.DataFrame(coco_captions["images"])
        df_captions = pd.DataFrame(coco_captions["annotations"])

        # Merge images with captions
        df_images = df_images.rename(columns={"id": "image_id"})
        df = pd.merge(
            df_images[["file_name", "image_id"]],
            df_captions[["image_id", "caption"]],
            how="left",
            on="image_id",
        )

        # Add full image paths
        df["image_path"] = df.file_name.apply(lambda x: str(self.images_dir / x))

        return df

    def get_dataset(self) -> pd.DataFrame:
        """Get the processed COCO dataset."""
        df = self.load_annotations()

        # Group multiple captions per image into a single row
        df = (
            df.groupby("image_id")
            .agg(
                {
                    "file_name": "first",
                    "image_path": "first",
                    "caption": lambda x: " ".join(x),
                }
            )
            .reset_index()
        )

        return df
