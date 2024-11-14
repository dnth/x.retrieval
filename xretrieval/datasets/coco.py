import json
from pathlib import Path

import pandas as pd

from ..registry import DatasetRegistry


@DatasetRegistry.register("coco-val-2017", "The COCO Validation Set")
class COCODataset:
    def __init__(
        self,
        data_dir: str = "/home/dnth/Desktop/automatic-retrieval-benchmark/data/coco/",
    ):
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir / "annotations"
        self.images_dir = self.data_dir / "val2017"

    # TODO: Download dataset if not in local folder
    def load_annotations(self) -> pd.DataFrame:
        """Load and process COCO annotations."""
        # Load caption and instance annotations
        captions_file = self.annotations_dir / "captions_val2017.json"
        instances_file = self.annotations_dir / "instances_val2017.json"

        with open(captions_file, "r") as f:
            coco_captions = json.load(f)
        with open(instances_file, "r") as f:
            coco_instances = json.load(f)

        # Create DataFrames
        df_images = pd.DataFrame(coco_captions["images"])
        df_captions = pd.DataFrame(coco_captions["annotations"])
        df_instances = pd.DataFrame(coco_instances["annotations"])
        df_categories = pd.DataFrame(coco_instances["categories"])

        # Prepare category information
        df_categories = df_categories.rename(columns={"id": "category_id"})
        df_instances = df_instances[["image_id", "category_id"]]

        # Get categories per image
        df_image_categories = pd.merge(
            df_instances, df_categories, on="category_id", how="left"
        )

        # Group categories by image
        df_image_categories = (
            df_image_categories.groupby("image_id")["name"]
            .agg(lambda x: ",".join(sorted(list(set(x)))))
            .reset_index()
        )

        # Merge everything together
        df_images = df_images.rename(columns={"id": "image_id"})
        df = pd.merge(
            df_images[["file_name", "image_id"]],
            df_captions[["image_id", "caption"]],
            how="left",
            on="image_id",
        )

        df = pd.merge(df, df_image_categories, how="left", on="image_id")

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
                    "name": "first",
                }
            )
            .reset_index()
        )

        return df
