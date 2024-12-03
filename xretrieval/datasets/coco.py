import json
from pathlib import Path

import pandas as pd
from loguru import logger

from ..datasets_registry import DatasetRegistry


@DatasetRegistry.register("coco-val-2017", "The COCO Validation Set with 5k images.")
class COCODataset:
    def __init__(
        self,
        data_dir: str = "./data/coco/",
    ):
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir / "annotations"
        self.images_dir = self.data_dir / "val2017"

        # Check if dataset exists, download if not
        if not self.images_dir.exists():
            logger.info(f"COCO validation dataset not found in {self.images_dir}")
            logger.info("Downloading COCO validation dataset...")
            self.download()
        else:
            logger.info(
                f"COCO validation dataset found in {self.images_dir}, skipping download"
            )

    def download(self):
        """Download and extract COCO validation dataset and annotations if not already present."""
        import os
        import zipfile

        import requests
        from tqdm import tqdm

        def download_file(url, filename):
            """Helper function to download a file with progress bar."""
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            with open(filename, "wb") as file, tqdm(
                desc=f"Downloading {filename}",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)

        # Store the original working directory
        original_dir = os.getcwd()

        try:
            # Create directories if they don't exist
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.data_dir / "annotations", exist_ok=True)

            # Change to the coco directory
            os.chdir(self.data_dir)

            # Download and extract validation images
            logger.info("Downloading COCO validation dataset...")
            download_file(
                "http://images.cocodataset.org/zips/val2017.zip", "val2017.zip"
            )

            logger.info("Extracting images...")
            with zipfile.ZipFile("val2017.zip", "r") as zip_ref:
                zip_ref.extractall()
            os.remove("val2017.zip")

            # Download and extract annotations
            logger.info("Downloading COCO annotations...")
            download_file(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                "annotations_trainval2017.zip",
            )

            logger.info("Extracting annotations...")
            with zipfile.ZipFile("annotations_trainval2017.zip", "r") as zip_ref:
                zip_ref.extractall()
            os.remove("annotations_trainval2017.zip")

            logger.info("Download and extraction complete!")

        finally:
            # Always restore the original working directory
            os.chdir(original_dir)

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


@DatasetRegistry.register(
    "coco-val-2017-blip2-captions",
    "The COCO Validation Set with 5k images and BLIP2 captions.",
)
class COCODatasetBLIP2Captions(COCODataset):
    def load_annotations(self) -> pd.DataFrame:
        url = "https://github.com/dnth/x.retrieval/releases/download/v0.1.1/blip2_captioned_coco_val_2017.parquet"
        df = pd.read_parquet(url)
        return df


@DatasetRegistry.register(
    "coco-val-2017-vlrm-captions",
    "The COCO Validation Set with 5k images and VLRM captions.",
)
class COCODatasetVLRMCaptions(COCODataset):
    def load_annotations(self) -> pd.DataFrame:
        url = "https://github.com/dnth/x.retrieval/releases/download/v0.1.1/vlrm_captioned_coco_val_2017.parquet"
        df = pd.read_parquet(url)
        return df
