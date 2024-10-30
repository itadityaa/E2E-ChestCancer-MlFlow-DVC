import os
from dotenv import load_dotenv
from e2e_cnnClassifier_ChestCancer.entity.config_entity import DataIngestionConfig
from pathlib import Path
import json

load_dotenv()

kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')

from kaggle.api.kaggle_api_extended import KaggleApi


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.api = KaggleApi()
        self.api.authenticate()
        self.metadata_path = os.path.join(self.config.unzip_dir, "dataset_metadata.json")

    def _record_metadata(self, folder_paths):
        """
        Records metadata about downloaded folders and files.
        """
        metadata = {
            "folders": {},
            "source_URL": self.config.source_URL
        }

        for folder_path in folder_paths:
            files = list(Path(folder_path).rglob("*.*"))
            metadata["folders"][folder_path] = len(files)

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f)

    def _load_metadata(self):
        """
        Loads existing metadata from the metadata file.
        """
        if not os.path.exists(self.metadata_path):
            return None
        
        with open(self.metadata_path, "r") as f:
            return json.load(f)

    def _check_if_download_needed(self):
        """
        Checks if the dataset needs to be downloaded based on the metadata.
        """
        existing_metadata = self._load_metadata()

        if not existing_metadata or existing_metadata["source_URL"] != self.config.source_URL:
            return True  

        for folder_path, file_count in existing_metadata["folders"].items():
            if not os.path.exists(folder_path):
                return True  
            if len(list(Path(folder_path).rglob("*.*"))) != file_count:
                return True  

        return False  

    def download_dataset(self):
        """
        Downloads the dataset only if it hasn't been downloaded or has been updated.
        """
        if self._check_if_download_needed():
            print("Downloading dataset...")
            self.api.dataset_download_files(self.config.source_URL, path=self.config.unzip_dir, unzip=True)

            folder_paths = [str(folder) for folder in Path(self.config.unzip_dir).glob("*") if folder.is_dir()]
            self._record_metadata(folder_paths)
            print("Dataset downloaded and metadata recorded.")
        else:
            print("Dataset already up-to-date, no download necessary.")
