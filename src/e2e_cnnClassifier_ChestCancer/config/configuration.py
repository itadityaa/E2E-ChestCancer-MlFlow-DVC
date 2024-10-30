from e2e_cnnClassifier_ChestCancer.constants import *
from e2e_cnnClassifier_ChestCancer.utils.utils import read_yaml, create_directories
from e2e_cnnClassifier_ChestCancer.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: str = CONFIG_FILE_PATH,
        params_filepath: str = PARAMS_FILE_PATH
    ):
        """
        Initializes the ConfigurationManager by loading configuration and parameter files.

        Args:
            config_filepath (str): Path to the config YAML file.
            params_filepath (str): Path to the params YAML file.
        """
        self.config = self._read_yaml(config_filepath)
        self.params = self._read_yaml(params_filepath)

        # self._print_config_and_params()

        self._create_directories([self.config.artificats_root])

    def _read_yaml(self, filepath: str):
        """Reads a YAML file and returns its contents."""
        return read_yaml(filepath)

    def _create_directories(self, dirs: list):
        """Creates directories if they don't exist."""
        create_directories(dirs)

    def _print_config_and_params(self):
        """Prints the contents of the config and params files."""
        print("Configuration Contents:")
        print(self.config)

        print("\nParameters Contents:")
        print(self.params)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Prepares and returns the DataIngestionConfig object with necessary configurations.

        Returns:
            DataIngestionConfig: Configuration object for data ingestion.
        """
        config = self.config.data_ingestion

        # Ensure data ingestion root directory exists
        self._create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            unzip_dir=config.unzip_dir
        )
