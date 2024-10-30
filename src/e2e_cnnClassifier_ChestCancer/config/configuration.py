from e2e_cnnClassifier_ChestCancer.constants import *
from e2e_cnnClassifier_ChestCancer.utils.utils import read_yaml, create_directories
from e2e_cnnClassifier_ChestCancer.entity.config_entity import BaseModelConfigAndParams, DataIngestionConfig


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

    def get_base_model_config(self) -> BaseModelConfigAndParams:
        """Creates and returns an instance of BaseModelConfigAndParams from the loaded configurations and parameters."""
        
        self._create_directories([Path(self.config['base_model']['root_dir'])])

        return BaseModelConfigAndParams(
            base_model_root_dir=Path(self.config['base_model']['root_dir']),
            base_model_path=Path(self.config['base_model']['base_model_path']),
            updated_model_path=Path(self.config['base_model']['updated_model']),
            
            model_name=self.params['model']['name'],
            model_input_shape=self.params['model']['input_shape'],
            model_num_classes=self.params['model']['num_classes'],
            
            data_train_data_dir=Path(self.params['data']['train_data_dir']),
            data_batch_size=self.params['data']['batch_size'],
            data_image_size=self.params['data']['image_size'],
            
            training_weights=self.params['training']['weights'],
            training_include_top=self.params['training']['include_top'],
            training_epochs=self.params['training']['epochs'],
            training_learning_rate=self.params['training']['learning_rate'],
        )