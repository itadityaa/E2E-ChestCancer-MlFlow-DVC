from e2e_cnnClassifier_ChestCancer.components.data_ingestion import DataIngestion
from e2e_cnnClassifier_ChestCancer.config.configuration import ConfigurationManager
from e2e_cnnClassifier_ChestCancer import logger

STAGE_NAME = 'DATA INGESTION STAGE'

class DataIngestionTrainingPipeline:
    def __init__(self):
        """Initializes the DataIngestionTrainingPipeline class."""
        self.config_manager = ConfigurationManager()   
        self.data_ingestion_config = self.config_manager.get_data_ingestion_config()   
        self.data_ingestion = DataIngestion(config=self.data_ingestion_config)  

    def main(self):
        """Executes the data ingestion process."""
        try:
            logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
            self.data_ingestion.download_dataset()
            logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n")
        except Exception as e:
            logger.exception(f"Error occurred in {STAGE_NAME}: {e}")
            raise e

if __name__ == '__main__':
    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()
