from e2e_cnnClassifier_ChestCancer.config.configuration import ConfigurationManager
from e2e_cnnClassifier_ChestCancer.components.base_model import BaseModel
from e2e_cnnClassifier_ChestCancer import logger

STAGE_NAME = 'PREPARE BASE MODEL STAGE'

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        """Initializes the PrepareBaseModelTrainingPipeline class."""
        self.config_manager = ConfigurationManager()   
        self.base_model_config = self.config_manager.get_base_model_config()   
        self.base_model = BaseModel(config=self.base_model_config)  

    def main(self):
        """Executes the base model preparation process."""
        try:
            logger.info(f">>>>>> {STAGE_NAME} start <<<<<<")
            self.base_model.download_and_save_base_model()
            self.base_model.update_base_model()
            logger.info(f">>>>>> {STAGE_NAME} complete <<<<<<")
        except Exception as e:
            logger.exception(f"Error occurred in {STAGE_NAME}: {e}")
            raise e

if __name__ == '__main__':
    pipeline = PrepareBaseModelTrainingPipeline()
    pipeline.main()
