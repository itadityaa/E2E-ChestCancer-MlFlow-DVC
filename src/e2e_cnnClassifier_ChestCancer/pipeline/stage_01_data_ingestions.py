from e2e_cnnClassifier_ChestCancer.components.data_ingestion import DataIngestion
from e2e_cnnClassifier_ChestCancer.config.configuration import ConfigurationManager


try:
    config_manager = ConfigurationManager()   
    data_ingestion_config = config_manager.get_data_ingestion_config()   
    data_ingestion = DataIngestion(config=data_ingestion_config)  
    data_ingestion.download_dataset()
except Exception as e:
    raise e