from e2e_cnnClassifier_ChestCancer import logger
from e2e_cnnClassifier_ChestCancer.pipeline import stage_01_data_ingestions

def main():
    """Main function to execute the pipeline."""
    try:
        logger.info(">>> Starting the data ingestion pipeline <<<")
        
        data_ingestion_pipeline = stage_01_data_ingestions.DataIngestionTrainingPipeline()
        data_ingestion_pipeline.main()
        
        # logger.info(">>> Starting data preprocessing stage <<<")
        # data_preprocessing_pipeline = stage_02_data_preprocessing.DataPreprocessingPipeline()
        # data_preprocessing_pipeline.main()
        
        logger.info(">>> Pipeline execution completed <<<")

    except Exception as e:
        logger.exception(f"An error occurred during pipeline execution: {e}")

if __name__ == '__main__':
    main()
