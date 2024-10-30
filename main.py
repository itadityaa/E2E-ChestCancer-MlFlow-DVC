from e2e_cnnClassifier_ChestCancer import logger
from e2e_cnnClassifier_ChestCancer.pipeline import stage_01_data_ingestions, stage_02_base_model

def main():
    """Main function to execute the pipeline."""
    try:
        logger.info(">>> Starting the data ingestion pipeline <<<")
        
        data_ingestion_pipeline = stage_01_data_ingestions.DataIngestionTrainingPipeline()
        data_ingestion_pipeline.main()
        
        logger.info(">>> Starting the transfer learning stage <<<")
        base_model_pipeline = stage_02_base_model.PrepareBaseModelTrainingPipeline()
        base_model_pipeline.main()

        
        logger.info(">>> Pipeline execution complete <<<")

    except Exception as e:
        logger.exception(f"An error occurred during pipeline execution: {e}")

if __name__ == '__main__':
    main()
