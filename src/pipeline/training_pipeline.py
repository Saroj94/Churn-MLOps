import os, sys
from src.exception import MyException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

##combine all into pipeline for training
class TrainingPipeline:
    def __init__(self):
        """Initialize the all components' configuration."""
        self.data_ingestion_config=DataIngestionConfig()

    def start_data_ingestion(self)->DataIngestionArtifact:
        "This method will starts the data ingestion file"
        try:
            logging.info("Initialize the object of DataIngestion class")
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            logging.info("Creating data ingestion artifact as it initiates the data ingestion.")
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def run_pipeline(self,)->None:
        "This method is responsible to run entire pipeline"
        try:
            data_ingestion_artifact=self.start_data_ingestion()
        except Exception as e:
            raise MyException(e,sys)
