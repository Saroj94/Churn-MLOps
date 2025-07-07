import os, sys
from src.exception import MyException
from src.logger import logging

##Data Ingestion stage
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

##Data validation stage
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact


##combine all into pipeline for training
class TrainingPipeline:
    def __init__(self):
        """Initialize the all components' configuration."""
        self.data_ingestion_config=DataIngestionConfig()
        self.data_validation_config=DataValidationConfig()

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
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact)->DataValidationArtifact:
        "This method will starts the data validation stage."
        try:
            logging.info("Initialization of Data Validation Class object.")
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                           data_validation_config=self.data_validation_config)
            data_validation_artifact=data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def run_pipeline(self,)->None:
        "This method is responsible to run entire pipeline"
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
        except Exception as e:
            raise MyException(e,sys)
        
