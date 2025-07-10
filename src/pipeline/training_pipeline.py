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


##Data Transformation stage
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact


##combine all into pipeline for training
class TrainingPipeline:
    def __init__(self):
        """Initialize the all components' configuration."""
        self.data_ingestion_config=DataIngestionConfig()
        self.data_validation_config=DataValidationConfig()
        self.data_transformation_config=DataTransformationConfig()

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
        
    def start_data_transformation(self,data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        "This method will starts the data transformation stage."
        try:
            logging.info("Initializing the Data Transformation class object")
            data_transformation=DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                   data_validation_artifact=data_validation_artifact,
                                                   data_transformation_config=self.data_transformation_config)
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e,sys)        
    def run_pipeline(self,)->None:
        "This method is responsible to run entire pipeline"
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                                        data_validation_artifact=data_validation_artifact)
        except Exception as e:
            raise MyException(e,sys)
        
