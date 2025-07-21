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

##Model trainer stage
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact

##Model Evaluation stage
from src.components.model_evaluation import ModelEvaluation
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifact

##Model pusher stage
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelPusherArtifact


##combine all into pipeline for training
class TrainingPipeline:
    def __init__(self):
        """Initialize the all components' configuration."""
        self.data_ingestion_config=DataIngestionConfig()
        self.data_validation_config=DataValidationConfig()
        self.data_transformation_config=DataTransformationConfig()
        self.model_trainer_config=ModelTrainerConfig()
        self.model_evaluation_config=ModelEvaluationConfig()
        self.model_pusher_config=ModelPusherConfig()

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

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model training
        """
        try:
            model_trainer=ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                       model_trainer_config=self.model_trainer_config)
            model_trainer_artifact=model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def start_model_evaluation(self,data_ingestion_artifact=DataIngestionArtifact,
                               model_trainer_artifact=ModelTrainerArtifact)->ModelEvaluationArtifact:
        """
        This method of TrainPipeline class is responsible for starting modle evaluation
        """
        try:
            model_evaluation=ModelEvaluation(model_eval_config=self.model_evaluation_config,
                                            data_ingestion_artifact=data_ingestion_artifact,
                                             model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact=model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e,sys)
    
    def start_model_pusher(self, model_evaluation_artifact=ModelEvaluationArtifact)->ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible for starting model pushing
        """
        try:
            model_pusher=ModelPusher(model_pusher_config=self.model_pusher_config,
                                     model_evaluation_artifact=model_evaluation_artifact)
            model_pusher_artifact=model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e,sys)      
    def run_pipeline(self,)->None:
        "This method is responsible to run entire pipeline"
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                                        data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact=self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                  model_trainer_artifact=model_trainer_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                logging.info(f"Model not accepted.")
                return None 
            model_pusher_artifact=self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
        except Exception as e:
            raise MyException(e,sys)
        
