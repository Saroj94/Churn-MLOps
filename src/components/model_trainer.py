import sys
from typing import Tuple
from src.logger import logging
from src.exception import MyException
import numpy as np
from sklearn.ensemble import VotingClassifier
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (DataTransformationArtifact,
                                        ModelTrainerArtifact,
                                        ClassificationMetricArtifact)
from src.utils.main_utils import (load_object,
                                  load_numpy_array,
                                  save_object,
                                  read_yaml_file)
from src.entity.model_resgistry import MODEL_REGISTRY
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from src.entity.estimator import ChurnModel

##Model training class
class ModelTrainer:
    "This class responsible for training the model"
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, 
                        model_trainer_config:ModelTrainerConfig):
        """data tranformation artifact
           model trainer config 
        """
        self.data_transformation_artifact=data_transformation_artifact
        self.model_trainer_config=model_trainer_config

    def load_model(self, model_file_path:str)->VotingClassifier:
        "This method responsible for loading the model from yaml"
        try:
            logging.info("Reading the yaml file to load VotingClassifier and voting type.")
            model_config=read_yaml_file(file_path=model_file_path)
            voting_config_name=model_config["VotingClassifier"]
            voting_type=voting_config_name["voting"]

            estimators=[]
            logging.info("Iterating through each estimator/model")
            for esti in voting_config_name["estimators"]:
                model_name=esti['model']
                model_params=esti.get("params",{})
                logging.info("Entering into the base estimator.")
                if model_name=="AdaBoostClassifier" and "estimator" in model_params:
                    base_estimator_info=model_params["estimator"]
                    base_model_name=base_estimator_info["model"]
                    base_model_params=base_estimator_info.get("params",{})
                    base_model_instance=MODEL_REGISTRY[base_model_name](**base_model_params)

                    ##replace base_estimator dict by its original instance
                    model_params["estimator"]=base_model_instance
                    del model_params["estimator"] ##remove the original dict
                model_instance = MODEL_REGISTRY[model_name](**model_params)
                estimators.append((esti['model'],model_instance))
            return VotingClassifier(estimators=estimators,voting=voting_type)
        except Exception as e:
            raise MyException(e,sys)
        
    def get_model_train_and_report(self, train:np.array, test:np.array)->Tuple[object,object]:
        "This method responsible to train model on np.array data, generate model performance report."
        try:
            logging.info("get models and report.")
            Voting_model=self.load_model(model_file_path=self.model_trainer_config.model_config_file_path)
            x_train,y_train,x_test,y_test=train[:, :-1],train[:, -1],test[:, :-1],test[:, -1]
            Voting_model.fit(x_train,y_train)
            logging.info("Prediction")
            y_pred=Voting_model.predict(x_test)
            accuracy=accuracy_score(y_test,y_pred)
            f1score=f1_score(y_test,y_pred)
            recallscore=recall_score(y_test,y_pred)
            precisionscore=precision_score(y_test,y_pred)

            metric_artifact=ClassificationMetricArtifact(f1_score=f1score,
                                                         recall_score=recallscore,
                                                         precision_score=precisionscore,
                                                         accuracy_score=accuracy)
            return Voting_model,metric_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        """This method use to initiate model trianing and wrap all into pipeline for prediction."""
        try:
            logging.info("Loading the transformed train and test data.")
            train_arr=load_numpy_array(file_path = self.data_transformation_artifact.transformed_train_file_path)
            test_arr=load_numpy_array(file_path=self.data_transformation_artifact.transformed_test_file_path)

            logging.info("Train model and get artifact")
            trained_model, model_metric_artifact=self.get_model_train_and_report(train=train_arr, test=test_arr)

            logging.info("Loading preprocessing object for prediction purpose.")
            preprocess_model=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            logging.info("Comparing whether the trained model's accuracy meets the threshold accuracy.")
            if accuracy_score(train_arr[:,-1],trained_model.predict(train_arr[:,:-1]))<self.model_trainer_config.expected_accuracy:
                logging.info("Model did not meet the expected accuracy threshold.")
                raise Exception("Model did not meet the expected accuracy threshold.")
            
            logging.info("Wrapping up the final model and Preprocessing model in a Pipeline and save.")
            final_model= ChurnModel(preprocessing_obj=preprocess_model, trained_model_obj=trained_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path,objects=final_model)
            logging.info("Saved the final model object that includes both preprocessing and the trained model")

            logging.info("Create the model trainer artifact.")
            model_trainer_artifact=ModelTrainerArtifact(
                                        trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                        metric_artifact=model_metric_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e,sys)
        
if __name__=="__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer
    from src.entity.config_entity import DataValidationConfig
    from src.entity.config_entity import DataTransformationConfig
    from src.entity.config_entity import ModelTrainerConfig

    data_ingestion_artifact=DataIngestion().initiate_data_ingestion()
    data_validation_artifact=DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                            data_validation_config=DataValidationConfig()).initiate_data_validation()
    data_transformation_config=DataTransformation(
                                                  data_ingestion_artifact=data_ingestion_artifact,
                                                  data_validation_artifact=data_validation_artifact,
                                                  data_transformation_config=DataTransformationConfig()
                                                  )
    data_transformation_artifact=data_transformation_config.initiate_data_transformation()
    model_trainer=ModelTrainer(
                                data_transformation_artifact=data_transformation_artifact,
                                model_trainer_config=ModelTrainerConfig())
    model_trainer.initiate_model_trainer()
        
    


