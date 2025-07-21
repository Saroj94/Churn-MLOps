import os,sys
import json
import pandas as pd
from typing import Optional
from src.constants import *
from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import read_yaml_file
from src.entity.s3_estimator import ChurnEstimator
from dataclasses import dataclass
from src.entity.estimator import ChurnModel
from src.entity.estimator import TargetFeatureMapping
from src.entity.artifact_entity import ModelEvaluationArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score, precision_score, recall_score
import dagshub
import mlflow

#______________________________________________________________________________________
# dagshub_token=DAGSHUB_ACCESS_TOKEN
# if dagshub_token is None:
#     raise EnvironmentError("Dagshub environment is not set yet.")

# os.environ("MLFLOW_TRACKING_USERNAME")=dagshub_token
# os.environ("MLFLOW_TRACKING_PASSWORD")=dagshub_token

# dagshub_url=DAGSHUB_URL
# repository_owner=REPOSITORY_OWNER
# repository_name=REPOSITORY_NAME

# mlflow.set_tracking_uri(f'{dagshub_url}/{repository_owner}/{repository_name}.mlflow')

#_______________________________________________________________________________________
model_tracking_uri="https://dagshub.com/Saroj94/Churn-MLOps.mlflow"
dagshub.init(repo_owner='Saroj94', repo_name='Churn-MLOps', mlflow=True)

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    f1_score: float
    precision_score: float
    recall_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[ChurnEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            churn_estimator = ChurnEstimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if churn_estimator.is_model_present(model_path=model_path):
                return churn_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            mapping_func=TargetFeatureMapping().convert_as_dict()
            y = y.replace(mapping_func).astype(int)

            # trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model_f1_score=None
            best_model_precision_score=None
            best_model_recall_score=None

            best_model = self.get_best_model()
            if best_model is not None:
                y_pred_best_model = pd.Series(best_model.predict(x)).replace(mapping_func).astype(int)

                best_model_f1_score = f1_score(y, y_pred_best_model)
                best_model_precision_score=precision_score(y,y_pred_best_model)
                best_model_recall_score=recall_score(y,y_pred_best_model)
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           f1_score=best_model_f1_score,
                                           precision_score=best_model_precision_score,
                                           recall_score=best_model_recall_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score)
            
            logging.info(f"Result: {result}")

            ## With mlflow tracking 
            mlflow.set_experiment(experiment_name="Churn_model_eval_experiment")
            with mlflow.start_run() as run:
                mlflow.log_param("trained_model_path",self.model_trainer_artifact.trained_model_file_path)
                mlflow.log_metric("Trained model f1 score", trained_model_f1_score)
                mlflow.log_metric("Accuracy difference",result.difference)

                if best_model_f1_score is not None:
                    mlflow.log_metric("Best F1 score",best_model_f1_score)
                if best_model_precision_score is not None:
                    mlflow.log_metric("Best Precision score", best_model_precision_score)
                if best_model_recall_score is not None:
                    mlflow.log_metric("Best recall score", best_model_recall_score)
                mlflow.set_tag("MLOps Developer", "Saroj Rai")    
            return result

        except Exception as e:
            raise MyException(e, sys)
        
    def save_eval_artifact(self, model_eval_artifact:ModelEvaluationArtifact)-> str:
        try:
            logging.info("Saving evaluation report.")
            saved_file_path=self.model_eval_config.model_eval_report_name
            logging.info("Create folder for file")
            folder_path_name=self.model_eval_config.model_eval_dir_name
            os.makedirs(folder_path_name,exist_ok=True)

            logging.info("Open the file path and save as dictionary format in the just created folder")
            with open(saved_file_path,"w") as file:
                json.dump(model_eval_artifact.__dict__,file, indent=4)
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
                f1_score=evaluate_model_response.f1_score,
                precision_score=evaluate_model_response.precision_score,
                recall_score=evaluate_model_response.recall_score)
            
            save_model_eval_path=self.save_eval_artifact(model_evaluation_artifact)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys)
        