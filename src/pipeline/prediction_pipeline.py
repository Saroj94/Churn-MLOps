import sys
from src.entity.config_entity import ChurnPredictorConfig
from src.entity.s3_estimator import ChurnEstimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class CustomerChurnData:
    def __init__(self,
                gender,
                SeniorCitizen,
                Partner,
                Dependents,
                tenure ,
                PhoneService,
                MultipleLines,
                InternetService,
                OnlineSecurity,
                OnlineBackup,
                DeviceProtection,
                TechSupport,
                StreamingTV,
                StreamingMovies,
                Contract,
                PaperlessBilling,
                PaymentMethod,
                MonthlyCharges,
                TotalCharges
                ):
        """
        Churn Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.gender = gender
            self.SeniorCitizen = SeniorCitizen
            self.Partner = Partner
            self.Dependents = Dependents
            self.tenure  = tenure
            self.PhoneService = PhoneService
            self.MultipleLines = MultipleLines
            self.InternetService = InternetService
            self.OnlineSecurity = OnlineSecurity
            self.OnlineBackup = OnlineBackup
            self.DeviceProtection = DeviceProtection
            self.TechSupport = TechSupport
            self.StreamingTV = StreamingTV
            self.StreamingMovies = StreamingMovies
            self.Contract = Contract
            self.PaperlessBilling = PaperlessBilling
            self.PaymentMethod = PaymentMethod
            self.MonthlyCharges = MonthlyCharges
            self.TotalCharges =  TotalCharges
        except Exception as e:
            raise MyException(e, sys) from e


    def get_churn_data_as_dict(self):
        """
        This function returns a dictionary from customerchurn class input
        """
        logging.info("Entered get_churn_data_as_dict method as churn class")

        try:
            input_data = {
                "gender" :[self.gender],
                "SeniorCitizen" :[self.SeniorCitizen],
                "Partner" :[self.Partner],
                "Dependents" :[self.Dependents],
                "tenure"  :[self.tenure],
                "PhoneService" :[self.PhoneService],
                "MultipleLines" :[self.MultipleLines],
                "InternetService" :[self.InternetService],
                "OnlineSecurity" :[self.OnlineSecurity],
                "OnlineBackup" :[self.OnlineBackup],
                "DeviceProtection" :[self.DeviceProtection],
                "TechSupport" :[self.TechSupport],
                "StreamingTV" :[self.StreamingTV],
                "StreamingMovies" :[self.StreamingMovies],
                "Contract" :[self.Contract],
                "PaperlessBilling" :[self.PaperlessBilling],
                "PaymentMethod" :[self.PaymentMethod],
                "MonthlyCharges" :[self.MonthlyCharges],
                "TotalCharges" :[self. TotalCharges]
            }

            logging.info("Created churn data dict")
            logging.info("Exited get_churn_data_as_dict method as churn class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e
        
    def get_churn_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from churn class input
        """
        try:
            
            churn_input_dict = self.get_churn_data_as_dict()
            return DataFrame(churn_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e

class ChurnClassifier:
    def __init__(self,prediction_pipeline_config: ChurnPredictorConfig = ChurnPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of InsuranceClaimDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of InsuranceClaimDataClassifier class")
            model = ChurnEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)