import os,sys,json
import pandas as pd
from pandas import DataFrame
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact

##Data validation class which perform 
class DataValidation:
    """This class performs the a complete data validation on incoming data and 
    generate a data validation report in a yaml format"""
    
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        self.data_ingestion_artifact=data_ingestion_artifact
        self.data_validation_config=data_validation_config
        self.schema_config=read_yaml_file(SCHEMA_FILE_PATH)

    #validated the nnumber of columns in a dataframe
    def validate_number_of_columns(self,dataframe: DataFrame)->bool:
        """Validate the number of columns"""
        try:
            logging.info("validating columns of the dataframe.")
            status=len(dataframe.columns)==len(self.schema_config["columns"])
            logging.info(f"Is required columns present :[{status}] ")
            return status
        except Exception as e:
            raise MyException(e,sys)
        
    #checking all column names are present or not
    def is_columns_exist(self,dataframe: DataFrame)->bool:
        try:
            dataframe_columns=dataframe.columns
            missing_categorical_cols=[]
            missing_numerical_cols=[]
            logging.info("Entering into the categorical columns validation block of code.")
            for cols in self.schema_config["categorical_columns"]:
                if cols not in dataframe_columns:
                    missing_categorical_cols.append(cols)
            if len(missing_categorical_cols)>0:
                logging.info(f"Missing categorical columns are: {missing_categorical_cols}")

            logging.info("Enterign into numerical columns validation block of code.")
            for cols in self.schema_config["numerical_columns"]:
                if cols not in dataframe_columns:
                    missing_numerical_cols.append(cols)
            if len(missing_numerical_cols)>0:
                logging.info(f"Missing numerical columns are: {missing_numerical_cols}")
            return False if len(missing_categorical_cols)>0 or len(missing_numerical_cols)>0 else True
        except Exception as e:
            raise MyException(e,sys)
        
    ##read data
    @staticmethod
    def read_data(file_path)->DataFrame:
        try:
            df=pd.read_csv(file_path)
            return df
        except Exception as e:
            raise MyException(e,sys)
        
    ##initiate the data validation function to validate the data
    def initiate_data_validation(self)->DataValidationArtifact:
        "Execute entire data validation pipeline functions to validate the data."
        try:
            validation_error_msg=""
            logging.info("Starts data validation by importing data ingestion train and test set of artifact.")
            train_df,test_df=(DataValidation.read_data(self.data_ingestion_artifact.trained_file_path), 
                             DataValidation.read_data(self.data_ingestion_artifact.test_file_path))
            logging.info("Train and Test data is imported.")

            logging.info("Start Validation on training set of data i.e train_df")
            status=self.validate_number_of_columns(dataframe=train_df)
            if not status:
                validation_error_msg +="Columns is missing in training dataframe."
            else:
                logging.info(f"All required columns are present in the training data :[{status}]")

            logging.info("Validation by datatype of the columns on training set of data.")
            status=self.is_columns_exist(dataframe=train_df)
            if not status:
                validation_error_msg +="Columns is missing based on datatype of training set of data."
            else:
                logging.info("All required categorical/numerical columns are present in the training data.")

            logging.info("Start Validation on Test data i.e Test_df")
            status=self.validate_number_of_columns(test_df)
            if not status:
                validation_error_msg +="Columns is missing in testing set of data."
            else:
                logging.info(f"All required columns are present in the testing data:{status}")
            
            logging.info("Validation by datatype of the columns on testing set of data.")
            status=self.is_columns_exist(test_df)
            if not status:
                validation_error_msg +="Columns is missing based on datatype of testing set of data."
            else:
                logging.info("All required categorical/numerical columns are present in the testing data.")

            ##checking the presence of validation error msg 
            logging.info("checking the presence of validation error msg.")
            validation_status=len(validation_error_msg)==0

            ##artifact 
            logging.info("Initializing the data validation artifact.")
            data_validation_artifact=DataValidationArtifact(
                                    validation_message=validation_error_msg,
                                    validation_status=validation_status,
                                    validation_report_file_path=self.data_validation_config.data_validation_report_file_path
            )

            #directory for the validation report
            report_dir_name=os.path.dirname(self.data_validation_config.data_validation_report_file_path)
            os.makedirs(report_dir_name, exist_ok=True)

            logging.info("Saving validation status and message in JSON format.")
            validation_report={
                                "Validation_status": validation_status,
                                "Message":validation_error_msg.strip()
            }

            logging.info("Data artifact created and save as JSON format file.")
            with open(self.data_validation_config.data_validation_report_file_path,"w") as report_obj:
                json.dump(validation_report, report_obj, indent=4)
            return data_validation_artifact
        except Exception as e:
            raise MyException(e,sys)
        
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    from src.entity.config_entity import DataValidationConfig

    data_ingestion_artifact = DataIngestion().initiate_data_ingestion()

    validation = DataValidation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_config=DataValidationConfig()
    )
    validation.initiate_data_validation()


