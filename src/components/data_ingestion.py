import os, sys
from src.data_access.Fetch_Data import FetchData
from src.constants import *
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from pandas import DataFrame
from sklearn.model_selection import train_test_split

## Create a class for a data ingestion pipeline
class DataIngestion:
    "This class is completely responsible for ingesting the data from the mongodb database."
    def __init__(self,data_ingestion_config: DataIngestionConfig=DataIngestionConfig()):
        "initialize Data ingestion configuration"
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
        
    def export_data_into_feature_store(self)->DataFrame:
        """This method is use to export the records of collection from database and convert them into dataframe.
        """
        try:
            logging.info("Exporting records from collection.")
            ##create object of FetchData class
            fd=FetchData()
            ##use one of the method (Export_collection_as_dataframe) of class FetchData to fetch the records and convert into dataframe 
            logging.info("Use one of the method (Export_collection_as_dataframe) of class FetchData to fetch the records and convert into dataframe")
            dataframe=fd.Export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            ##initialise the directory path name to stored the exported data
            logging.info("Initialise the directory path name to stored the exported data")
            exported_dataframe_store_file_path=self.data_ingestion_config.feature_store_file_path
            ##Returns exported_dataframe_store_file_path as a directory name to store data
            logging.info("Returns exported_dataframe_store_file_path as a directory name to store data")
            dir_name=os.path.dirname(exported_dataframe_store_file_path)
            ##create folder using directory name 
            logging.info("create directory using dir_name and make folder too")
            os.makedirs(dir_name,exist_ok=True)
            logging.info("Saving the exported data into exported_dataframe_store_file_path")
            dataframe.to_csv(exported_dataframe_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise MyException(e,sys)
        
    def split_data_into_train_test(self, dataframe: DataFrame)->None:
        "This method will be responsible for splitting the data into train test split."
        try:
            train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)
            ##assign the direrctory name 
            dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)
            ##create the directory using dir_path
            os.makedirs(dir_path,exist_ok=True)
            #save the train and test set of data into the training folder and testing folder respectively
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Train and Test set of data is saved.")
        except Exception as e:
            raise MyException(e,sys)
    
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        "This method will be responsible to execute the entire data ingestion pipeline."
        try:
            logging.info("Exporting data from mongodb collection into dataframe...")
            dataframe=self.export_data_into_feature_store()
            logging.info("Spliting data into train and test set...")
            self.split_data_into_train_test(dataframe=dataframe)
            logging.info("Saving Train and Test dataset as data ingestion artifact...")
            data_ingestion_artifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                          test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info("Data Ingestion artifact is created.")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e,sys)
        
if __name__=="__main__":
    ingestion=DataIngestion()
    ingestion.initiate_data_ingestion()