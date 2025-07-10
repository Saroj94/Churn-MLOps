import os
from datetime import datetime
from dotenv import load_dotenv
##loading private variables from the virtual environment
load_dotenv()

##mongodb connection string url to connect with mongodb database
MONGODB_URL=os.getenv("MDB_URL")
DATABASE_NAME="Churndb"
COLLECTION_NAME="churn"

##Artifact folder where all the project outputs are stored
##Training pipeline configuration constants
PIPELINE_NAME: str=""
ARTIFACT_DIR: str = "Artifact"
TIMESTAMP: str=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
FILE_NAME: str="data.csv"
TRAIN_FILE_NAME: str="train.csv"
TEST_FILE_NAME: str="test.csv"
SCHEMA_FILE_PATH=os.path.join("config","schema.yaml")


##Data ingestion constants
DATA_INGESTION_COLLECTION_NAME: str = "churn"
DATA_INGESTION_DIR_NAME: str="Data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str="Raw_Data"
DATA_INGESTION_INGESTED_DIR: str="Ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float=0.30

#Data validation constant
DATA_VALIDATION_DIR_NAME: str = "Data_validation"
DATA_VALIDATION_REPORT_FILE_NAME:str = "Report.yaml"

##Data transformation constants
DATA_TRANSFORMATION_DIR_NAME: str ="Data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR: str = "Transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str ="Transformed_object"
DATA_PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"
