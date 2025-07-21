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
TARGET_COLUMN: str = "Churn"


##Data ingestion constants
DATA_INGESTION_COLLECTION_NAME: str = "churn"
DATA_INGESTION_DIR_NAME: str="Data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str="Raw_Data"
DATA_INGESTION_INGESTED_DIR: str="Ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float=0.20

#Data validation constant
DATA_VALIDATION_DIR_NAME: str = "Data_validation"
DATA_VALIDATION_REPORT_FILE_NAME:str = "Report.yaml"

##Data transformation constants
DATA_TRANSFORMATION_DIR_NAME: str ="Data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR: str = "Transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str ="Transformed_object"
DATA_PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

##Model training constants
MODEL_TRAINER_DIR_NAME: str = "Model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "Trained_model"
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config","model.yaml")
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
TRAINED_MODEL_NAME: str = "model.pkl"

##AWS credentials constant
AWS_ACCESS_KEY_ID_ENV_KEY=os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY_ENV_KEY=os.getenv('AWS_SECRET_ACCESS_KEY')
REGION_NAME=os.getenv('REGION_NAME')


##Dagshub Credentials
DAGSHUB_ACCESS_TOKEN = os.getenv("dagshub_access_token")

DAGSHUB_URL="https://dagshub.com"
REPOSITORY_OWNER='Saroj94',
REPOSITORY_NAME='Churn-MLOps'

##AWS S3: Model Evaluation related constants
MODEL_EVALUATION_DIR_NAME: str = "Model_evaluation"
MODEL_EVALUATION_REPORT_NAME:str = "Model_evlauation_report.json"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "churnstorage"
MODEL_PUSHER_S3_KEY = "model-registry"


##Server details
APP_HOST = "localhost"
APP_PORT = 5000
