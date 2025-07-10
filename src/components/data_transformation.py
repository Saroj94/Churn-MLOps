import sys
import pandas as pd
import numpy as np
import pickle
from src.exception import MyException
from src.logger import logging
from src.utils.MyEncoder import MyLabelEncoder
from sklearn.preprocessing import OneHotEncoder
from src.utils.main_utils import save_object,read_yaml_file,save_numpy_array_data
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTEN
from src.constants import *
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact


##Data transformation stage
class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        """Initializing all required objects"""
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
            self.schema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
        
    ##static method specifically use to read the file in csv format
    @staticmethod
    def read_file(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
    
    ##get the data is transformed
    def get_data_transformer_object(self)->Pipeline:
        "This method use to create a preprocessor pipeline object."
        try:
            logging.info("Initializing the objects of all required transformers")
            numeric_power_transformer=PowerTransformer(method='yeo-johnson')
            label_encoder_transformer=MyLabelEncoder()
            categorical_transformer=OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            logging.info("Instances of all transformer is created.")

            logging.info("Loading all schemas to transform.")
            num_transformation_features=self.schema_config['num_transformation_features']
            cat_transformation_features=self.schema_config['cat_transformation_features']
            label_transformation_features=self.schema_config['label_transformation_features']
            logging.info("All features/schemas are ready to transform.")

            logging.info("Setting up the transformer pipeline.")
            preprocessor=ColumnTransformer(
                transformers=[
                    ("Power Transformation",numeric_power_transformer,num_transformation_features),
                    ("Categorical Transformation",categorical_transformer,cat_transformation_features),
                    ("Label Transformation",label_encoder_transformer,label_transformation_features)
                ],remainder='passthrough',sparse_threshold=0.0
            )
            preprocessor_pipeline=Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Preprocessing transformers' final pipeline is ready")
            return preprocessor_pipeline
        except Exception as e:
            raise MyException(e,sys)
        
    ##change the churn values
    # def target_column_mapper(self,df:pd.Series):
    #     try:
    #         logging.info("Churn binary value Yes/No is mapped into 1/0")
    #         target_col=self.schema_config['target_column']
    #         if isinstance(target_col, list):
    #             target = pd.Series(target_col)
    #             mapped_series=target.map({'Yes':1,'No':0})
    #             if mapped_series.isnull().sum() > 0:
    #                 raise MyException("Target column contains unmapped or null values after mapping.", sys)
    #         return mapped_series
    #     except Exception as e:
    #         raise MyException(e,sys)
        
    ##droping the unused column
    def drop_columns(self,df):
        "Drop unused column from the data."
        try:
            logging.info("Droping the unused columns")
            drop_feature=self.schema_config['drop_columns']
            # Ensure columns exist before dropping
            drop_cols = [col for col in drop_feature if col in df.columns]
            if drop_cols:
                df=df.drop(drop_cols,axis=1)
            return df
        except Exception as e:
            raise MyException(e,sys)
    
    ##initiates and combine all of the above methods into one pipeline 
    def initiate_data_transformation(self)->DataTransformationArtifact:
        "A method that initiates to combine all of the above methods into one pipeline before executing."
        try:
            ##bring in target feature
            target_column=self.schema_config['target_column']

            logging.info("Validating the data ⏳")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.validation_message)
            
            logging.info("loading data once data validation is clear.")
            train_df=self.read_file(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df=self.read_file(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train and Test file loaded")

            logging.info("Partioning into independent features and dependent(Target) feature of Train data.")
            input_feature_train_df=train_df.drop(columns=target_column,axis=1)
            target_fetaure_train_df=train_df[target_column]

            logging.info("Partioning into independent features and dependent(Target) feature of Test data.")
            input_feature_test_df=test_df.drop(columns=target_column,axis=1)
            target_fetaure_test_df=test_df[target_column]
            logging.info("Input and Target columns defined for both train and Test data.")

            logging.info("Applying custom transformation methods to just created input and target columns of Train data.")
            # target_fetaure_train_df=self.target_column_mapper(target_fetaure_train_df)
            input_feature_train_df=self.drop_columns(input_feature_train_df)

            logging.info("Applying custom transformation methods to just created input and target columns of Test data.")
            # target_fetaure_test_df=self.target_column_mapper(target_fetaure_test_df)
            input_feature_test_df=self.drop_columns(input_feature_test_df)
            logging.info("Custom transformations applied to Train and Test data")

            logging.info("Starting data transformation by instantiating an object of get_data_transform method.")
            preprocessor=self.get_data_transformer_object()

            logging.info("Fit and Transform the training and testing data")
            input_feature_train_arr=preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor.fit_transform(input_feature_test_df)
            logging.info("Data Transformation is performed on independent feature of training and testing data")

            logging.info("Handling imbalanced data using SMOTEN technique.")
            smtn=SMOTEN(k_neighbors=5,random_state=42,sampling_strategy=0.6)
            logging.info("over sampling on minority data of training dataset")
            final_input_feature_train,final_target_feature_train=smtn.fit_resample(input_feature_train_arr,target_fetaure_train_df)

            logging.info("Over sampling on minority data of testing data.")
            final_input_feature_test,final_target_feature_test=smtn.fit_resample(input_feature_test_arr,target_fetaure_test_df)

            logging.info("Combining final independently sampled data together into one data")
            train_arr=np.c_[final_input_feature_train,np.array(final_target_feature_train)]
            test_arr=np.c_[final_input_feature_test,np.array(final_target_feature_test)]
            logging.info("input feature-target feature concatenation is done for train and test data.")

            logging.info("Saving transformation object and transformed files.")
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)
            logging.info("Data is saved.")

            return DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path)
        
        except Exception as e:
            raise MyException(e,sys)
        
if __name__=="__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    from src.entity.config_entity import DataValidationConfig
    from src.components.data_transformation import DataTransformation
    from src.entity.config_entity import DataTransformationConfig

    data_ingestion_artifact = DataIngestion().initiate_data_ingestion()

    validation_artifact = DataValidation( data_ingestion_artifact=data_ingestion_artifact,
                                          data_validation_config=DataValidationConfig()).initiate_data_validation()
    data_transform=DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                      data_validation_artifact=validation_artifact,
                                      data_transformation_config=DataTransformationConfig())
    data_transform.initiate_data_transformation()

