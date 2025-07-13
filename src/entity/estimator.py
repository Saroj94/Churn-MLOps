import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetFeatureMapping:
    def __init__(self):
        self.Yes:int = 1
        self.No:int = 0
    def convert_as_dict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self.convert_as_dict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))
    

class ChurnModel:
    def __init__(self,preprocessing_obj:Pipeline , trained_model_obj:object):
        self.preprocessing_object=preprocessing_obj
        self.trained_model_obj=trained_model_obj

    def predict(self,dataframe:DataFrame)->DataFrame:
        "This function expects preprocessed inputs(with all custom transformations already applied) and does prediction."
        try:
            logging.info("Applying transformation using pre-trained preprocessed object.")
            transformed_feature=self.preprocessing_object.transform(dataframe)
            logging.info("Perform prediction on transformed fetaure using trained model.")
            prediction=self.trained_model_obj.predict(transformed_feature)
            return prediction
        except Exception as e:
            raise MyException(e,sys)
        
    def __repr__(self):
        return f"{type(self.trained_model_obj).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_obj).__name__}()"