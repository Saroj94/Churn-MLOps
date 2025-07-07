import os,sys
import yaml
import dill
import numpy as np
from pandas import DataFrame
from src.exception import MyException
from src.logger import logging

##function to read/load yaml file
def read_yaml_file(file_path: str)-> dict:
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise MyException(e,sys)

##function use to write on yaml file
def write_yaml_file(file_path: str, content: object, replace: bool=False)->None:
    """
    Parameters:
             1. file_path: requires a yaml file
             2. content: requires the text/values to write
             3. replace: anything exist then replace, basically, this avoids issues with overwriting files that might be locked, 
                corrupted, or contains old content
    """ 
    try:
        if replace:## see replace is False
            if os.path.exists(file_path): ##search for already existing file_path/file
                os.remove(file_path) ## safely remove/delete the already existing file/file_path
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as file_obj:
            yaml.dump(content,file_obj)
    except Exception as e:
        raise MyException(e,sys)
    
##function to load the file/object
def load_object(file_path: str)-> object:
    """This is responsible to load the file/object from the directory and returns model/object.
            Parameter:
                    1. File_path: provide file name.
    """
    try:
        with open(file_path,'rb') as file_obj:
            file=dill.load(file_obj)
            return file
    except Exception as e:
        raise MyException(e,sys)
    
##function use to save the numpy array object
def save_numpy_array_data(file_path: str, array: np.array):
    """    Save numpy array data to file.
        Parameters:
                1. file_path: str location of file to save
                2. array: np.array data to save
    """
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise MyException(e,sys)
    
##function to load the numpy array
def load_numpy_array(file_path: str)-> np.array:
    """    load numpy array data from file
        Parameters:
            1. file_path: str location of file to load
            2. return: np.array data loaded
    """
    try:
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise MyException(e,sys)
    
##function save any type of object
def save_object(file_path: str, objects: object)->None:
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(objects, file_obj)
    except Exception as e:
        raise MyException(e,sys)
    
## function use to drop the columns from data
def drop_column(data: DataFrame, column: list)->DataFrame:
    """
     drop the columns form a pandas DataFrame
     Parameters:
            data: pandas DataFrame
            column: list of columns to be dropped
    """
    try:
        df=data.drop(columns=column, axis=1)
        return df
    except Exception as e:
        raise MyException(e,sys)