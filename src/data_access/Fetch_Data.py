##once the mongodb client connection is establish, we should fetch the data
import sys
import pandas as pd
import numpy as np
from typing import Optional
from src.exception import MyException
from src.configuration.mongo_db_connection import MongodbClient
from src.logger import logging
from src.constants import DATABASE_NAME

##class to fetch data from mongodb

class FetchData:
    """
    Class responsible to fetch/export data from Mongodb database into workspace.

    1. Initializes the mongodb using __init__(self) method.
    2. Use database to data and convert into pandas dataframe
    """

    def __init__(self)->None:
        "Initialize the mongodb client connection."
        try:
            self.mongo_client=MongodbClient(Database=DATABASE_NAME)
        except Exception as e:
            raise MyException(e,sys)
    
    def Export_collection_as_dataframe(self, collection_name: str, Database:Optional[str]=None)-> pd.DataFrame:
        """Exports entire collection records into as pandas DataFrame
        
        Method Parameters:
                        1. collection_name: Name of the mongodb collection to export.
                        2. Database:Optional[str]
                            By default it has DATABASE_NAME that's why it is optional.
        """
        try:
           ##check first the database is available or not
           if Database is None:
               collection=self.mongo_client.database[collection_name]
           else:
               collection=self.mongo_client[Database][collection_name] 
        
           ##convert the collection into a dataframe
           logging.info("Fetching data from Database")
           print("Fetching data from database")
           df=pd.DataFrame(list(collection.find()))
           print("Recorded data is fetched")
           return df
        except Exception as e:
            raise MyException(e,sys)