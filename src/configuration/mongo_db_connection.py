##This file helps to establish connection setup but not fetch the data
import os
import sys
import pymongo
import certifi
import pymongo
from src.exception import MyException
from src.logger import logging
from src.constants import DATABASE_NAME,MONGODB_URL

##load the certificate authority file to avoid timeout errors while connecting with database
ca=certifi.where()

##to setup mongodb client 
class MongodbClient:
    """MongodbClient class is responsible to establish the connection with mongodb database.
    Class Attributes:
                1. Client (Class attribute/Static attribute)
                2. Database name (Method attribute)  
    Methods:
            __init__(Database name)             
    """
    client=None
    def __init__(self, Database: str=DATABASE_NAME)->None:
        """It initializes the connection of Mongodb database. 
        if no existing connection is found then it establish the new one."""
        try:
            ##check the connection that exist or not
            if MongodbClient.client is None:
                mongodb_url=MONGODB_URL ##if client is none then establish the connection string

                ##even if mongodb_url/MONGODB_URL is none then perform this step
                if mongodb_url is None:
                    raise Exception(f'Environment Variable for connection string: {MONGODB_URL} is not set.')
                ##then establish a new mongodb client connection
                MongodbClient.client=pymongo.MongoClient(mongodb_url,tlsCAFile=ca)

            ##use shared mongodb client
            ##initialize all instancess based on shared mongodb client
            self.client=MongodbClient.client
            self.database=self.client[Database]
            self.database_name=Database
            logging.info("Mongodb Client connection is succesfully established.")
        except Exception as e:
            raise MyException(e, sys)

    
