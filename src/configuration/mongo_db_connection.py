import os
import sys
import pymongo
import certifi
from src.exception import MyException
from src.logger import logging
from src.constants import DATABASE_NAME
from dotenv import load_dotenv

# Load environment variables from .env (for local development)
load_dotenv()

# Get CA certificate
ca = certifi.where()

class MongodbClient:
    """Responsible for establishing MongoDB client connection."""
    
    client = None

    def __init__(self, Database: str = DATABASE_NAME) -> None:
        try:
            if MongodbClient.client is None:
                # ✅ Securely fetch MongoDB URL at runtime
                mongodb_url = os.getenv("MONGODB_URL")

                if not mongodb_url:
                    raise Exception("Environment Variable for connection string: MONGODB_URL is not set.")

                MongodbClient.client = pymongo.MongoClient(mongodb_url, tlsCAFile=ca)

            self.client = MongodbClient.client
            self.database = self.client[Database]
            self.database_name = Database

            logging.info("✅ MongoDB client connection established successfully.")

        except Exception as e:
            raise MyException(e, sys)