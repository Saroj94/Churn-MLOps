import logging
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

##constant for log
LOG_DIR="logs"
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE=5*1024*1024
BACKUP_COUNT=3

##Construct the log file path
log_dir_path=os.path.join(from_root(),LOG_DIR) #choose file path dir in root folder and create Log folder
os.makedirs(log_dir_path,exist_ok=True) ##pass file path dir and create the log folder
log_file_path=os.path.join(log_dir_path,LOG_FILE) ##pass the dir path and create log file with LOG_FILE name

def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler.
    """
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Define formatter
    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    # File handler with rotation
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Configure the logger
configure_logger()
