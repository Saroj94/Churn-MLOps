import os
from pathlib import Path

project_source_name='src'

Files_Folder_list=[
    f"{project_source_name}/__init__.py",
    f"{project_source_name}/components/__init__.py",
    f"{project_source_name}/components/data_ingestion.py",  
    f"{project_source_name}/components/data_validation.py",
    f"{project_source_name}/components/data_transformation.py",
    f"{project_source_name}/components/model_trainer.py",
    f"{project_source_name}/components/model_evaluation.py",
    f"{project_source_name}/components/model_pusher.py",
    f"{project_source_name}/configuration/__init__.py",
    f"{project_source_name}/configuration/mongo_db_connection.py",
    f"{project_source_name}/configuration/aws_connection.py",
    f"{project_source_name}/cloud_storage/__init__.py",
    f"{project_source_name}/cloud_storage/aws_storage.py",
    f"{project_source_name}/data_access/__init__.py",
    f"{project_source_name}/data_access/churn_data.py",
    f"{project_source_name}/constants/__init__.py",
    f"{project_source_name}/entity/__init__.py",
    f"{project_source_name}/entity/config_entity.py",
    f"{project_source_name}/entity/artifact_entity.py",
    f"{project_source_name}/entity/estimator.py",
    f"{project_source_name}/entity/s3_estimator.py",
    f"{project_source_name}/exception/__init__.py",
    f"{project_source_name}/logger/__init__.py",
    f"{project_source_name}/pipline/__init__.py",
    f"{project_source_name}/pipline/training_pipeline.py",
    f"{project_source_name}/pipline/prediction_pipeline.py",
    f"{project_source_name}/utils/__init__.py",
    f"{project_source_name}/utils/main_utils.py",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    "pyproject.toml",
    "config/model.yaml",
    "config/schema.yaml"]


for filepath in Files_Folder_list:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")