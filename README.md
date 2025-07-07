# Churn-MLOps
End-to-End implementation of project

Completed the first phase Data Ingestion in productionizing my ML model.

✅ Built a reusable pipeline as follows:
 - src/logger/__init__.py
 - src/exception/__init__.py
 - src/configuration/mongo_db_connection.py
 - src/constants/__init__.py
 - src/data_access/Fetch_Data.py
 - src/entity/config_entity.py
 - src/entity/artifact_entity.py
 - src/components/data_ingestion.py

✅ Outputs of the pipeline:
- Artifact/Data_ingestion/Raw_Data
 Load raw customer data
- Artifact/Data_ingestion/Ingested
 Split it into train/test sets
- Artifact/Data_ingestion
 Store structured outputs in organized directories
