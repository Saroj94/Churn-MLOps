from dataclasses import dataclass

##DATA INGESTION ARTIFACTS
"What output/artifact we can expect from a data ingestion pipeline"

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

##DATA VALIDATION ARTIFACT
@dataclass
class DataValidationArtifact:
    validation_report_file_path: str
    validation_message: str
    validation_status:bool
