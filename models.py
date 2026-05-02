from pydantic import BaseModel

class EDAResult(BaseModel):
    summary: str
    problems: list[str]

class CleaningResult(BaseModel):
    fixedproblem: str
    csv: str
    description: str

class FeatureEngineeringResult(BaseModel):
    newfeatures: list[str]
    rationale: str
    new_file_path: str

class ModelingResult(BaseModel):
    task: str
    target: str
    models: list[str]
    metrics: list[str]

class TrainingResult(BaseModel):
    model_name: str
    metrics: dict[str, float]
    saved_model_path: str

class ReportResult(BaseModel):
    executive_summary: str
    data_quality_notes: str
    model_comparison: str
    recommendations: str