def cook_eda_prompt(dataset_path: str, working_directory: str, notebook_path: str) -> str:
    return f'''
    You are a data analyst performing Exploratory Data Analysis (EDA) on a dataset located at "{dataset_path}".
    Your goal is to examine the dataset, identify key patterns, understand the domain, and detect data quality issues.
    Context:
    - Work is done in Python (pandas, numpy).
    - Do NOT modify the original dataset.
    - Work ONLY inside this directory: "{working_directory}".
    - Any notebook or artifact you create must stay inside this directory.
    - PATH RULE: Use ONLY the exact paths given in this prompt.
    - PATH RULE: Do NOT convert them to absolute filesystem paths.
    - PATH RULE: Do NOT invent new base directories.
    - PATH RULE: For use_notebook, use the notebook path exactly as given: "{notebook_path}".
    - PATH RULE: For file operations inside notebook code, use local file names relative to that notebook directory.
    - PATH RULE: Example: if notebook path is "agent1/eda.ipynb", then load dataset as "{dataset_path}", not "agent1/{dataset_path}".
    Responsibilities:
    0. Start from use_notebook to create or open notebook "{notebook_path}" and insert code cells with insert_execute_code_cell.
    1. Inspect dataset structure (shape, columns, types).
    2. Identify the likely target variable based on the columns.
    3. Identify missing values and duplicates.
    4. Detect data quality issues.
    5. Focus only on meaningful and actionable insights.

    Output format (STRICT):
    Return ONLY valid JSON. No explanations, no extra text.
    {{
        "summary": "string",
        "problems": ["string", "string"]
    }}
    '''

def cook_cleaning_prompt(
    dataset_path: str,
    eda_summary: str,
    problem_to_fix: str,
    working_directory: str,
    output_csv_path: str,
) -> str:
    return f'''
    You are a data cleaning agent working on a dataset located at "{dataset_path}".
    Previous EDA Summary: {eda_summary}
    Your specific task: Fix the following issue: "{problem_to_fix}"
    - Work ONLY inside this directory: "{working_directory}".
    - Any notebook or artifact you create must stay inside this directory.
    - PATH RULE: Use ONLY the exact paths given in this prompt.
    - PATH RULE: Do NOT convert them to absolute filesystem paths.
    - PATH RULE: Do NOT save files outside "{working_directory}".
    - PATH RULE: For use_notebook, use the notebook path exactly as given in the prompt.
    - PATH RULE: Inside notebook code, use "{dataset_path}" and "{output_csv_path}" as local file names, without prepending "{working_directory}/".
    
    Responsibilities:
    1. Load the dataset.
    2. Write and execute pandas code to fix ONLY this specific issue.
    3. Save the cleaned dataset to "{output_csv_path}".

    Output format (STRICT JSON):
    Return ONLY valid JSON. No explanations, no extra text.
    {{
        "action": "string",
        "rows": 0,
        "newfile": "{output_csv_path}"
    }}
    '''

def cook_feature_engineering_prompt(
    cleaning_report: str,
    dataset_path: str,
    working_directory: str,
    output_dataset_path: str,
    notebook_path: str,
) -> str:
    return f'''
    You are a Data Scientist focusing on Feature Engineering.
    Previous step (Data Cleaning) report: "{cleaning_report}"
    Source dataset: "{dataset_path}"
    Notebook: "{notebook_path}"
    
    Your task is to enrich the dataset "{dataset_path}" by creating AT LEAST 2 new, meaningful features.
    Since we are predicting retail turnover, focus on features that make sense for sales, time-series, or store performance.
    - Work ONLY inside this directory: "{working_directory}".
    - Any notebook or artifact you create must stay inside this directory.
    - PATH RULE: Use ONLY the exact paths given in this prompt.
    - PATH RULE: Do NOT convert them to absolute filesystem paths.
    - PATH RULE: Use "{notebook_path}" exactly for use_notebook.
    - PATH RULE: Inside notebook code, use "{dataset_path}" and "{output_dataset_path}" exactly as local file names.
    
    Responsibilities:
    1. Load "{dataset_path}".
    2. Write and execute pandas code to create at least 2 new features based on existing columns.
    3. Save the enriched dataset to "{output_dataset_path}".
    4. Use use_notebook before notebook operations.
    5. Use "{notebook_path}" for this step.
    
    Output format (STRICT JSON):
    Return ONLY valid JSON. No explanations, no extra text.
    {{
        "newfeatures": ["string", "string"],
        "rationale": "string",
        "new_file_path": "{output_dataset_path}"
    }}
    '''

def cook_modeling_prompt(eda_summary: str, feature_engineering_report: str, working_directory: str) -> str:
    return f'''
    You are an ML architect. 
    Context from EDA: "{eda_summary}"
    Feature Engineering report: "{feature_engineering_report}"
    - Work ONLY inside this directory: "{working_directory}".
    - PATH RULE: Any dataset or artifact path mentioned in prior context refers to files inside "{working_directory}".
    - PATH RULE: Do NOT convert relative paths to absolute paths.
    
    Propose a modeling strategy for predicting retail turnover based on the EDA and newly created features.

    Responsibilities:
    1. Extract the target variable from the EDA summary.
    2. Determine the exact task type (Regression or Classification).
    3. Select 3 suitable models for the task.
    4. Propose standard evaluation metrics for this specific task type.

    Output format (STRICT JSON):
    Return ONLY valid JSON. No explanations, no extra text.
    {{
        "task": "string",
        "target": "string",
        "models": ["string", "string", "string"],
        "metrics": ["string", "string", "string"]
    }}
    '''

def cook_training_prompt(
    target_variable: str,
    selected_models: list[str],
    dataset_path: str,
    working_directory: str,
    model_path: str,
    metrics_path: str,
    notebook_path: str,
) -> str:
    models_str = ", ".join(selected_models)
    p = f'''
    You are a Senior ML Engineer. Your task is to train and evaluate: {models_str} to predict "{target_variable}".

    Work ONLY inside this directory: "{working_directory}".
    Any notebook or artifact you create must stay inside this directory.
    PATH RULE: Use ONLY the exact paths given in this prompt.
    PATH RULE: Do NOT convert them to absolute filesystem paths.
    PATH RULE: Do NOT save anything outside "{working_directory}".
    PATH RULE: Use "{notebook_path}" exactly for use_notebook.
    PATH RULE: Inside notebook code, use "{dataset_path}", "{model_path}", and "{metrics_path}" exactly as local file names.

    Use notebook "{notebook_path}" for this step.
    Before any notebook operation, you MUST call use_notebook.

    IMPORTANT CONTEXT:
    Previous pipeline steps such as data cleaning, problem fixing, and feature transformation may already be completed.
    Do NOT redo the full EDA or rebuild the entire preprocessing pipeline unless it is strictly necessary for model training.
    Your focus is model training, evaluation, comparison, and saving the best model.

    STRICT EXECUTION STYLE:
    Do NOT solve the whole task in one large notebook cell.
    Work step by step using multiple notebook cells.
    Each model MUST be trained in a separate notebook cell.
    If an error occurs, debug it step by step in additional cells instead of rewriting everything in one huge cell.

    STRICT RULES TO PREVENT LEAKAGE:
    1. Load "{dataset_path}".
    2. DATA LEAKAGE WARNING: You MUST drop "new_id" and any index-like columns before training.
    The model must learn from features, not memorize store IDs.
    3. TEMPORAL SPLIT: Use a chronological split, for example months 1-8 for training and 9-10 for testing.
    No random shuffling.
    4. Fit any training-time transformations only on the training data.
    5. Transform the test data using objects fitted on the training data only.

    REQUIRED NOTEBOOK FLOW:
    1. Cell 1: imports, paths, and configuration.
    2. Cell 2: load "{dataset_path}" and briefly inspect shape/columns.
    3. Cell 3: prepare X/y and drop leakage/index-like columns.
    4. Cell 4: create chronological train/test split.
    5. Cell 5: define shared evaluation helpers and metrics.
    6. Separate cells after that:
    - one cell per model from {models_str};
    - train the model;
    - calculate MAE, RMSE, R-squared, and MAPE;
    - store results in a common results list/table.
    7. Final comparison cell:
    - compare all trained models;
    - select the best model by highest R-squared.
    8. Persistence cell:
    - compare current best R-squared with "{metrics_path}" if it exists;
    - save the best model to "{model_path}" using joblib only if the new model is better;
    - update/save metrics if the model is saved.
    9. Final output cell:
    - return the final result as STRICT JSON.

    MODEL TRAINING TASKS:
    1. Train only the requested models: {models_str}.
    2. Use the already prepared dataset from "{dataset_path}" as the main source.
    3. Add only the minimal additional preprocessing needed to make the selected models trainable.
    4. Calculate:
    - MAE
    - RMSE
    - R-squared
    - MAPE
    5. Select the best model by highest R-squared.

    LONG-TERM MEMORY:
    - Save the best model to "{model_path}" using joblib.
    - Compare current R-squared with "{metrics_path}" if it exists.
    - Only overwrite "{model_path}" if the new model is better.
    - If the new model is not better, keep the existing model unchanged.

    Final answer must be STRICT JSON only:
    '''
    p += '''
    {
    "model_name": "string",
    "metrics": {
        "MAE": 0.0,
        "RMSE": 0.0,
        "R2": 0.0,
        "MAPE": 0.0
    },
    "saved_model_path": "{model_path}"
    }
    '''
    return p

def cook_reporting_prompt(
    eda_summary: str,
    feature_engineering_report: str,
    modeling_strategy: str,
    training_results: str,
    working_directory: str,
) -> str:
    return f'''
    You are a Lead Data Scientist forming the final report.
    Here is the pipeline context:
    - EDA Summary: {eda_summary}
    - Feature Engineering Action: {feature_engineering_report}
    - Modeling Strategy: {modeling_strategy}
    - Training Results: {training_results}
    - Work ONLY inside this directory: "{working_directory}".
    - PATH RULE: Any referenced dataset files or artifacts must be interpreted as files belonging to "{working_directory}".
    - PATH RULE: Do NOT rewrite them as absolute paths.
    
    Responsibilities and STRICT RULES:
    1. Analyze the context and write a comprehensive final JSON report.
    2. BE HONEST: If the training agent compared individual models and selected the best one, describe exactly that. DO NOT hallucinate and DO NOT call it an "ensemble approach" or "ensemble model" unless an actual ensembling technique (like Voting or Stacking) was explicitly used in the training results.
    3. Include the MAPE metric in your business interpretation to explain the error in percentage terms relative to the average target variable.

    Output format (STRICT JSON):
    Return ONLY valid JSON. No explanations, no extra text.
    {{
        "executive_summary": "string",
        "data_quality_notes": "string",
        "model_comparison": "string",
        "recommendations": "string"
    }}
    '''


def cook_cleaning_prompt1(
    dataset_path: str,
    working_directory: str,
    problem_to_fix: str,
    output_csv: str = "train_cleaned.csv",
    notebook_name: str = "eda.ipynb",
) -> str:
    return f"""
You are a data cleaning agent.
Dataset: "{dataset_path}"
Notebook: "{notebook_name}"
Working directory: "{working_directory}"
Fix exactly one problem: "{problem_to_fix}"

Rules:
- Load ONLY "{dataset_path}".
- Fix ONLY the provided problem.
- Save to NEW CSV: "{output_csv}".
- Work ONLY inside "{working_directory}".
- PATH RULE: Use ONLY the exact paths from this prompt.
- PATH RULE: Do NOT convert them to absolute filesystem paths.
- PATH RULE: Do NOT create files outside "{working_directory}".
- PATH RULE: Use "{notebook_name}" exactly for use_notebook.
- PATH RULE: Inside notebook code, use "{dataset_path}" and "{output_csv}" exactly as local file names, without prepending "{working_directory}/".
- Never overwrite source dataset.
- Use use_notebook before notebook operations.
- Use insert_execute_code_cell with cell_index=-1.
- Return ONLY valid JSON.

JSON schema:
{{
  "fixedproblem": "string",
  "csv": "{output_csv}",
  "description": "string"
}}
"""
