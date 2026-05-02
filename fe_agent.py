from typing import Any
from langchain.agents import create_agent
from models import CleaningResult, FeatureEngineeringResult
from prompts import cook_feature_engineering_prompt
from utils import invoke_agent_with_retries

async def run_feature_engineering_agent(
    model: Any,
    tools: list[Any],
    cleaning_result: CleaningResult,
    dataset_path: str,
    working_directory: str,
    output_dataset_path: str,
    notebook_path: str,
) -> FeatureEngineeringResult:
    instructions = cook_feature_engineering_prompt(
        cleaning_result.description,
        dataset_path,
        working_directory,
        output_dataset_path,
        notebook_path,
    )
    agent = create_agent(model, tools, system_prompt=instructions)
    return await invoke_agent_with_retries(
        name="Feature Engineering",
        agent=agent,
        prompt="Execute the necessary python code to engineer features and return the JSON result.",
        retries=3,
        result_model=FeatureEngineeringResult,
    )
