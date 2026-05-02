from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
import asyncio
import json
from pathlib import Path
from typing import Any
from models import ModelingResult

from prompts import (
    cook_eda_prompt,
    cook_cleaning_prompt1,
    cook_training_prompt,
    cook_reporting_prompt,
)
from utils import (
    PipelineState,
    build_mcp_config,
    create_model,
    invoke_agent_with_retries,
    prompt_for_dataset_directory,
    resolve_dataset_path,
)
from models import CleaningResult, EDAResult, ReportResult, TrainingResult
from fe_agent import run_feature_engineering_agent



EXECUTION_TOOL_ALLOWLIST = {
    "use_notebook",
    "read_notebook",
    "read_cell",
    "execute_cell",
    "insert_execute_code_cell",
}

READONLY_TOOL_ALLOWLIST = {
    "use_notebook",
    "read_notebook",
    "read_cell",
}

def join_path(base: str, name: str) -> str:
    return str(Path(base) / name)


def filename(path: str) -> str:
    return Path(path).name


def to_runtime_path(path: str) -> str:
    candidate = Path(path).resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(candidate.relative_to(cwd))
    except ValueError:
        return str(candidate)


def print_available_tools(tools: list[Any]) -> None:
    print("[tools] available tools:")
    for tool in tools:
        print(f"  - {tool.name}")


def filter_tools(tools: list[Any], allowed_names: set[str]) -> list[Any]:
    available_names = set()
    filtered_tools = []

    for tool in tools:
        if tool.name in allowed_names:
            filtered_tools.append(tool)
        available_names.add(tool.name)

    missing_names = sorted(allowed_names - available_names)
    if missing_names:
        raise ValueError(f"Configured MCP tools not found: {', '.join(missing_names)}")

    return filtered_tools


async def run_agent(
    name: str,
    prompt: str,
    model: Any,
    tools: list[Any],
    allowed_tools: set[str],
    result_model: type[Any] | None = None,
) -> Any:
    agent_tools = filter_tools(tools, allowed_tools)
    agent = create_agent(model, agent_tools)

    return await invoke_agent_with_retries(
        name=name,
        agent=agent,
        prompt=prompt,
        retries=3,
        result_model=result_model,
    )


async def run_pipeline():
    working_directory = prompt_for_dataset_directory()
    dataset_path = resolve_dataset_path(working_directory)

    state = PipelineState(
        working_directory=str(working_directory),
        dataset_path=str(dataset_path),
        current_dataset=str(dataset_path),
    )

    model = create_model()
    client = MultiServerMCPClient(build_mcp_config())

    eda_notebook = "eda.ipynb"
    fe_notebook = "feature_engineering.ipynb"
    training_notebook = "training.ipynb"

    engineered_dataset_path = join_path(state.working_directory, "engineered_data.csv")
    best_model_path = join_path(state.working_directory, "best_model.pkl")
    best_metrics_path = join_path(state.working_directory, "best_metrics.json")

    runtime_workdir = to_runtime_path(state.working_directory)

    runtime_eda_notebook = join_path(runtime_workdir, eda_notebook)
    runtime_fe_notebook = join_path(runtime_workdir, fe_notebook)
    runtime_training_notebook = join_path(runtime_workdir, training_notebook)

    local_dataset = filename(state.dataset_path)
    local_current_dataset = filename(state.current_dataset)
    local_engineered_dataset = filename(engineered_dataset_path)
    local_best_model = filename(best_model_path)
    local_best_metrics = filename(best_metrics_path)

    print(f"[pipeline] working directory: {state.working_directory}")
    print(f"[pipeline] source dataset: {state.dataset_path}")

    print("[mcp] opening jupyter session...")

    try:
        async with client.session("jupyter") as session:
            print("[mcp] session initialized")

            tools = await load_mcp_tools(session)
            print_available_tools(tools)

            # --- 1. EDA ---
            eda_prompt = cook_eda_prompt(
                local_dataset,
                runtime_workdir,
                runtime_eda_notebook,
            )

            eda_result = await run_agent(
                "EDA",
                eda_prompt,
                model,
                tools,
                EXECUTION_TOOL_ALLOWLIST,
                result_model=EDAResult,
            )

            state.eda_summary = eda_result.summary
            state.identified_problems = eda_result.problems

            # --- 2. Cleaning ---
            if state.identified_problems:
                for i, problem in enumerate(state.identified_problems):
                    output_csv_path = join_path(
                        state.working_directory, f"cleaned_{i}.csv"
                    )
                    runtime_output_csv = to_runtime_path(output_csv_path)

                    notebook_name = f"cleaning_{i}.ipynb"
                    runtime_notebook = join_path(runtime_workdir, notebook_name)

                    local_output_csv = filename(output_csv_path)

                    print(f"[pipeline] Fixing problem: {problem}")

                    cleaning_prompt = cook_cleaning_prompt1(
                        dataset_path=local_current_dataset,
                        working_directory=runtime_workdir,
                        problem_to_fix=problem,
                        output_csv=local_output_csv,
                        notebook_name=runtime_notebook,
                    )

                    cleaning_result = await run_agent(
                        "Cleaning",
                        cleaning_prompt,
                        model,
                        tools,
                        EXECUTION_TOOL_ALLOWLIST,
                        result_model=CleaningResult,
                    )

                    state.cleaning_report.append(
                        f"Problem: {problem} | Descriptoin: {cleaning_result.description}"
                    )

                    state.current_dataset = runtime_output_csv
                    local_current_dataset = cleaning_result.csv

            else:
                print("[pipeline] No problems found. Skipping cleaning.")

                cleaning_result = CleaningResult(
                    fixedproblem="",
                    csv=local_dataset,
                    description="No action taken.",
                )

                state.cleaning_report.append(cleaning_result.description)

            # --- 2.5 Feature Engineering ---
            fe_result = await run_feature_engineering_agent(
                model=model,
                tools=tools,
                cleaning_result=cleaning_result,
                dataset_path=local_current_dataset,
                working_directory=runtime_workdir,
                output_dataset_path=local_engineered_dataset,
                notebook_path=runtime_fe_notebook,
            )

            state.feature_engineering_report = (
                f"Created: {fe_result.newfeatures}. "
                f"Rationale: {fe_result.rationale}"
            )

            # --- 3. Modeling ---
            modeling_result = ModelingResult(
                task="regression",
                target="turnover",  # ← важно: совпадает с prompt логикой
                models=["RandomForest", "XGBoost", "LinearRegression"],
                metrics=["MAE", "RMSE", "R2"]
            )

            # --- 4. Training ---
            training_prompt = cook_training_prompt(
                modeling_result.target,
                modeling_result.models,
                local_engineered_dataset,
                runtime_workdir,
                local_best_model,
                local_best_metrics,
                runtime_training_notebook,
            )

            training_result = await run_agent(
                "Training",
                training_prompt,
                model,
                tools,
                EXECUTION_TOOL_ALLOWLIST,
                result_model=TrainingResult,
            )

            state.training_results = str(training_result.metrics)

            # --- 5. Reporting ---
            reporting_prompt = cook_reporting_prompt(
                state.eda_summary,
                state.feature_engineering_report,
                state.modeling_strategy,
                state.training_results,
                runtime_workdir,
            )

            report_result = await run_agent(
                "Reporting",
                reporting_prompt,
                model,
                tools,
                READONLY_TOOL_ALLOWLIST,
                result_model=ReportResult,
            )

            state.final_report = json.dumps(
                report_result.model_dump(),
                indent=2,
                ensure_ascii=False,
            )

            print("\n==================================")
            print("PIPELINE COMPLETED. FINAL REPORT:")
            print("==================================")
            print(state.final_report)

    except Exception as exc:
        print(f"[mcp:error] {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    asyncio.run(run_pipeline())