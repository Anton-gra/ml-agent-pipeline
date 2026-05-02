import re
import os
import json
from pathlib import Path
from typing import Any, TypeVar

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, ValidationError, Field


class ToolLoggingHandler(BaseCallbackHandler):
    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs
    ) -> None:
        tool_name = serialized.get("name", "unknown tool")
        print(f"[tool:start] {tool_name} with parameters {str(input_str)[:200]} ...")

    def on_tool_end(self, output: Any, **kwargs) -> None:
        print(f"[tool:end] {str(output)[:200]} ...")

    def on_tool_error(self, error: BaseException, **kwargs) -> None:
        print(f"[tool:error] {error}")


class PipelineState(BaseModel):
    working_directory: str = ""
    dataset_path: str = ""
    eda_summary: str = ""
    current_dataset: str = ""
    identified_problems: list[str] = Field(default_factory=list)
    cleaning_report: list[str] = Field(default_factory=list)
    feature_engineering_report: str = ""
    modeling_strategy: str = ""
    training_results: str = ""
    final_report: str = ""


ModelT = TypeVar("ModelT", bound=BaseModel)


def safe_print(text: str, limit: int = 200) -> str:
    return str(text)[:limit]


def extract_final_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])

    for message in reversed(messages):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content

    return ""


def extract_json_result(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in model output.")

    json_str = match.group(0)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        print(f"[warning] JSON parse failed. Raw string:\n{json_str}")
        raise ValueError("Failed to parse JSON from model output.") from exc


async def invoke_agent_with_retries(
    name: str,
    agent: Any,
    prompt: str,
    retries: int = 3,
    result_model: type[ModelT] | None = None,
) -> dict[str, Any] | ModelT:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            print(f"\n=== Running Agent: {name} (attempt {attempt}/{retries}) ===")

            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                },
                config={"callbacks": [ToolLoggingHandler()]},
            )

            final_text = extract_final_text(result)
            print(f"[agent:{name}] raw output:\n{final_text}\n")

            json_result = extract_json_result(final_text)

            if result_model is None:
                return json_result

            return result_model(**json_result)

        except (ValueError, ValidationError) as exc:
            last_error = exc
            print(f"[agent:{name}:parse_error] {type(exc).__name__}: {exc}")

        except Exception as exc:
            last_error = exc
            print(f"[agent:{name}:error] {type(exc).__name__}: {exc}")

    raise RuntimeError(f"Agent {name} failed after {retries} retries.") from last_error


def prompt_for_dataset_directory() -> Path:
    while True:
        raw_value = input("Enter dataset directory path: ").strip()

        if not raw_value:
            print("[input:error] Dataset directory path cannot be empty.")
            continue

        candidate = Path(raw_value).expanduser().resolve()

        if not candidate.exists():
            print(f"[input:error] Directory does not exist: {candidate}")
            continue

        if not candidate.is_dir():
            print(f"[input:error] Path is not a directory: {candidate}")
            continue

        return candidate


def resolve_dataset_path(working_directory: Path) -> Path:
    csv_files = sorted(
        path
        for path in working_directory.glob("*.csv")
        if path.name not in {
            "cleaned_data.csv",
            "engineered_data.csv",
            "best_metrics.json",
        }
        and not path.name.startswith("cleaned_")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No source CSV dataset found in directory: {working_directory}"
        )

    if len(csv_files) == 1:
        return csv_files[0]

    print("[input] Multiple CSV files found:")

    for index, csv_path in enumerate(csv_files, start=1):
        print(f"  {index}. {csv_path.name}")

    while True:
        raw_value = input("Enter dataset file name from this directory: ").strip()

        if not raw_value:
            print("[input:error] Dataset file name cannot be empty.")
            continue

        candidate = (working_directory / raw_value).resolve()

        if candidate in csv_files:
            return candidate

        print(f"[input:error] File is not one of the detected CSV files: {raw_value}")


def build_mcp_config() -> dict[str, Any]:
    return {
        "jupyter": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["jupyter-mcp-server==1.0.0"],
            "env": {
                "PROVIDER": "jupyter",
                "JUPYTERLAB": "true",
                "RUNTIME_URL": os.getenv("JUPYTER_URL"),
                "RUNTIME_TOKEN": os.getenv("JUPYTER_TOKEN"),
                "ALLOW_IMG_OUTPUT": "true",
            },
        }
    }


def create_model() -> ChatOpenRouter:
    load_dotenv()

    token = os.getenv("OPENROUTER_API_KEY")

    if not token:
        raise ValueError("OPENROUTER_API_KEY is not set in environment variables.")

    return ChatOpenRouter(
        model_name="google/gemma-4-26b-a4b-it",
        api_key=token,
        temperature=0,
    )