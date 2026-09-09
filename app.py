from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union, Iterable, Sequence
import gradio as gr
# local imports
import settings
from query.querier import Querier
import logging


logger = logging.getLogger(__name__)

DEFAULT_SYNTHESIS_TEMPLATE = (
    "Synthesize the following document-specific answers into one consolidated answer. "
    "Focus on agreements, key differences, and confidence-limiting gaps.\n\n"
    "Question:\n{question}\n\n"
    "Document answers:\n{answer_string}"
)


@lru_cache(maxsize=1)
def _get_utils_module() -> Any:
    """
    Import and cache the local utils module.

    Returns
    -------
    Any
        imported `utils` module.
    """
    import utils as ut

    return ut


class _MissingOpenAIAPIError(Exception):
    pass


@lru_cache(maxsize=1)
def _get_openai_api_error_type() -> type[BaseException]:
    """
    Resolve the OpenAI API error class if available.

    Returns
    -------
    type[BaseException]
        OpenAI APIError type when import succeeds, otherwise a local fallback type.
    """
    try:
        from openai import APIError as OpenAIAPIError

        return OpenAIAPIError
    except Exception:
        return _MissingOpenAIAPIError

REVIEW_COLUMNS = [
    "question",
    "task",
    "instruction",
    "classes",
]
ANALYSIS_DEFAULT_HEADERS = ["col_1", "col_2", "col_3"]

ROW_NUMBER_COLUMN = "id"
TABLE_COLUMNS = [ROW_NUMBER_COLUMN, *REVIEW_COLUMNS]
REVIEW_COLUMN_WIDTHS: List[str | int] = ["3%", "24%", "8%", "42%", "23%"]

TASK_OPTIONS = ("answer", "classify", "synthesis")
CLASSES_VALUE_SEPARATOR = "\n"
KEEPALIVE_INTERVAL_MS = 45_000

KEEPALIVE_HEAD = f"""
<script>
(() => {{
    const intervalMs = {KEEPALIVE_INTERVAL_MS};
    const keepAlive = () => {{
        fetch(window.location.pathname, {{ method: "HEAD", cache: "no-store" }}).catch(() => {{}});
    }};

    setInterval(keepAlive, intervalMs);
    document.addEventListener("visibilitychange", () => {{
        if (document.visibilityState === "visible") {{
            keepAlive();
        }}
    }});
}})();
</script>
"""

# For styling purposes of the DataFrame components
TABLE_CSS = """
#questions-table table {
    table-layout: fixed;
    width: 100%;
}

#questions-table table,
#questions-table table * {
    font-size: 0.8rem !important;
}

#add-row-btn {
    background-color: #0b5d1e !important;
    border-color: #0b5d1e !important;
    color: #ffffff !important;
}

#add-row-btn:hover {
    background-color: #094a18 !important;
    border-color: #094a18 !important;
}

#go-btn,
#go-btn button {
    width: 420px !important;
    min-width: 420px !important;
    max-width: 420px !important;
    background-color: #0b5d1e !important;
    border-color: #0b5d1e !important;
    color: #ffffff !important;
}

#go-btn:hover,
#go-btn button:hover {
    background-color: #094a18 !important;
    border-color: #094a18 !important;
}

#questions-table {
    --cell-line-height: 1.25rem;
    --cell-max-lines: 12;
    font-size: 0.8rem;
}

#questions-table th,
#questions-table td,
#analysis-output-table th,
#analysis-output-table td {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere;
    word-break: break-word;
    vertical-align: top !important;
    font-size: 0.8rem !important;
}

/* Preserve embedded newlines across all DataFrame cells and editors. */
.gradio-container [data-testid="dataframe"] td,
.gradio-container [data-testid="dataframe"] td *,
.gradio-container [data-testid="dataframe"] td textarea,
.gradio-container [data-testid="dataframe"] td input {
    white-space: pre-wrap !important;
}

#questions-table td {
    line-height: var(--cell-line-height);
    max-height: calc(var(--cell-line-height) * var(--cell-max-lines));
    overflow-y: auto;
}

#questions-table td textarea,
#questions-table td input {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere;
    word-break: break-word;
    line-height: var(--cell-line-height);
    max-height: calc(var(--cell-line-height) * var(--cell-max-lines));
    overflow-y: auto;
    font-size: 0.8rem !important;
}

/* Gradio renders different nested elements after data is loaded; enforce cell typography there too. */
#questions-table tbody td *,
#questions-table .cell-wrap *,
#questions-table .wrap * {
    font-size: 0.8rem !important;
}

#questions-table .cell-wrap,
#questions-table .wrap,
#analysis-output-table .cell-wrap,
#analysis-output-table .wrap {
    align-items: flex-start !important;
}

/* Classes column (5th column including row number) should show one list element per line. */
#questions-table tbody td:nth-child(5),
#questions-table tbody td:nth-child(5) * {
    white-space: pre-line !important;
}

/* Classes input: show each selected class on its own line regardless of dropdown width. */
#classes-input [data-testid="dropdown"] .wrap,
#classes-input [data-testid="dropdown"] .secondary-wrap,
#classes-input [data-testid="dropdown"] .tokens {
    display: block !important;
}

#classes-input [data-testid="dropdown"] .token,
#classes-input [data-testid="dropdown"] [data-testid="dropdown-token"] {
    display: block !important;
    width: 100% !important;
    margin: 0 0 4px 0 !important;
    white-space: normal !important;
}

/* Also allow long option labels in the dropdown menu to wrap instead of truncating. */
#classes-input [role="option"] {
    white-space: normal !important;
    line-height: 1.25 !important;
}

"""

def list_document_paths(
    documents_root: Union[Path, str],
    *,
    valid_extensions: Optional[Iterable[str]] = None,
) -> List[Path]:
    """
    Recursively list supported document files under a root directory.

    Parameters
    ----------
    documents_root : Union[Path, str]
        root folder that contains documents.
    valid_extensions : Optional[Iterable[str]]
        allowed file extensions; when omitted defaults to configured valid extensions.

    Returns
    -------
    List[Path]
        sorted list of matching file paths.
    """
    root = Path(documents_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    ut = _get_utils_module()
    allowed = {ext.lower() for ext in (valid_extensions or ut.VALID_EXTENSIONS)}
    # project_root = root / settings.project.artifacts_dirname
    paths: List[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # if project_root in path.parents:
        #     continue
        if path.suffix.lower() not in allowed:
            continue
        paths.append(path)

    return sorted(paths, key=lambda item: str(item).lower())


def _list_document_paths_direct(documents_root: Path) -> List[Path]:
    """
    List supported document files that are direct children of a folder.

    Parameters
    ----------
    documents_root : Path
        folder to scan.

    Returns
    -------
    List[Path]
        supported files whose parent is exactly `documents_root`.
    """
    return [path for path in list_document_paths(documents_root) if path.parent == documents_root]


def _ingestion_payload(
    status: str,
    active_folder_value: str,
) -> Tuple[str, str]:
    """
    Build a standard status payload returned to ingestion UI outputs.

    Parameters
    ----------
    status : str
        status message text.
    active_folder_value : str
        currently active folder path.

    Returns
    -------
    Tuple[str, str]
        tuple of status text and active folder value.
    """
    return status, active_folder_value


def _questions_json_path(folder_path: str) -> Path:
    """
    Compute the canonical path to `questions.json` for a document folder.

    Parameters
    ----------
    folder_path : str
        document folder path.

    Returns
    -------
    Path
        resolved path to `review/questions.json`.
    """
    return Path(folder_path).expanduser().resolve() / "review" / "questions.json"


def _normalize_classification(value: Any) -> bool:
    """
    Convert mixed truthy/falsey inputs to a boolean classification flag.

    Parameters
    ----------
    value : Any
        input value from configuration or persisted question rows.

    Returns
    -------
    bool
        normalized boolean value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower() if value is not None else ""
    return normalized in {"true", "1", "yes", "y", "on"}


def _normalize_task(value: Any) -> str:
    """
    Normalize task values to a supported task identifier.

    Parameters
    ----------
    value : Any
        raw task value.

    Returns
    -------
    str
        one of `answer`, `classify`, or `synthesis`.
    """
    normalized = str(value).strip().lower() if value is not None else ""
    if normalized in TASK_OPTIONS:
        return normalized
    if normalized in {"classification", "class"}:
        return "classify"
    return "answer"


def _task_from_mapping(row: dict[str, Any]) -> str:
    """
    Derive a normalized task value from a question row mapping.

    Parameters
    ----------
    row : dict[str, Any]
        dictionary containing task fields and optional backward-compatible booleans.

    Returns
    -------
    str
        normalized task value.
    """
    explicit_task = _first_present_value(row, "task", "Task")
    if str(explicit_task).strip():
        return _normalize_task(explicit_task)

    # Backward-compatible mapping for old boolean schema.
    if _normalize_classification(_first_present_value(row, "synthesis", "Synthesis")):
        return "synthesis"
    if _normalize_classification(_first_present_value(row, "classification", "Classification")):
        return "classify"
    return "answer"


def _normalize_review_row(values: List[Any]) -> List[Union[str, bool]]:
    """
    Normalize a review row to the canonical questions schema.

    Parameters
    ----------
    values : List[Any]
        row values in review column order.

    Returns
    -------
    List[Union[str, bool]]
        normalized row values `[question, task, instruction, classes]`.
    """
    padded = values[: len(REVIEW_COLUMNS)] + [""] * (len(REVIEW_COLUMNS) - len(values))
    normalized_task = _normalize_task(padded[1])
    classes_value = padded[3]
    if isinstance(classes_value, (list, tuple)):
        normalized_classes = CLASSES_VALUE_SEPARATOR.join(
            str(item).strip() for item in classes_value if str(item).strip()
        )
    else:
        normalized_classes = str(classes_value or "").strip()

    if normalized_task != "classify":
        normalized_classes = ""

    return [
        str(padded[0] or ""),
        normalized_task,
        str(padded[2] or ""),
        normalized_classes,
    ]


def _first_present_value(row: dict[str, Any], *keys: str) -> Any:
    """
    Return the first present non-None value from a mapping by key priority.

    Parameters
    ----------
    row : dict[str, Any]
        source mapping.
    *keys : str
        ordered candidate keys.

    Returns
    -------
    Any
        first non-None value or an empty string when no key matches.
    """
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return ""


def _row_from_mapping(row: dict[str, Any]) -> List[Union[str, bool]]:
    """
    Convert a mapping-based question record into the normalized row format.

    Parameters
    ----------
    row : dict[str, Any]
        source mapping from JSON.

    Returns
    -------
    List[Union[str, bool]]
        normalized question row.
    """
    return _normalize_review_row(
        [
            _first_present_value(row, "question", "Question"),
            _task_from_mapping(row),
            _first_present_value(row, "instruction", "Instruction"),
            _first_present_value(row, "classes", "Classes"),
        ]
    )


def _load_questions_from_json(json_path: Path) -> List[List[Union[str, bool]]]:
    """
    Load and normalize question rows from a JSON file.

    Parameters
    ----------
    json_path : Path
        path to the question list JSON.

    Returns
    -------
    List[List[Union[str, bool]]]
        normalized question rows.
    """
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows: List[List[Union[str, bool]]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                rows.append(_row_from_mapping(item))
            elif isinstance(item, list):
                rows.append(_normalize_review_row(item))
    elif isinstance(payload, dict):
        possible_rows = payload.get("rows") or payload.get("questions")
        if isinstance(possible_rows, list):
            for item in possible_rows:
                if isinstance(item, dict):
                    rows.append(_row_from_mapping(item))

    return rows


def _with_row_numbers(rows: List[List[Union[str, bool]]]) -> List[List[Union[str, bool]]]:
    """
    Prefix each normalized question row with a 1-based row number.

    Parameters
    ----------
    rows : List[List[Union[str, bool]]]
        normalized question rows.

    Returns
    -------
    List[List[Union[str, bool]]]
        rows with row number as first column.
    """
    numbered_rows: List[List[Union[str, bool]]] = []
    for index, row in enumerate(rows, start=1):
        numbered_rows.append([str(index), *row])
    return numbered_rows


def _questions_table_update(value: Any) -> Any:
    """
    Build a Gradio update payload for the question overview table.

    Parameters
    ----------
    value : Any
        table rows to display.

    Returns
    -------
    Any
        Gradio update object for table value and fixed row configuration.
    """
    # Render the table as output-only; editing happens via the custom form.
    row_count = max(len(value), 1) if isinstance(value, list) else None
    return gr.update(value=value, row_count=row_count, interactive=False)


def _normalize_questions_rows(table_data: Any) -> List[List[Union[str, bool]]]:
    """
    Normalize table data to question rows and drop empty rows.

    Parameters
    ----------
    table_data : Any
        table-like data from the UI.

    Returns
    -------
    List[List[Union[str, bool]]]
        normalized non-empty rows.
    """
    return _normalize_questions_rows_with_options(table_data, drop_empty=True)


def _normalize_questions_rows_with_options(
    table_data: Any,
    *,
    drop_empty: bool,
) -> List[List[Union[str, bool]]]:
    """
    Normalize question rows from various table representations.

    Parameters
    ----------
    table_data : Any
        DataFrame-like or list-like row data.
    drop_empty : bool
        whether to drop rows where question/instruction/classes are all empty.

    Returns
    -------
    List[List[Union[str, bool]]]
        normalized question rows.
    """
    if table_data is None:
        return []

    raw_rows: list[Any]
    if hasattr(table_data, "values") and hasattr(table_data.values, "tolist"):
        raw_rows = table_data.values.tolist()
    elif isinstance(table_data, list):
        raw_rows = table_data
    else:
        return []

    normalized_rows: List[List[Union[str, bool]]] = []
    for raw_row in raw_rows:
        if isinstance(raw_row, dict):
            row = _normalize_review_row([raw_row.get(col, "") for col in REVIEW_COLUMNS])
        elif isinstance(raw_row, (list, tuple)):
            cells = list(raw_row)
            if len(cells) >= len(TABLE_COLUMNS):
                cells = cells[1:]
            row = _normalize_review_row(cells)
        else:
            continue

        if not drop_empty or any(str(cell).strip() for cell in [row[0], row[2], row[3]]):
            normalized_rows.append(row)

    return normalized_rows


def _classes_text_from_storage(value: Any) -> str:
    """
    Convert persisted class values into newline-delimited text.

    Parameters
    ----------
    value : Any
        stored classes value.

    Returns
    -------
    str
        newline-delimited classes string.
    """
    if isinstance(value, (list, tuple, set)):
        cleaned_values = [str(item).strip() for item in value if str(item).strip()]
        return CLASSES_VALUE_SEPARATOR.join(cleaned_values)

    raw = str(value or "").strip()
    if not raw:
        return ""

    cleaned_values = [item.strip() for item in raw.split(CLASSES_VALUE_SEPARATOR) if item.strip()]
    return CLASSES_VALUE_SEPARATOR.join(cleaned_values)


def _default_form_values() -> Tuple[str, str, str, str]:
    """
    Return default values for the question row editor form.

    Returns
    -------
    Tuple[str, str, str, str]
        default `(question, task, instruction, classes)` values.
    """
    return "", "answer", "", ""


def _row_selector_update(rows: List[List[Union[str, bool]]], selected_index: Optional[int] = None) -> Any:
    """
    Build a Gradio update for the row selector dropdown.

    Parameters
    ----------
    rows : List[List[Union[str, bool]]]
        normalized question rows.
    selected_index : Optional[int]
        zero-based row index to select.

    Returns
    -------
    Any
        Gradio update object for selector choices and value.
    """
    if not rows:
        return gr.update(choices=[], value=None)

    choices: List[str] = []
    for index, _ in enumerate(rows):
        choices.append(str(index + 1))

    if selected_index is None or selected_index < 0 or selected_index >= len(rows):
        selected_index = 0

    # Dropdown values are 1-based row numbers for a clearer UX.
    return gr.update(choices=choices, value=str(selected_index + 1))


def _selected_row_index(selected_row: Optional[str], rows: List[List[Union[str, bool]]]) -> Optional[int]:
    """
    Convert a selected row number string into a validated zero-based index.

    Parameters
    ----------
    selected_row : Optional[str]
        selected row number from UI.
    rows : List[List[Union[str, bool]]]
        available normalized rows.

    Returns
    -------
    Optional[int]
        zero-based index when valid, otherwise None.
    """
    if selected_row is None:
        return None
    try:
        row_number = int(selected_row)
    except (TypeError, ValueError):
        return None

    index = row_number - 1

    if index < 0 or index >= len(rows):
        return None
    return index


def _form_values_from_row(row: List[Union[str, bool]]) -> Tuple[str, str, str, str]:
    """
    Convert a normalized row into row editor form values.

    Parameters
    ----------
    row : List[Union[str, bool]]
        normalized question row.

    Returns
    -------
    Tuple[str, str, str, str]
        `(question, task, instruction, classes)` suitable for form widgets.
    """
    return (
        str(row[0] or ""),
        _normalize_task(row[1]),
        str(row[2] or ""),
        _classes_text_from_storage(row[3]),
    )


def _classes_input_update(task: str, classes: Any = None) -> Any:
    """
    Update classes input visibility based on the selected task.

    Parameters
    ----------
    task : str
        selected task value.
    classes : Any
        current classes value.

    Returns
    -------
    Any
        Gradio update object for classes input visibility and value.
    """
    normalized_task = _normalize_task(task)
    if normalized_task == "classify":
        value = _classes_text_from_storage(classes)
        return gr.update(visible=True, value=value)
    return gr.update(visible=False, value="")


def _screen_mode_update(mode: str, ingest_folder_path: str) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Toggle ingest and analysis UI sections based on selected screen mode.

    Parameters
    ----------
    mode : str
        selected mode value from the sidebar.
    ingest_folder_path : str
        current ingest folder path from the ingest sidebar input.

    Returns
    -------
    Tuple[Any, Any, Any, Any, Any, Any]
        Gradio updates for ingest sidebar, input form accordion, analyse sidebar,
        analyse main group, question list overview accordion, and GO button.
    """
    is_ingest_mode = (mode or "").strip().lower() == "ingest folder"
    ingest_visibility_update = gr.update(visible=is_ingest_mode)
    analyse_visibility_update = gr.update(visible=not is_ingest_mode)

    if is_ingest_mode:
        (
            question_overview_update,
            input_form_update,
            go_button_update,
        ) = _ingest_main_visibility_update(ingest_folder_path)
    else:
        hidden_main_ingest_update = gr.update(visible=False)
        question_overview_update = hidden_main_ingest_update
        input_form_update = hidden_main_ingest_update
        go_button_update = hidden_main_ingest_update

    return (
        ingest_visibility_update,
        input_form_update,
        analyse_visibility_update,
        analyse_visibility_update,
        question_overview_update,
        go_button_update,
    )


def _ingest_main_visibility_update(folder_path: str) -> Tuple[Any, Any, Any]:
    """
    Show ingest main-screen components only when a valid document folder is submitted.

    Parameters
    ----------
    folder_path : str
        submitted folder path from ingest sidebar.

    Returns
    -------
    Tuple[Any, Any, Any]
        Gradio updates for question overview accordion, input form accordion,
        and GO button visibility.
    """
    is_valid_folder = False
    if folder_path:
        try:
            is_valid_folder = Path(folder_path).expanduser().resolve().is_dir()
        except OSError:
            is_valid_folder = False

    visibility_update = gr.update(visible=is_valid_folder)
    return (
        visibility_update,
        visibility_update,
        visibility_update,
    )


def load_review_output_folders(folder_path: str) -> Any:
    """
    List available review output subfolders for analyse mode.

    Parameters
    ----------
    folder_path : str
        selected document folder path.

    Returns
    -------
    Any
        Gradio update object for output-folder dropdown visibility, choices, and value.
    """
    if not folder_path:
        return gr.update(visible=False, choices=[], value=None)

    try:
        folder = Path(folder_path).expanduser().resolve()
    except OSError:
        return gr.update(visible=False, choices=[], value=None)

    review_folder = folder / "review"
    if not review_folder.is_dir():
        return gr.update(visible=False, choices=[], value=None)

    output_folders = sorted(
        [child.name for child in review_folder.iterdir() if child.is_dir()],
        key=str.lower,
        reverse=True,
    )

    if not output_folders:
        return gr.update(visible=True, choices=[], value=None)

    return gr.update(visible=True, choices=output_folders, value=output_folders[0])


def refresh_review_output_folders_for_mode(mode: str, folder_path: str) -> Any:
    """
    Refresh output-folder dropdown only while analyse mode is active.

    Parameters
    ----------
    mode : str
        selected mode value from the sidebar.
    folder_path : str
        document folder path used to locate the review output directory.

    Returns
    -------
    Any
        Gradio update object for output-folder dropdown visibility, choices, and value.
    """
    is_analyse_mode = (mode or "").strip().lower() == "analyse output"
    if not is_analyse_mode:
        return gr.update(visible=False, choices=[], value=None)
    return load_review_output_folders(folder_path)


def _analysis_result_type_update(selected_output_folder: Optional[str]) -> Any:
    """
    Show or hide analysis result type radio based on output folder selection.

    Parameters
    ----------
    selected_output_folder : Optional[str]
        selected review output folder.

    Returns
    -------
    Any
        Gradio update object for result type radio visibility and value.
    """
    if not selected_output_folder:
        return gr.update(visible=False, value=None)
    return gr.update(visible=True, value="answers")


def _json_payload_to_table(payload: Any) -> Tuple[List[str], List[List[Any]]]:
    """
    Convert a JSON payload into table headers and rows for display.

    Parameters
    ----------
    payload : Any
        JSON-decoded value from an output file.

    Returns
    -------
    Tuple[List[str], List[List[Any]]]
        table headers and row values.
    """
    records: List[dict[str, Any]] = []

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                records.append(item)
            else:
                records.append({"value": item})
    elif isinstance(payload, dict):
        possible_rows = payload.get("rows") or payload.get("questions")
        if isinstance(possible_rows, list):
            for item in possible_rows:
                if isinstance(item, dict):
                    records.append(item)
                else:
                    records.append({"value": item})
        else:
            records.append(payload)
    else:
        records.append({"value": payload})

    headers: List[str] = []
    for record in records:
        for key in record.keys():
            key_str = str(key)
            if key_str not in headers:
                headers.append(key_str)

    rows: List[List[Any]] = []
    for record in records:
        row: List[Any] = []
        for header in headers:
            value = record.get(header, "")
            if isinstance(value, (dict, list)):
                row.append(json.dumps(value, ensure_ascii=False))
            elif isinstance(value, str):
                normalized_value = value.replace("\r\n", "\n").replace("\\r\\n", "\n").replace("\\n", "\n")
                row.append(normalized_value)
            else:
                row.append(value)
        rows.append(row)

    return headers, rows


def load_selected_output_table(
    document_folder_path: str,
    selected_output_folder: Optional[str],
    selected_result_type: Optional[str],
) -> Tuple[Any, str]:
    """
    Load selected output JSON and prepare a table update for analyse mode.

    Parameters
    ----------
    document_folder_path : str
        document folder path that contains the `review` directory.
    selected_output_folder : Optional[str]
        selected output subfolder inside `review`.
    selected_result_type : Optional[str]
        result type selector (`answers` or `syntheses`).

    Returns
    -------
    Tuple[Any, str]
        table update payload and status message.
    """
    if not document_folder_path:
        return gr.update(headers=ANALYSIS_DEFAULT_HEADERS, value=[], visible=False), "Provide a document folder first."

    if not selected_output_folder:
        return gr.update(headers=ANALYSIS_DEFAULT_HEADERS, value=[], visible=False), "Select an output folder to continue."

    normalized_result_type = (selected_result_type or "answers").strip().lower()
    output_filename = "answers_summary.json" if normalized_result_type == "syntheses" else "answers.json"

    try:
        document_folder = Path(document_folder_path).expanduser().resolve()
    except OSError:
        return gr.update(headers=ANALYSIS_DEFAULT_HEADERS, value=[], visible=False), "Invalid document folder path."

    output_path = document_folder / "review" / selected_output_folder / output_filename
    if not output_path.is_file():
        return (
            gr.update(headers=ANALYSIS_DEFAULT_HEADERS, value=[], visible=False),
            f"No {output_filename} found in review/{selected_output_folder}.",
        )

    try:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read output file %s: %s", output_path, exc)
        return gr.update(headers=ANALYSIS_DEFAULT_HEADERS, value=[], visible=False), f"Failed to load {output_filename}: {exc}"

    headers, rows = _json_payload_to_table(payload)
    if not rows:
        return (
            gr.update(headers=ANALYSIS_DEFAULT_HEADERS, value=[], visible=False),
            f"Loaded {output_filename}, but it contains no rows.",
        )

    return (
        gr.update(headers=headers, value=rows, visible=True),
        f"Loaded {output_filename} from review/{selected_output_folder}.",
    )


def populate_form_from_selected_row(
    table_data: Any,
    selected_row: Optional[str],
) -> Tuple[str, str, str, str]:
    """
    Populate row editor fields from the currently selected table row.

    Parameters
    ----------
    table_data : Any
        question table data.
    selected_row : Optional[str]
        selected row number.

    Returns
    -------
    Tuple[str, str, str, str]
        row editor values `(question, task, instruction, classes)`.
    """
    rows = _normalize_questions_rows_with_options(table_data, drop_empty=False)
    index = _selected_row_index(selected_row, rows)
    if index is None:
        return _default_form_values()
    return _form_values_from_row(rows[index])


def add_question_row_from_form(
    table_data: Any,
    question: str,
    task: str,
    instruction: str,
    classes: Any,
) -> Tuple[Any, Any, str, str, str, str, str]:
    """
    Append a new question row from row-editor form values.

    Parameters
    ----------
    table_data : Any
        existing question table data.
    question : str
        question text.
    task : str
        task value.
    instruction : str
        instruction template.
    classes : Any
        classes value.

    Returns
    -------
    Tuple[Any, Any, str, str, str, str, str]
        updated table payload, row selector update, normalized form values, and status message.
    """
    rows = _normalize_questions_rows_with_options(table_data, drop_empty=False)
    new_row = _normalize_review_row([question, task, instruction, classes])
    rows.append(new_row)

    return (
        _questions_table_update(_with_row_numbers(rows)),
        _row_selector_update(rows, len(rows) - 1),
        *_form_values_from_row(new_row),
        f"Added row {len(rows)}.",
    )


def update_question_row_from_form(
    table_data: Any,
    selected_row: Optional[str],
    question: str,
    task: str,
    instruction: str,
    classes: Any,
) -> Tuple[Any, Any, str, str, str, str, str]:
    """
    Update an existing question row from row-editor form values.

    Parameters
    ----------
    table_data : Any
        existing question table data.
    selected_row : Optional[str]
        selected row number to update.
    question : str
        question text.
    task : str
        task value.
    instruction : str
        instruction template.
    classes : Any
        classes value.

    Returns
    -------
    Tuple[Any, Any, str, str, str, str, str]
        updated table payload, row selector update, normalized form values, and status message.
    """
    rows = _normalize_questions_rows_with_options(table_data, drop_empty=False)
    index = _selected_row_index(selected_row, rows)
    if index is None:
        return (
            _questions_table_update(_with_row_numbers(rows)),
            _row_selector_update(rows),
            question,
            _normalize_task(task),
            instruction,
            _classes_text_from_storage(classes),
            "Select a row before updating.",
        )

    updated_row = _normalize_review_row([question, task, instruction, classes])
    rows[index] = updated_row

    return (
        _questions_table_update(_with_row_numbers(rows)),
        _row_selector_update(rows, index),
        *_form_values_from_row(updated_row),
        f"Updated row {index + 1}.",
    )


def delete_question_row_from_form(
    table_data: Any,
    selected_row: Optional[str],
) -> Tuple[Any, Any, str, str, str, str, str]:
    """
    Delete the selected question row from table data.

    Parameters
    ----------
    table_data : Any
        existing question table data.
    selected_row : Optional[str]
        selected row number to delete.

    Returns
    -------
    Tuple[Any, Any, str, str, str, str, str]
        updated table payload, row selector update, next form values, and status message.
    """
    rows = _normalize_questions_rows_with_options(table_data, drop_empty=False)
    if not rows:
        return (
            _questions_table_update([]),
            _row_selector_update([]),
            *_default_form_values(),
            "No rows to delete.",
        )

    index = _selected_row_index(selected_row, rows)
    if index is None:
        return (
            _questions_table_update(_with_row_numbers(rows)),
            _row_selector_update(rows),
            *_default_form_values(),
            "Select a row before deleting.",
        )

    deleted_row_number = index + 1
    del rows[index]

    if not rows:
        return (
            _questions_table_update([]),
            _row_selector_update([]),
            *_default_form_values(),
            f"Deleted row {deleted_row_number}.",
        )

    next_index = min(index, len(rows) - 1)
    return (
        _questions_table_update(_with_row_numbers(rows)),
        _row_selector_update(rows, next_index),
        *_form_values_from_row(rows[next_index]),
        f"Deleted row {deleted_row_number}.",
    )


def save_questions_table(table_data: Any, folder_path: str) -> str:
    """
    Save normalized question rows to `review/questions.json`.

    Parameters
    ----------
    table_data : Any
        question table data.
    folder_path : str
        document folder path.

    Returns
    -------
    str
        status message describing save outcome.
    """
    if not folder_path:
        return "Questions not saved: provide a valid document folder path first."

    try:
        folder = Path(folder_path).expanduser().resolve()
    except OSError:
        return "Questions not saved: invalid folder path."

    if not folder.is_dir():
        return "Questions not saved: folder does not exist."

    rows = _normalize_questions_rows(table_data)
    json_path = _questions_json_path(folder_path)

    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload: List[dict[str, Any]] = []
        for row in rows:
            classes_text = _classes_text_from_storage(row[3]) if len(row) > 3 else ""
            payload.append(
                {
                    "question": str(row[0] or ""),
                    "task": _normalize_task(row[1] if len(row) > 1 else "answer"),
                    "instruction": str(row[2] or "") if len(row) > 2 else "",
                    "classes": classes_text,
                }
            )
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("Failed to save question list for %s: %s", folder_path, exc)
        return f"Question list not saved: {exc}"

    return f"Saved {len(rows)} question row(s) to question list at {json_path}."


def _load_questions_rows(folder_path: str) -> List[List[Union[str, bool]]]:
    """
    Load normalized question rows for a document folder.

    Parameters
    ----------
    folder_path : str
        document folder path.

    Returns
    -------
    List[List[Union[str, bool]]]
        normalized question rows, or an empty list when unavailable.
    """
    if not folder_path:
        return []

    try:
        folder = Path(folder_path).expanduser().resolve()
    except OSError:
        return []

    if not folder.is_dir():
        return []

    json_path = _questions_json_path(folder_path)

    try:
        if json_path.exists():
            return _load_questions_from_json(json_path)
    except Exception as exc:
        logger.warning("Failed to load questions for %s: %s", folder_path, exc)

    return []


def load_questions_table(folder_path: str) -> Any:
    """
    Load question rows and build an update payload for the overview table.

    Parameters
    ----------
    folder_path : str
        document folder path.

    Returns
    -------
    Any
        Gradio update object for the overview questions table.
    """
    rows = _load_questions_rows(folder_path)
    return _questions_table_update(_with_row_numbers(rows))


def load_row_editor(folder_path: str) -> Tuple[Any, str, str, str, str]:
    """
    Load row selector and initial editor values from saved questions.

    Parameters
    ----------
    folder_path : str
        document folder path.

    Returns
    -------
    Tuple[Any, str, str, str, str]
        row selector update and initial row editor values.
    """
    rows = _load_questions_rows(folder_path)
    if not rows:
        return _row_selector_update([]), *_default_form_values()

    return _row_selector_update(rows, 0), *_form_values_from_row(rows[0])


def save_and_reload_questions_table(
    table_data: Any,
    selected_row: Optional[str],
    question: str,
    task: str,
    instruction: str,
    classes: Any,
    folder_path: str,
) -> Tuple[str, Any, Any, str, str, str, str]:
    """
    Save current editor changes, reload persisted questions, and refresh UI state.

    Parameters
    ----------
    table_data : Any
        current question table data.
    selected_row : Optional[str]
        selected row number to persist.
    question : str
        question text.
    task : str
        task value.
    instruction : str
        instruction template.
    classes : Any
        classes value.
    folder_path : str
        document folder path.

    Returns
    -------
    Tuple[str, Any, Any, str, str, str, str]
        status message, refreshed table update, row selector update, and refreshed form values.
    """
    rows = _normalize_questions_rows_with_options(table_data, drop_empty=False)
    index = _selected_row_index(selected_row, rows)
    if index is None:
        return (
            "Select a row before saving.",
            _questions_table_update(_with_row_numbers(rows)),
            _row_selector_update(rows),
            question,
            _normalize_task(task),
            instruction,
            _classes_text_from_storage(classes),
        )

    updated_row = _normalize_review_row([question, task, instruction, classes])
    rows[index] = updated_row

    save_status = save_questions_table(table_data=rows, folder_path=folder_path)
    rows = _load_questions_rows(folder_path=folder_path)
    table_update = _questions_table_update(_with_row_numbers(rows))

    if not rows:
        return save_status, table_update, _row_selector_update([]), *_default_form_values()

    next_index = min(index, len(rows) - 1)
    return (
        f"Updated row {index + 1}. {save_status}",
        table_update,
        _row_selector_update(rows, next_index),
        *_form_values_from_row(rows[next_index]),
    )


def refresh_ingest_file_filter(folder_path: str) -> Any:
    """
    Refresh the ingest file filter choices for files in the selected folder.

    Parameters
    ----------
    folder_path : str
        document folder path.

    Returns
    -------
    Any
        Gradio update object for ingest file filter choices and selected values.
    """
    if not folder_path:
        return gr.update(choices=[], value=[])

    documents_root = Path(folder_path).expanduser().resolve()
    if not documents_root.is_dir():
        return gr.update(choices=[], value=[])

    file_names = [path.name for path in _list_document_paths_direct(documents_root)]
    return gr.update(choices=file_names, value=[])


def sync_folder_path(folder_path: str) -> Any:
    """
    Mirror a submitted folder path into the counterpart textbox.

    Parameters
    ----------
    folder_path : str
        submitted folder path.

    Returns
    -------
    Any
        Gradio update that sets the textbox value.
    """
    return gr.update(value=folder_path or "")


def ingest_or_load_documents(
    content_folder_name: str, content_folder_path: str, vecdb_folder_path: str
) -> None:
    """
    Depending on whether the vector store already exists, files will be chunked and stored in vectorstore or not

    Parameters
    ----------
    content_folder_name : str
        the name of the folder with content
    content_folder_path : str
        the full path of the folder with content
    vecdb_folder_path : str
        the full path of the folder with the vector stores
    """
    # if documents in source folder path are not ingested yet
    if not os.path.exists(vecdb_folder_path):
        from ingest.ingester import Ingester

        # ingest documents
        ingester = Ingester(collection_name=content_folder_name,
                            content_folder=content_folder_path,
                            vecdb_folder=vecdb_folder_path)
        ingester.ingest()
        logger.info(f"Created vector store in folder {vecdb_folder_path}")
    else:
        logger.info(f"Vector store already exists for folder {content_folder_name}")


def write_settings(
    input_path: str | os.PathLike[str],
    confidential: bool,
    output_path: str | os.PathLike[str],
) -> None:
    """
    Stores relevant settings to the output file, for reproducability purposes

    Parameters
    ----------
    input_path : os.PathLike
        path of the input settings file
    confidential : bool
        whether the documents are confidential or not
    output_path : os.PathLike
        path of the output file to write the used settings to
    """
    with open(file=output_path, mode="w", encoding="utf8") as file:
        file.write(f"input path =  {input_path} \n")
        file.write(f"confidential =  {confidential} \n")
        file.write(f"settings.TEXT_SPLITTER_METHOD =  {settings.TEXT_SPLITTER_METHOD} \n")
        file.write(f"settings.TEXT_SPLITTER_CHILD =  {settings.TEXT_SPLITTER_METHOD_CHILD} \n")
        file.write(f"settings.CHUNK_SIZE =  {settings.CHUNK_SIZE} \n")
        file.write(f"settings.CHUNK_SIZE_CHILD =  {settings.CHUNK_SIZE_CHILD} \n")
        file.write(f"settings.CHUNK_K =  {settings.CHUNK_K} \n")
        file.write(f"settings.CHUNK_K_CHILD =  {settings.CHUNK_K_CHILD} \n")
        file.write(f"settings.CHUNK_OVERLAP =  {settings.CHUNK_OVERLAP} \n")
        file.write(f"settings.CHUNK_OVERLAP_CHILD =  {settings.CHUNK_OVERLAP_CHILD} \n")
        if not confidential:
            file.write(f"settings.EMBEDDINGS_PROVIDER =  {settings.EMBEDDINGS_PROVIDER} \n")
            file.write(f"settings.EMBEDDINGS_MODEL =  {settings.EMBEDDINGS_MODEL} \n")
            file.write(f"settings.LLM_PROVIDER =  {settings.LLM_PROVIDER} \n")
            file.write(f"settings.LLM_MODEL =  {settings.LLM_MODEL} \n")
        else:
            file.write(f"settings.PRIVATE_EMBEDDINGS_PROVIDER =  {settings.PRIVATE_EMBEDDINGS_PROVIDER} \n")
            file.write(f"settings.PRIVATE_EMBEDDINGS_MODEL =  {settings.PRIVATE_EMBEDDINGS_MODEL} \n")
            file.write(f"settings.PRIVATE_LLM_PROVIDER =  {settings.PRIVATE_LLM_PROVIDER} \n")
            file.write(f"settings.PRIVATE_LLM_MODEL =  {settings.PRIVATE_LLM_MODEL} \n")
        file.write(f"settings.SEARCH_TYPE =  {settings.SEARCH_TYPE} \n")
        file.write(f"settings.SCORE_THRESHOLD =  {settings.SCORE_THRESHOLD} \n")
        file.write(f"settings.RETRIEVER_TYPE =  {settings.RETRIEVER_TYPE} \n")
        file.write(f"settings.RERANK =  {settings.RERANK} \n")
        file.write(f"settings.RERANK_PROVIDER =  {settings.RERANK_PROVIDER} \n")
        file.write(f"settings.RERANK_MODEL =  {settings.RERANK_MODEL} \n")
        file.write(f"settings.CHUNK_K_FOR_RERANK =  {settings.CHUNK_K_FOR_RERANK} \n")
        file.write(f"settings.RETRIEVER_PROMPT_TEMPLATE =  {settings.RETRIEVER_PROMPT_TEMPLATE} \n\n")


def check_string_formatting(
    review_instruction: Optional[str],
    review_task: str,
) -> None:
    """
    Check if the string formatting is correct for the given instruction and task.

    Parameters
    ----------
    review_instruction : Optional[str]
        the instruction to be checked
    review_task : str
        normalized task type (answer, classify, synthesis)
    Raises
    ------
    ValueError
        if the string formatting of one of the arguments is not correct
    """
    instruction = (review_instruction or "").strip()
    normalized_task = _normalize_task(review_task)

    if not instruction:
        raise ValueError(
            f"Instruction is required for task '{normalized_task}'."
        )

    if normalized_task in {"answer", "classify"}:
        if ("{question}" not in instruction) or ("{context}" not in instruction):
            logger.info(
                "Instruction for task '%s' is missing required placeholders {question} and/or {context}.",
                normalized_task,
            )
            raise ValueError(
                "Instruction must contain {question} and {context} for tasks 'answer' and 'classification'."
            )

    if normalized_task == "synthesis":
        if ("{question}" not in instruction) or ("{answer_string}" not in instruction):
            logger.info(
                "Instruction for synthesis task is missing required placeholders {question} and/or {answer_string}."
            )
            raise ValueError(
                "Instruction must contain {question} and {answer_string} for task 'synthesis'."
            )


def _is_synthesis_template(candidate: Optional[str]) -> bool:
    """
    Check whether a template supports synthesis placeholders.

    Parameters
    ----------
    candidate : Optional[str]
        candidate prompt template.

    Returns
    -------
    bool
        True when template includes `{question}` and `{answer_string}` placeholders.
    """
    if not candidate:
        return False
    return "{question}" in candidate and "{answer_string}" in candidate

def generate_answer(
    querier: Querier, review_question: str
) -> Tuple[str, str]:
    """
    Generate an answer to the given question with the provided Querier instance

    Parameters
    ----------
    querier : Querier
        the Querier object
    review_question :str
        the question to be answered

    Returns
    -------
    Tuple[str, str]
        tuple containing the answer and the associated sources used (in string form)
    """
    # Clear history before answering the question
    querier.clear_history()
    source_docs = ""

    try:
        response = querier.ask_question(review_question)
        for doc in response["source_documents"]:
            source_docs += (
                f"page {str(doc.metadata['page_number'])}\n{doc.page_content}\n\n"
            )
        return response["answer"], source_docs
    except Exception as e:
        api_error_type = _get_openai_api_error_type()
        if isinstance(e, api_error_type):
            error_code = getattr(e, "code", "unknown")
            if error_code == "content_filter":
                return "APIError: Content filtered.", "No sources"
            print(f"APIError: {error_code}")
            return f"APIError: {error_code}", "No sources"
        print(f"Unexpected error: {e}")
        return f"Error {e}", f"No sources"



def create_answers_for_folder(question_list_path: str,
                              review_files: Sequence[Path | str],
                              content_folder_name: str,
                              querier: Querier,
                              vecdb_folder_path: str,
                              output_path: str | os.PathLike[str]) -> Iterator[str]:
    """
    Loop over all questions and documents, gather answers, and store on disk.

    Parameters
    ----------
    question_list_path : str
        path of the JSON file with the list of questions
    review_files : Sequence[Path | str]
        list of file paths (or filenames) to be reviewed
    content_folder_name : str
        name of the document folder
    querier : Querier
        the Querier object
    vecdb_folder_path : str
        path of the vector database
    output_path : os.PathLike
        path of the output file

    Yields
    ------
    Iterator[str]
        progress status messages for the UI
    """
    answer_rows: List[dict[str, Any]] = []
    # load review questions from JSON
    question_list_json_path = Path(question_list_path)
    if not question_list_json_path.exists():
        message = f"Questions file not found at {question_list_path}."
        logger.warning(message)
        yield message
        return

    review_questions = _load_questions_from_json(question_list_json_path)
    if not review_questions:
        message = f"No review questions found in {question_list_path}."
        logger.warning(message)
        yield message
        return

    question_rows = [
        dict(zip(REVIEW_COLUMNS, row))
        for row in review_questions
    ]
    synthesis_templates_by_question_id: dict[int, str] = {}
    for question_id, row in enumerate(question_rows, start=1):
        if _normalize_task(row.get("task")) != "synthesis":
            continue
        candidate = str(row.get("instruction") or "").strip()
        if _is_synthesis_template(candidate):
            synthesis_templates_by_question_id[question_id] = candidate

    # loop over each file, then over each row in questions.json
    for review_file in review_files:
        review_file_name = Path(review_file).name
        logger.info(f"Reviewing file {review_file_name}...")
        for question_id, row in enumerate(question_rows, start=1):
            review_question = str(row.get("question") or "")
            review_instruction_raw = str(row.get("instruction") or "")
            review_task = _normalize_task(row.get("task"))
            review_classes = _classes_text_from_storage(row.get("classes"))
            review_instruction = review_instruction_raw if review_instruction_raw.strip() else None

            if not review_question.strip():
                continue

            logger.info(f"reviewing question {review_question}")
            yield f"Reviewing file: {review_file_name} | question: {review_question}"

            try:
                # Check task-specific placeholder requirements on the row instruction.
                check_string_formatting(
                    review_instruction=review_instruction,
                    review_task=review_task,
                )

                # Synthesis instructions are validated, but not used as QA templates.
                chain_instruction = None if review_task == "synthesis" else review_instruction

                # create the query chain with a search filter and answer each question for each document
                querier.make_chain(content_folder=content_folder_name,
                                   vecdb_folder=vecdb_folder_path,
                                   search_filter={"filename": review_file_name},
                                   qa_template_file_path_or_string=chain_instruction)

                # Generate answer
                answer, sources = generate_answer(querier=querier,
                                                  review_question=review_question)
            except Exception as exc:
                logger.exception(
                    "Error while reviewing question '%s' for file '%s'",
                    review_question,
                    review_file_name,
                )
                yield (
                    f"Error while reviewing file: {review_file_name} | "
                    f"question: {review_question} | {exc}"
                )
                continue

            # For synthesis tasks, add document reference to support cross-document synthesis.
            answer_plus_document_reference = f"This answer is from {review_file_name}:\n {answer}"
            final_answer = answer_plus_document_reference if review_task == "synthesis" else answer

            # add resulting answer and input data to dataframe
            answer_rows.append(
                {
                    "filename": review_file_name,
                    "id": question_id,
                    "question": review_question,
                    "instruction": review_instruction_raw,
                    "task": review_task,
                    "classes": review_classes,
                    "answer": final_answer,
                    "sources": sources,
                }
            )

    # If one of the rows is a synthesis task, create one synthesized answer per question.
    if answer_rows and any(row.get("task") == "synthesis" for row in answer_rows):
        summary_records: List[dict[str, Any]] = []
        synthesis_questions: dict[int, str] = {}
        for row in answer_rows:
            if row.get("task") != "synthesis":
                continue
            question_id = int(row.get("id") or 0)
            if question_id not in synthesis_questions:
                synthesis_questions[question_id] = str(row.get("question") or "")

        for question_id in sorted(synthesis_questions.keys()):
            question = synthesis_questions[question_id]
            answers = [
                str(row.get("answer") or "")
                for row in answer_rows
                if int(row.get("id") or 0) == question_id
            ]
            if not answers:
                continue

            template = synthesis_templates_by_question_id.get(question_id, "")
            if not _is_synthesis_template(template):
                template = DEFAULT_SYNTHESIS_TEMPLATE

            synthesis_prompt = template.format(
                question=question,
                answer_string="\n\n".join(answers),
            )
            if querier.llm is None:
                logger.warning("Skipping synthesis for question '%s': querier.llm is None.", question)
                continue

            synthesis_response = querier.llm.invoke(synthesis_prompt)
            synthesis_answer = getattr(synthesis_response, "content", str(synthesis_response))

            summary_records.append(
                {
                    "id": question_id,
                    "question": question,
                    "answer": synthesis_answer,
                }
            )

            yield f"Synthesized cross-document answers for question: {question}"

        if summary_records:
            output_path_str = str(output_path)
            filename, extension = os.path.splitext(output_path_str)
            output_path_summary = f"{filename}_summary{extension}"
            with open(file=output_path_summary, mode="w", encoding="utf8") as summary_file:
                json.dump(summary_records, summary_file, indent=2, ensure_ascii=False)

    # Then save to JSON
    for row in answer_rows:
        row["classes"] = _classes_text_from_storage(row.get("classes"))

    sorted_answer_rows = sorted(
        answer_rows,
        key=lambda row: (
            str(row.get("filename") or "").lower(),
            int(row.get("id") or 0),
        ),
    )
    with open(file=output_path, mode="w", encoding="utf8") as file:
        json.dump(sorted_answer_rows, file, indent=2, ensure_ascii=False)


def handle_ingestion(folder_path: str, ingest_file_filter: List[str]) -> Iterator[Tuple[str, str]]:
    """
    Main loop of this module

    Parameters
    ----------
    folder_path : str
        the path of the folder with documents
    """
    # determine the list of relevant files in document folder
    status_lines: List[str] = []

    def _emit_status(message: str, active_folder: str) -> Tuple[str, str]:
        """
        Append a status line and return the aggregated ingestion payload.

        Parameters
        ----------
        message : str
            status message to append.
        active_folder : str
            active document folder path.

        Returns
        -------
        Tuple[str, str]
            tuple containing aggregated status text and active folder value.
        """
        status_lines.append(message)
        return _ingestion_payload("\n".join(status_lines), active_folder)

    # Get content folder name from path
    content_folder_name = os.path.basename(folder_path)
    # if content folder path does not exist, stop
    if not folder_path or not Path(folder_path).is_dir():
        yield _ingestion_payload("Please provide a valid folder path.", "")
        return

    documents_root = Path(folder_path).expanduser().resolve()
    yield _emit_status(f"Starting review run for folder: {documents_root}", str(documents_root))

    all_paths = _list_document_paths_direct(documents_root)
    if not all_paths:
        yield _emit_status("No supported files found in the selected folder.", str(documents_root))
        return

    selected_names = {str(name).strip().lower() for name in (ingest_file_filter or []) if str(name).strip()}
    paths = (
        [path for path in all_paths if path.name.lower() in selected_names]
        if selected_names
        else all_paths
    )

    if selected_names and not paths:
        yield _emit_status(
            "No selected files were found in the folder. Refresh file selection and try again.",
            str(documents_root),
        )
        return

    yield _emit_status(f"Selected {len(paths)} file(s) for processing.", str(documents_root))

    try:
        # get relevant models
        confidential = False
        ut = _get_utils_module()
        yield _emit_status("Loading model configuration...", str(documents_root))
        llm_provider, llm_model, embeddings_provider, embeddings_model = ut.get_relevant_models(summary=False,
                                                                                                private=confidential)
        # created vector db path
        vecdb_folder_path = ut.create_vectordb_path(content_folder_path=folder_path,
                                                    embeddings_provider=embeddings_provider,
                                                    embeddings_model=embeddings_model)
        yield _emit_status("Prepared vector store path.", str(documents_root))

        # create output folder with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d_%Hhour_%Mmin_%Ssec")
        os.mkdir(os.path.join(folder_path, f"review/{timestamp}"))
        yield _emit_status(f"Created output folder review/{timestamp}.", str(documents_root))

        # path of file with review questions
        question_list_path = os.path.join(folder_path, "review", "questions.json")
        question_list_path_obj = Path(question_list_path)
        if not question_list_path_obj.is_file():
            yield _emit_status(
                "Run failed: no review/questions.json found. Save the question list first.",
                str(documents_root),
            )
            return

        try:
            latest_mtime = datetime.fromtimestamp(question_list_path_obj.stat().st_mtime)
            yield _emit_status(
                f"Using latest questions.json (last saved {latest_mtime.isoformat(timespec='seconds')}).",
                str(documents_root),
            )
        except OSError:
            yield _emit_status("Using latest questions.json.", str(documents_root))

        # copy the question list file to the output folder and run from this snapshot
        destination_path = os.path.join(folder_path, f"review/{timestamp}", "questions.json")
        shutil.copy(question_list_path, destination_path)
        yield _emit_status("Copied questions.json snapshot for this run.", str(documents_root))

        # ingest documents if documents in source folder path are not ingested yet
        yield _emit_status("Checking or building vector store (may take a while)...", str(documents_root))
        ingest_or_load_documents(content_folder_name=content_folder_name,
                                 content_folder_path=folder_path,
                                 vecdb_folder_path=vecdb_folder_path)
        yield _emit_status("Vector store ready.", str(documents_root))

        # write settings to file
        output_path_settings = os.path.join(
            folder_path, f"review/{timestamp}", "settings.txt"
        )
        write_settings(input_path=folder_path,
                       confidential=confidential,
                       output_path=output_path_settings)
        yield _emit_status("Saved run settings snapshot.", str(documents_root))

        # Create answers and store them in the file specified by output_path
        output_path_review = os.path.join(
            folder_path, f"review/{timestamp}", "answers.json"
        )

        # create instance of Querier once
        yield _emit_status("Initializing query engine...", str(documents_root))
        querier = Querier(
            llm_provider=llm_provider,
            llm_model=llm_model,
            embeddings_provider=embeddings_provider,
            embeddings_model=embeddings_model,
        )
        yield _emit_status("Query engine ready.", str(documents_root))
        yield _emit_status("Starting question processing...", str(documents_root))

        for progress_message in create_answers_for_folder(
            question_list_path=destination_path,
            review_files=paths,
            content_folder_name=content_folder_name,
            querier=querier,
            vecdb_folder_path=vecdb_folder_path,
            output_path=output_path_review,
        ):
            yield _emit_status(progress_message, str(documents_root))

        completion_message = "Successfully reviewed the documents."
        logger.info(completion_message)
        yield _emit_status(completion_message, str(documents_root))
    except Exception as exc:
        logger.exception("Review run failed for folder %s", folder_path)
        yield _emit_status(f"Run failed: {exc}", str(documents_root))



################# User Interface #################
with gr.Blocks(analytics_enabled=False, head=KEEPALIVE_HEAD) as demo:
    active_folder = gr.State("")

    with gr.Sidebar():
        gr.Markdown("# ChatPBL DocReviewer")
        screen_mode = gr.Radio(
            label="Mode",
            choices=["ingest folder", "analyse output"],
            value="ingest folder",
            interactive=True,
        )

        with gr.Group() as ingest_sidebar_group:
            folder_path_input = gr.Textbox(label="Document folder", 
                                           info="Insert the complete path and hit Enter", 
                                           placeholder="C:\\path\\to\\documents")

            ingest_file_filter = gr.CheckboxGroup(
                choices=[],
                value=[],
                label="Files to process",
                info="Select one or more files. Leave empty to process all supported files.",
                elem_id="ingest-file-filter",
            )
            status_messages = gr.Textbox(label="Status", interactive=False, lines=8)

        with gr.Group(visible=False) as analyse_sidebar_group:
            analyse_folder_path_input = gr.Textbox(
                label="Document folder",
                info="Insert the complete path and hit Enter",
                placeholder="C:\\path\\to\\documents",
            )
            analyse_output_folder_dropdown = gr.Dropdown(
                label="Available output folders",
                choices=[],
                value=None,
                visible=False,
                interactive=True,
            )
            analyse_result_type_radio = gr.Radio(
                label="Output type",
                choices=["answers", "syntheses"],
                value="answers",
                visible=False,
                interactive=True,
            )
            analyse_status_messages = gr.Textbox(label="Status", interactive=False, lines=3)

    with gr.Accordion("Overview of Current Question list", open=False, visible=False) as question_overview_accordion:
        questions_table = gr.Dataframe(
            headers=TABLE_COLUMNS,
            column_widths=REVIEW_COLUMN_WIDTHS,
            datatype="markdown",
            value=[],
            row_count=1,
            label="Overview of Review Questions (output)",
            elem_id="questions-table",
            wrap=True,
            interactive=False,
            static_columns=[0],
        )

    with gr.Accordion("Input form for review questions", open=False, visible=False) as main_screen_accordion:
        with gr.Group() as main_screen_group:
            with gr.Row():
                row_selector = gr.Dropdown(
                    label="id",
                    choices=[],
                    value=None,
                    interactive=True,
                    scale=1,
                )
                question_input = gr.Textbox(label="question", lines=3, scale=7)
                task_input = gr.Radio(
                    label="task",
                    choices=list(TASK_OPTIONS),
                    value="answer",
                    interactive=True,
                    scale=2,
                )
                classes_input = gr.Textbox(
                    label="classes",
                    interactive=True,
                    visible=False,
                    scale=3,
                    lines=3,
                    placeholder="One class per line",
                    info="Use one class per line.",
                )
            instruction_input = gr.Textbox(label="instruction", lines=6)

            with gr.Row():
                add_question_row_btn: Any = gr.Button(value="Add row", variant="primary", elem_id="add-row-btn")
                delete_question_row_btn: Any = gr.Button(value="Delete row", variant="stop")

            with gr.Row():
                save_questions_btn: Any = gr.Button(value="Save Question list", variant="primary")

    go_btn: Any = gr.Button(value="Query selected files with current question list", variant="primary", elem_id="go-btn", visible=False)

    with gr.Group(visible=False) as analyse_main_group:
        gr.Markdown("## Output Analysis")
        analyse_output_table = gr.Dataframe(
            headers=ANALYSIS_DEFAULT_HEADERS,
            datatype="markdown",
            value=[],
            label="Selected output",
            elem_id="analysis-output-table",
            visible=False,
            interactive=False,
            wrap=True,
        )

    # Event handlers
    folder_path_input.submit(
        fn=refresh_ingest_file_filter,
        inputs=[folder_path_input],
        outputs=[ingest_file_filter],
    )

    folder_path_input.submit(
        fn=sync_folder_path,
        inputs=[folder_path_input],
        outputs=[analyse_folder_path_input],
    )

    folder_path_input.submit(
        fn=load_questions_table,
        inputs=[folder_path_input],
        outputs=[questions_table],
    )

    folder_path_input.submit(
        fn=_ingest_main_visibility_update,
        inputs=[folder_path_input],
        outputs=[
            question_overview_accordion,
            main_screen_accordion,
            go_btn,
        ],
    )

    load_row_editor_event = folder_path_input.submit(
        fn=load_row_editor,
        inputs=[folder_path_input],
        outputs=[
            row_selector,
            question_input,
            task_input,
            instruction_input,
            classes_input,
        ],
    )
    load_row_editor_event.then(
        fn=_classes_input_update,
        inputs=[task_input, classes_input],
        outputs=[classes_input],
    )

    row_selector_change_event = row_selector.change(
        fn=populate_form_from_selected_row,
        inputs=[questions_table, row_selector],
        outputs=[
            question_input,
            task_input,
            instruction_input,
            classes_input,
        ],
    )
    row_selector_change_event.then(
        fn=_classes_input_update,
        inputs=[task_input, classes_input],
        outputs=[classes_input],
    )

    task_input.change(
        fn=_classes_input_update,
        inputs=[task_input, classes_input],
        outputs=[classes_input],
    )

    add_question_row_event = add_question_row_btn.click(
        fn=add_question_row_from_form,
        inputs=[
            questions_table,
            question_input,
            task_input,
            instruction_input,
            classes_input,
        ],
        outputs=[
            questions_table,
            row_selector,
            question_input,
            task_input,
            instruction_input,
            classes_input,
            status_messages,
        ],
    )
    add_question_row_event.then(
        fn=_classes_input_update,
        inputs=[task_input, classes_input],
        outputs=[classes_input],
    )

    delete_question_row_event = delete_question_row_btn.click(
        fn=delete_question_row_from_form,
        inputs=[questions_table, row_selector],
        outputs=[
            questions_table,
            row_selector,
            question_input,
            task_input,
            instruction_input,
            classes_input,
            status_messages,
        ],
    )
    delete_question_row_event.then(
        fn=_classes_input_update,
        inputs=[task_input, classes_input],
        outputs=[classes_input],
    )

    save_questions_event = save_questions_btn.click(
        fn=save_and_reload_questions_table,
        inputs=[
            questions_table,
            row_selector,
            question_input,
            task_input,
            instruction_input,
            classes_input,
            folder_path_input,
        ],
        outputs=[
            status_messages,
            questions_table,
            row_selector,
            question_input,
            task_input,
            instruction_input,
            classes_input,
        ],
    )
    save_questions_event.then(
        fn=_classes_input_update,
        inputs=[task_input, classes_input],
        outputs=[classes_input],
    )

    go_btn_run_event = go_btn.click(
        fn=handle_ingestion,
        inputs=[folder_path_input, ingest_file_filter],
        outputs=[status_messages, active_folder],
    )
    go_btn_run_event.then(
        fn=load_review_output_folders,
        inputs=[folder_path_input],
        outputs=[analyse_output_folder_dropdown],
    )

    screen_mode_change_event = screen_mode.change(
        fn=_screen_mode_update,
        inputs=[screen_mode, folder_path_input],
        outputs=[
            ingest_sidebar_group,
            main_screen_accordion,
            analyse_sidebar_group,
            analyse_main_group,
            question_overview_accordion,
            go_btn,
        ],
    )
    screen_mode_change_event.then(
        fn=refresh_review_output_folders_for_mode,
        inputs=[screen_mode, analyse_folder_path_input],
        outputs=[analyse_output_folder_dropdown],
    )
    screen_mode_change_event.then(
        fn=_analysis_result_type_update,
        inputs=[analyse_output_folder_dropdown],
        outputs=[analyse_result_type_radio],
    )
    screen_mode_change_event.then(
        fn=load_selected_output_table,
        inputs=[analyse_folder_path_input, analyse_output_folder_dropdown, analyse_result_type_radio],
        outputs=[analyse_output_table, analyse_status_messages],
    )

    analyse_folder_submit_event = analyse_folder_path_input.submit(
        fn=load_review_output_folders,
        inputs=[analyse_folder_path_input],
        outputs=[analyse_output_folder_dropdown],
    )

    analyse_folder_path_input.submit(
        fn=sync_folder_path,
        inputs=[analyse_folder_path_input],
        outputs=[folder_path_input],
    )

    analyse_folder_submit_event.then(
        fn=_analysis_result_type_update,
        inputs=[analyse_output_folder_dropdown],
        outputs=[analyse_result_type_radio],
    )

    analyse_folder_submit_event.then(
        fn=load_selected_output_table,
        inputs=[analyse_folder_path_input, analyse_output_folder_dropdown, analyse_result_type_radio],
        outputs=[analyse_output_table, analyse_status_messages],
    )

    analyse_output_folder_dropdown.change(
        fn=_analysis_result_type_update,
        inputs=[analyse_output_folder_dropdown],
        outputs=[analyse_result_type_radio],
    )

    analyse_output_folder_dropdown.change(
        fn=load_selected_output_table,
        inputs=[analyse_folder_path_input, analyse_output_folder_dropdown, analyse_result_type_radio],
        outputs=[analyse_output_table, analyse_status_messages],
    )

    analyse_result_type_radio.change(
        fn=load_selected_output_table,
        inputs=[analyse_folder_path_input, analyse_output_folder_dropdown, analyse_result_type_radio],
        outputs=[analyse_output_table, analyse_status_messages],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    enable_pwa = os.getenv("APP_ENABLE_PWA", "0").strip().lower() in {"1", "true", "yes", "on"}
    demo.launch(inbrowser=True, pwa=enable_pwa, css=TABLE_CSS)
