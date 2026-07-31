from __future__ import annotations

import json
import os
import shutil
import csv
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union, Iterable, Sequence
import gradio as gr
import pandas as pd
from openai import APIError
# local imports
import settings
from query.querier import Querier
import logging


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_utils_module() -> Any:
    import utils as ut

    return ut

REVIEW_COLUMNS = [
    "question",
    "task",
    "instruction",
    "classes",
]

ROW_NUMBER_COLUMN = "row"
TABLE_COLUMNS = [ROW_NUMBER_COLUMN, *REVIEW_COLUMNS]
REVIEW_COLUMN_WIDTHS: List[str | int] = ["6%", "24%", "12%", "38%", "20%"]

TASK_OPTIONS = ("answer", "classify", "synthesis")

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

#go-btn {
    width: 50px !important;
    min-width: 50px !important;
    max-width: 50px !important;
}

#go-btn > button {
    width: 50px !important;
    min-width: 50px !important;
    max-width: 50px !important;
    background-color: #0b5d1e !important;
    border-color: #0b5d1e !important;
    color: #ffffff !important;
}

#go-btn > button:hover {
    background-color: #094a18 !important;
    border-color: #094a18 !important;
}

#questions-table {
    --cell-line-height: 1.25rem;
    --cell-max-lines: 12;
    font-size: 0.8rem;
}

#questions-table th,
#questions-table td {
    white-space: normal !important;
    overflow-wrap: anywhere;
    word-break: break-word;
    vertical-align: top;
    font-size: 0.8rem !important;
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

"""

def list_document_paths(
    documents_root: Union[Path, str],
    *,
    valid_extensions: Optional[Iterable[str]] = None,
) -> List[Path]:
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
    return [path for path in list_document_paths(documents_root) if path.parent == documents_root]


def _ingestion_payload(
    status: str,
    active_folder_value: str,
) -> Tuple[str, str]:
    return status, active_folder_value


def _questions_json_path(folder_path: str) -> Path:
    return Path(folder_path).expanduser().resolve() / "review" / "questions.json"


def _questions_csv_path(folder_path: str) -> Path:
    return Path(folder_path).expanduser().resolve() / "review" / "questions.csv"


def _normalize_classification(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower() if value is not None else ""
    return normalized in {"true", "1", "yes", "y", "on"}


def _normalize_task(value: Any) -> str:
    normalized = str(value).strip().lower() if value is not None else ""
    if normalized in TASK_OPTIONS:
        return normalized
    if normalized in {"classification", "class"}:
        return "classify"
    return "answer"


def _task_from_mapping(row: dict[str, Any]) -> str:
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
    padded = values[: len(REVIEW_COLUMNS)] + [""] * (len(REVIEW_COLUMNS) - len(values))
    return [
        str(padded[0] or ""),
        _normalize_task(padded[1]),
        str(padded[2] or ""),
        str(padded[3] or ""),
    ]


def _first_present_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return ""


def _row_from_mapping(row: dict[str, Any]) -> List[Union[str, bool]]:
    return _normalize_review_row(
        [
            _first_present_value(row, "question", "Question"),
            _task_from_mapping(row),
            _first_present_value(row, "instruction", "Instruction"),
            _first_present_value(row, "classes", "Classes"),
        ]
    )


def _load_questions_from_json(json_path: Path) -> pd.DataFrame:
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

    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def _load_questions_from_csv(csv_path: Path) -> List[List[Union[str, bool]]]:
    rows: List[List[Union[str, bool]]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(_row_from_mapping(row))
    return rows


def _with_row_numbers(rows: List[List[Union[str, bool]]]) -> List[List[Union[str, bool]]]:
    numbered_rows: List[List[Union[str, bool]]] = []
    for index, row in enumerate(rows, start=1):
        numbered_rows.append([str(index), *row])
    return numbered_rows


def _questions_table_update(value: Any) -> Any:
    # Render the table as output-only; editing happens via the custom form.
    row_count = max(len(value), 1) if isinstance(value, list) else None
    row_limits = (row_count, row_count) if row_count is not None else None
    return gr.update(value=value, row_count=row_count, row_limits=row_limits, interactive=False)


def _normalize_questions_rows(table_data: Any) -> List[List[Union[str, bool]]]:
    return _normalize_questions_rows_with_options(table_data, drop_empty=True)


def _normalize_questions_rows_with_options(
    table_data: Any,
    *,
    drop_empty: bool,
) -> List[List[Union[str, bool]]]:
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


def _default_form_values() -> Tuple[str, str, str, str]:
    return "", "answer", "", ""


def _row_selector_update(rows: List[List[Union[str, bool]]], selected_index: Optional[int] = None) -> Any:
    if not rows:
        return gr.update(choices=[], value=None)

    choices = []
    for index, _ in enumerate(rows):
        choices.append((str(index + 1), str(index)))

    if selected_index is None or selected_index < 0 or selected_index >= len(rows):
        selected_index = 0

    return gr.update(choices=choices, value=str(selected_index))


def _selected_row_index(selected_row: Optional[str], rows: List[List[Union[str, bool]]]) -> Optional[int]:
    if selected_row is None:
        return None
    try:
        index = int(selected_row)
    except (TypeError, ValueError):
        return None

    if index < 0 or index >= len(rows):
        return None
    return index


def _form_values_from_row(row: List[Union[str, bool]]) -> Tuple[str, str, str, str]:
    return (
        str(row[0] or ""),
        _normalize_task(row[1]),
        str(row[2] or ""),
        str(row[3] or ""),
    )


def populate_form_from_selected_row(
    table_data: Any,
    selected_row: Optional[str],
) -> Tuple[str, str, str, str]:
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
    classes: str,
) -> Tuple[Any, Any, str, str, str, str, str]:
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
    classes: str,
) -> Tuple[Any, Any, str, str, str, str, str]:
    rows = _normalize_questions_rows_with_options(table_data, drop_empty=False)
    index = _selected_row_index(selected_row, rows)
    if index is None:
        return (
            _questions_table_update(_with_row_numbers(rows)),
            _row_selector_update(rows),
            question,
            _normalize_task(task),
            instruction,
            classes,
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
        payload = [dict(zip(REVIEW_COLUMNS, row)) for row in rows]
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("Failed to save question list for %s: %s", folder_path, exc)
        return f"Question list not saved: {exc}"

    return f"Saved {len(rows)} question row(s) to question list at {json_path}."


def _load_questions_rows(folder_path: str) -> List[List[Union[str, bool]]]:
    if not folder_path:
        return []

    try:
        folder = Path(folder_path).expanduser().resolve()
    except OSError:
        return []

    if not folder.is_dir():
        return []

    json_path = _questions_json_path(folder_path)
    csv_path = _questions_csv_path(folder_path)

    try:
        if json_path.exists():
            rows_df = _load_questions_from_json(json_path)
            return rows_df.values.tolist()

        if csv_path.exists():
            rows = _load_questions_from_csv(csv_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_payload = [dict(zip(REVIEW_COLUMNS, row)) for row in rows]
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(json_payload, handle, indent=2, ensure_ascii=False)
            return rows
    except Exception as exc:
        logger.warning("Failed to load questions for %s: %s", folder_path, exc)

    return []


def load_questions_table(folder_path: str) -> Any:
    rows = _load_questions_rows(folder_path)
    return _questions_table_update(_with_row_numbers(rows))


def load_row_editor(folder_path: str) -> Tuple[Any, str, str, str, str]:
    rows = _load_questions_rows(folder_path)
    if not rows:
        return _row_selector_update([]), *_default_form_values()

    return _row_selector_update(rows, 0), *_form_values_from_row(rows[0])


def save_and_reload_questions_table(
    table_data: Any,
    folder_path: str,
) -> Tuple[str, Any, Any, str, str, str, str]:
    save_status = save_questions_table(table_data=table_data, folder_path=folder_path)
    rows = _load_questions_rows(folder_path=folder_path)
    table_update = _questions_table_update(_with_row_numbers(rows))

    if not rows:
        return save_status, table_update, _row_selector_update([]), *_default_form_values()

    return save_status, table_update, _row_selector_update(rows, 0), *_form_values_from_row(rows[0])


def refresh_ingest_file_filter(folder_path: str) -> Any:
    if not folder_path:
        return gr.update(choices=[], value=[])

    documents_root = Path(folder_path).expanduser().resolve()
    if not documents_root.is_dir():
        return gr.update(choices=[], value=[])

    file_names = [path.name for path in _list_document_paths_direct(documents_root)]
    return gr.update(choices=file_names, value=[])


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
) -> None:
    """
    Check if the string formatting is correct for the given instruction.

    Parameters
    ----------
    review_instruction : Optional[str]
        the instruction to be checked
    Raises
    ------
    ValueError
        if the string formatting of one of the arguments is not correct
    """
    # if there is an entry for the instruction
    if not pd.isna(review_instruction):
        # check if the instruction is properly formatted
        if ("{question}" not in review_instruction) or \
           ("{context}" not in review_instruction):
            logger.info(f"""The instruction {review_instruction} does not contain
                         the required placeholders {{question}} and {{context}}.""")
            raise ValueError(
                f"""The instruction {review_instruction} does not contain
                the required placeholders {{question}} and/or {{context}}.""")

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
    except APIError as e:
        if e.code == "content_filter":
            return "APIError: Content filtered.", "No sources"
        else:
            print(f"APIError: {e.code}")
            return f"APIError: {e.code}", "No sources"
    except Exception as e:
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
    # create empty dataframe
    df_result = pd.DataFrame(
        columns=[
            "filename",
            "question_id",
            "question",
            "instruction",
            "task",
            "classes",
            "answer",
            "sources"
        ]
    )
    # load review questions from JSON
    question_list_json_path = Path(question_list_path)
    if not question_list_json_path.exists():
        message = f"Questions file not found at {question_list_path}."
        logger.warning(message)
        yield message
        return

    review_questions = _load_questions_from_json(question_list_json_path)
    if review_questions.empty:
        message = f"No review questions found in {question_list_path}."
        logger.warning(message)
        yield message
        return

    question_rows = review_questions.to_dict(orient="records")

    # loop over each file, then over each row in questions.json
    for review_file in review_files:
        review_file_name = Path(review_file).name
        logger.info(f"Reviewing file {review_file_name}...")
        for question_id, row in enumerate(question_rows, start=1):
            review_question = str(row.get("question") or "")
            review_instruction_raw = str(row.get("instruction") or "")
            review_task = _normalize_task(row.get("task"))
            review_classes = str(row.get("classes") or "")

            review_instruction = review_instruction_raw if review_instruction_raw.strip() else None

            if not review_question.strip():
                continue

            logger.info(f"reviewing question {review_question}")
            yield f"Reviewing file: {review_file_name} | question: {review_question}"

            # check if the string formatting is correct
            check_string_formatting(review_instruction=review_instruction)

            # create the query chain with a search filter and answer each question for each document
            querier.make_chain(content_folder=content_folder_name,
                               vecdb_folder=vecdb_folder_path,
                               search_filter={"filename": review_file_name},
                               qa_template_file_path_or_string=review_instruction)

            # Generate answer
            answer, sources = generate_answer(querier=querier,
                                              review_question=review_question)

            # For synthesis tasks, add document reference to support cross-document synthesis.
            answer_plus_document_reference = f"This answer is from {review_file_name}:\n {answer}"
            final_answer = answer_plus_document_reference if review_task == "synthesis" else answer

            # add resulting answer and input data to dataframe
            new_row = pd.DataFrame(
                [
                    {
                        "filename": review_file_name,
                        "question_id": question_id,
                        "question": review_question,
                        "instruction": review_instruction or "",
                        "task": review_task,
                        "classes": review_classes,
                        "answer": final_answer,
                        "sources": sources,
                    }
                ]
            )
            df_result = pd.concat([df_result, new_row], ignore_index=True)

    # Then save to JSON
    df_result = df_result.sort_values(by=["filename", "question_id"])
    with open(file=output_path, mode="w", encoding="utf8") as file:
        json.dump(df_result.to_dict(orient="records"), file, indent=2, ensure_ascii=False)


def handle_ingestion(folder_path: str, ingest_file_filter: List[str]) -> Iterator[Tuple[str, str]]:
    """
    Main loop of this module

    Parameters
    ----------
    folder_path : str
        the path of the folder with documents
    """
    # determine the list of relevant files in document folder
    # Get content folder name from path
    content_folder_name = os.path.basename(folder_path)
    # if content folder path does not exist, stop
    if not folder_path or not Path(folder_path).is_dir():
        yield _ingestion_payload("Please provide a valid folder path.", "")
        return

    documents_root = Path(folder_path).expanduser().resolve()
    all_paths = _list_document_paths_direct(documents_root)
    if not all_paths:
        yield _ingestion_payload(
            "No supported files found in the selected folder.",
            str(documents_root),
        )
        return

    selected_names = {str(name).strip().lower() for name in (ingest_file_filter or []) if str(name).strip()}
    paths = (
        [path for path in all_paths if path.name.lower() in selected_names]
        if selected_names
        else all_paths
    )

    if selected_names and not paths:
        yield _ingestion_payload(
            "No selected files were found in the folder. Refresh file selection and try again.",
            str(documents_root),
        )
        return

    # get relevant models
    confidential = False
    ut = _get_utils_module()
    llm_provider, llm_model, embeddings_provider, embeddings_model = ut.get_relevant_models(summary=False,
                                                                                            private=confidential)
    # created vector db path
    vecdb_folder_path = ut.create_vectordb_path(content_folder_path=folder_path,
                                                embeddings_provider=embeddings_provider,
                                                embeddings_model=embeddings_model)

    # create output folder with timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%Hhour_%Mmin_%Ssec")
    os.mkdir(os.path.join(folder_path, f"review/{timestamp}"))

    # path of file with review questions
    question_list_path = os.path.join(folder_path, "review", "questions.json")

    # copy the question list file to the output folder
    destination_path = os.path.join(folder_path, f"review/{timestamp}", "questions.json")
    shutil.copy(question_list_path, destination_path)
    # ingest documents if documents in source folder path are not ingested yet
    ingest_or_load_documents(content_folder_name=content_folder_name,
                             content_folder_path=folder_path,
                             vecdb_folder_path=vecdb_folder_path)

    # write settings to file
    output_path_settings = os.path.join(
        folder_path, f"review/{timestamp}", "settings.txt"
    )
    write_settings(input_path=folder_path,
                   confidential=confidential,
                   output_path=output_path_settings)

    # Create answers and store them in the file specified by output_path
    output_path_review = os.path.join(
        folder_path, f"review/{timestamp}", "answers.json"
    )

    # create instance of Querier once
    querier = Querier(llm_provider=llm_provider,
                      llm_model=llm_model,
                      embeddings_provider=embeddings_provider,
                      embeddings_model=embeddings_model)

    status_lines: List[str] = []
    for progress_message in create_answers_for_folder(
        question_list_path=question_list_path,
        review_files=paths,
        content_folder_name=content_folder_name,
        querier=querier,
        vecdb_folder_path=vecdb_folder_path,
        output_path=output_path_review,
    ):
        status_lines.append(progress_message)
        yield _ingestion_payload("\n".join(status_lines), str(documents_root))

    completion_message = "Successfully reviewed the documents."
    logger.info(completion_message)
    status_lines.append(completion_message)
    yield _ingestion_payload("\n".join(status_lines), str(documents_root))



################# User Interface #################
with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown("# ChatPBL DocReviewer")
    gr.Markdown("## Step 1: Select documents to review from a folder of choice")
    active_folder = gr.State("")

    with gr.Group():
        folder_path_input = gr.Textbox(label="Document folder", placeholder="C:\\path\\to\\documents")
        ingest_file_filter = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="Files to process",
            info="Select one or more files. Leave empty to process all supported files.",
            elem_id="ingest-file-filter",
        )

    gr.Markdown("## Step 2: Load or update your review questions")
    with gr.Row():
        row_selector = gr.Dropdown(
            label="row",
            choices=[],
            value=None,
            interactive=True,
            scale=1,
        )
        question_input = gr.Textbox(label="question", lines=2, scale=7)
        task_input = gr.Radio(
            label="task",
            choices=list(TASK_OPTIONS),
            value="answer",
            interactive=True,
            scale=2,
        )
    instruction_input = gr.Textbox(label="instruction", lines=6)
    classes_input = gr.Textbox(label="classes", lines=2)

    with gr.Row():
        add_question_row_btn: Any = gr.Button(value="Add row", variant="primary", elem_id="add-row-btn")
        update_question_row_btn: Any = gr.Button(value="Update row", variant="secondary")
        delete_question_row_btn: Any = gr.Button(value="Delete row", variant="stop")

    questions_table = gr.Dataframe(
        headers=TABLE_COLUMNS,
        column_widths=REVIEW_COLUMN_WIDTHS,
        datatype=["str", "str", "str", "str", "str"],
        value=[],
        row_count=1,
        row_limits=(1, 1),
        label="Review Questions (output)",
        elem_id="questions-table",
        wrap=True,
        interactive=False,
        static_columns=[0],
    )

    save_questions_btn: Any = gr.Button(value="Save Question list", variant="secondary")

    gr.Markdown("## Step 3: Query selected files with the question list")
    go_btn: Any = gr.Button(value="GO", variant="primary", elem_id="go-btn")

    status_messages = gr.Textbox(label="Status", interactive=False, lines=16)

    folder_path_input.submit(
        fn=refresh_ingest_file_filter,
        inputs=[folder_path_input],
        outputs=[ingest_file_filter],
    )

    folder_path_input.submit(
        fn=load_questions_table,
        inputs=[folder_path_input],
        outputs=[questions_table],
    )

    folder_path_input.submit(
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

    row_selector.change(
        fn=populate_form_from_selected_row,
        inputs=[questions_table, row_selector],
        outputs=[
            question_input,
            task_input,
            instruction_input,
            classes_input,
        ],
    )

    add_question_row_btn.click(
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

    update_question_row_btn.click(
        fn=update_question_row_from_form,
        inputs=[
            questions_table,
            row_selector,
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

    delete_question_row_btn.click(
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

    save_questions_btn.click(
        fn=save_and_reload_questions_table,
        inputs=[questions_table, folder_path_input],
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

    go_btn.click(
        fn=handle_ingestion,
        inputs=[folder_path_input, ingest_file_filter],
        outputs=[status_messages, active_folder],
    )


if __name__ == "__main__":
    # demo.queue()
    demo.launch(inbrowser=True, pwa=True, css=TABLE_CSS)
