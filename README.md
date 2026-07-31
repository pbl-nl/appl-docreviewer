# appl-docreviewer

appl-docreviewer is a document review application that runs a structured list of questions against a folder of documents using retrieval-augmented generation (RAG).

It is designed for repeatable, batch-style analysis where you want to compare multiple documents with the same question set and store all outputs for audit and follow-up.

## Purpose

The application helps you:

- answer the same questions for many documents in a consistent way
- classify answers into predefined classes (for example policy categories)
- optionally synthesize answers across documents for a question
- persist outputs and run settings for reproducibility

Typical use case:

1. Provide a folder with policy or report documents.
2. Provide a question list with question prompts and optional synthesis prompts.
3. Run the review in the UI.
4. Inspect generated answers and summaries in timestamped output folders.

## How It Works

At a high level, the pipeline is:

1. Ingest documents into a Chroma vector store (or reuse existing store if compatible with current settings).
2. For each selected document and each question:
	 - retrieve relevant chunks from the vector store
	 - run the question with the configured LLM
	 - store answer and source excerpts
3. If a synthesis template exists for a question, generate a cross-document synthesis.
4. Write all artifacts to a timestamped review folder.

## Main Features

- Gradio UI for end-to-end workflow
- Supported input formats: `.pdf`, `.docx`, `.html`, `.md`, `.txt`
- Question list editor in the UI (add/remove rows and save)
- Per-file filtering in the UI (process all files or only selected ones)
- Configurable model providers:
	- LLM: `azureopenai`, `openai`, `huggingface`, `ollama`
	- Embeddings: `azureopenai`, `openai`, `huggingface`, `ollama`
- Optional reranking (FlashRank)
- Timestamped outputs for traceability

## Repository Overview

- `app.py`: main Gradio application and orchestration flow
- `review.py`: script-style review workflow utilities
- `ingest/`: parsing, chunking, embedding, vector store creation
- `query/`: retriever and LLM query chain creation
- `prompts/`: prompt templates used by retrieval/querying
- `docs/`: example datasets and generated review artifacts
- `tests/`: ingestion/query tests
- `settings.py`: active runtime settings
- `settings_template.py`: template settings to start from

## Prerequisites

- Python 3.13 (recommended in this repository)
- Windows PowerShell (commands below use PowerShell examples)
- Access to the model provider you configure in `settings.py`

Notes:

- `.docx` parsing uses conversion to PDF via `docx2pdf` and may require a local Microsoft Word installation.
- The repository includes many dependencies; installation can take several minutes.

## Installation

1. Create and activate a virtual environment.
2. Install dependencies.
3. Configure settings and environment variables.

PowerShell example:

```powershell
python -m venv venv313
.\venv313\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

### 1) Prepare settings

If needed, initialize your settings file from template:

```powershell
Copy-Item settings_template.py settings.py
```

Review and update in `settings.py`:

- `ENVLOC`: folder that contains your `.env` file
- `LLM_PROVIDER` and `LLM_MODEL`
- `EMBEDDINGS_PROVIDER` and `EMBEDDINGS_MODEL`
- Azure deployment maps if using Azure OpenAI:
	- `AZURE_LLM_DEPLOYMENT_MAP`
	- `AZURE_EMBEDDING_DEPLOYMENT_MAP`
- Retrieval/chunking controls such as:
	- `RETRIEVER_TYPE`
	- `TEXT_SPLITTER_METHOD`, `CHUNK_SIZE`, `CHUNK_OVERLAP`
	- `TEXT_SPLITTER_METHOD_CHILD`, `CHUNK_SIZE_CHILD`, `CHUNK_OVERLAP_CHILD`
	- `CHUNK_K`, `SEARCH_TYPE`, `SCORE_THRESHOLD`
	- `RERANK`, `RERANK_MODEL`

### 2) Prepare `.env`

The app loads environment variables from:

`<ENVLOC>/.env`

Add variables required by your provider selection. Common examples:

- Azure OpenAI:
	- `AZURE_OPENAI_API_KEY`
- OpenAI:
	- `OPENAI_API_KEY`
- Hugging Face:
	- `HUGGINGFACEHUB_API_TOKEN`

If you use `azureopenai`, ensure `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_VERSION` in `settings.py` are valid.

## Input Folder Requirements

Select a folder that contains:

- the documents to process (currently direct files in that folder)
- `review/questions.json` question list file

Example structure:

```text
your_folder/
	document1.pdf
	document2.txt
	review/
		questions.json
```

During processing, the app creates/uses:

- `vector_stores/` for embeddings/chunks
- `review/<timestamp>/` for outputs

## Question List Format

The UI saves and loads `review/questions.json` with rows containing:

- `question`
- `instruction`
- `synthesis template`
- `classification`
- `classes`

Template placeholders:

- Instruction must include `{question}` and `{context}` when provided.
- Synthesis template must include `{question}` and `{answer_string}` when provided.

If these placeholders are missing, the run will fail validation for that row.

## Run The Application

From repository root:

```powershell
python app.py
```

This launches the Gradio app in your browser.

## Using The UI

1. Enter the document folder path in the `Document folder` field.
2. Optionally select specific files in `Files to process`.
3. Review or edit the question table.
4. Click `Save Question list` if you changed questions.
5. Click `Query selected files with the question list`.
6. Monitor progress in the status box.

## Outputs

For each run, output is written to:

`<document_folder>/review/<YYYY_MM_DD_HHhour_MMmin_SSsec>/`

Artifacts include:

- `questions.json` (snapshot used in the run)
- `settings.txt` (run settings snapshot)
- `answers.json` (answers per file and question)
- `answers_summary.json` (only when synthesis templates are present)

Vector store location is derived from settings and stored under:

`<document_folder>/vector_stores/<retriever_provider_model_splitter...>/`

If the same configuration is reused, ingestion can reuse and sync the existing vector store.

## Testing

Unit/integration tests are available in `tests/test_ingest_and_query.py`.

Run:

```powershell
python -m unittest tests/test_ingest_and_query.py
```

Note: Some tests rely on environment-specific paths and provider credentials and may require adaptation for your machine.

## Troubleshooting

- Error: missing API key
	- Verify your `.env` exists at `<ENVLOC>/.env` and contains the correct keys.
- No files are processed
	- Ensure files are in the selected folder root and use supported extensions.
- Questions not loaded
	- Ensure `review/questions.json` exists under the selected folder.
- Placeholder validation errors
	- Verify required placeholders in question/synthesis templates.
- `.docx` parsing issues
	- Ensure Microsoft Word is installed and available for `docx2pdf` conversion.
- Slow first run
	- Initial ingestion, embedding, and model warm-up can take time; later runs are faster when vector stores are reused.

## License

See `LICENSE`.


