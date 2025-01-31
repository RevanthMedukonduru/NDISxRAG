import logging
import pandas as pd
from celery import shared_task
from typing import Union, List
from langchain.schema import Document
from .models import UploadedFile
from Embedding.vector_store import QdrantStore

from NDISxRAG.settings import LOGGER_NAME
logger = logging.getLogger(LOGGER_NAME)

@shared_task
def process_excel_task(uploaded_file_id):
    """
    Task to process the uploaded Excel file 
    
    # Load the uploaded file
    # Process the Excel file
    """
    # Create a QdrantStore instance
    instance = QdrantStore()

    try:
        # Load the uploaded file
        uploaded_file = UploadedFile.objects.get(id=uploaded_file_id)
    except UploadedFile.DoesNotExist:
        logger.error("UploadedFile not found | ID: %s", uploaded_file_id)
        return "File not found"
    
    try:
        # Process the Excel file
        logger.info("Started processing Excel file | ID: %s | Filename: %s, TYPE: %s", uploaded_file.id, uploaded_file.file, type(uploaded_file.file))
        documents = build_semantic_documents(file_path=str(uploaded_file.file))
        uploaded_file.last_processed_step = "Excel file converted to docs"
        uploaded_file.save()
        logger.info("UploadedFile found | ID: %s | loaded n docs: %s", uploaded_file.id, len(documents))
        
        # Convert the documents to embeddings and upload them to Vector DB
        is_connected = instance.reconnect_or_open_connection()
        if not is_connected:
            raise Exception("Error connecting to Vector DB")
        is_stored = instance.ingest_data(collection_name="data", documents=documents)
        is_closed = instance.close_connection()
        logger.info("UploadedFile found | ID: %s | stored: %s | closed: %s", uploaded_file.id, is_stored, is_closed)
        
        uploaded_file.finished_processing = is_stored
        if is_stored:
            uploaded_file.last_processed_step = "Docs converted to embeddings and stored"
        uploaded_file.save()
        logger.info("UploadedFile processed | ID: %s", uploaded_file.id)
    except Exception as e:
        logger.error("Error processing Excel file | ID: %s | Error: %s", uploaded_file.id, str(e))
        uploaded_file.last_processed_step = str(e)
        uploaded_file.finished_processing = False
        return f"Error processing Excel file | ID: {uploaded_file.id} | Error: {str(e)}"

def build_semantic_documents(
    file_path: str,
    sheet_names: Union[str, int, List[Union[str, int]], None] = None,
    encoding: str = "utf-8"
) -> List[Document]:
    """
    To Generate a list of Documents from a CSV or Excel file. For Excel, can read multiple sheets.

    If an Excel file is provided:
      - If `sheet_names` is None or empty, reads **all** available sheets.
      - If `sheet_names` is a single sheet name or index, reads just that sheet.
      - If `sheet_names` is a list of sheet names/indices, reads them all.
    
    For each sheet (or CSV):
      1) Identify columns with string data (dtype==object).
      2) Combine those column values into a single text string per row.
      3) Create a Document for each row. `Document.page_content` holds
         the combined text, and `Document.metadata` includes:
           - source: file path
           - row_no: row index in that sheet
           - sheet: which sheet it came from (or '' for CSV)
    
    Args:
        file_path (str): Path to the CSV or Excel file.
        sheet_names (Union[str, int, List[str|int], None]):
            - None => read all sheets in Excel
            - str/int => read a single sheet by name/index
            - list of str/int => read multiple sheets
        encoding (str): File encoding for CSV reads.

    Returns:
        List[Document]: A combined list of Document objects across
                        all relevant sheets or just one (if CSV).
    """

    # -- Handle CSV case --
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, encoding=encoding)
        return _process_single_df_to_documents(
            df=df,
            source=file_path,
            sheet_name="",  # No sheet name for CSV
        )

    # -- Handle Excel case --
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        # Figure out which sheets to read
        excel_file = pd.ExcelFile(file_path)
        all_sheet_names = excel_file.sheet_names  # All available sheets in the file

        if sheet_names is None or sheet_names == "":
            # Read *all* sheets
            sheets_to_process = all_sheet_names
        elif isinstance(sheet_names, (str, int)):
            # Single sheet (by name or index)
            sheets_to_process = [sheet_names]
        else:
            # Assume it's a list of sheet names or indices
            sheets_to_process = sheet_names

        # Convert any int sheet indexes to actual sheet names
        normalized_sheets = []
        for s in sheets_to_process:
            if isinstance(s, int):
                # Validate index within range
                if s < 0 or s >= len(all_sheet_names):
                    raise ValueError(f"Sheet index {s} out of range for file {file_path}")
                normalized_sheets.append(all_sheet_names[s])
            else:
                # Just a string sheet name
                if s not in all_sheet_names:
                    raise ValueError(f"Sheet name '{s}' not found in file {file_path}")
                normalized_sheets.append(s)

        # Process each sheet, accumulate results
        all_documents: List[Document] = []
        for sheet_name in normalized_sheets:
            df_sheet = pd.read_excel(file_path, sheet_name=sheet_name)
            docs_in_sheet = _process_single_df_to_documents(
                df=df_sheet,
                source=file_path,
                sheet_name=sheet_name,
            )
            all_documents.extend(docs_in_sheet)

        return all_documents

    else:
        raise ValueError("Unsupported file format. Must be CSV, XLS, or XLSX.")


def _process_single_df_to_documents(
    df: pd.DataFrame,
    source: str,
    sheet_name: str,
) -> List[Document]:
    """
    Internal helper that converts a single Pandas DataFrame into a list of Documents.
    Only columns with dtype=object (string columns) are used to build the text.
    """
    # Identify all columns with string (object) type
    str_columns = df.select_dtypes(include=["object"]).columns
    if not len(str_columns):
        return []

    # Convert them to string (in case there are NaNs) and combine them per row
    combined_texts = (
        df[str_columns]
        .fillna("")
        .astype(str)
        .apply(
            lambda row: " ".join(f"{col}: {row[col]}" for col in row.index),
            axis=1,
        )
    )

    # Build Document objects with metadata
    documents = []
    for idx, text in enumerate(combined_texts):
        metadata = {"source": source, "row_no": idx, "sheet": sheet_name}
        documents.append(Document(page_content=text, metadata=metadata))

    return documents
