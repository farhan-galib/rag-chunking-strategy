"""
File extraction utilities for multiple document formats
Supports: TXT, MD, PDF, DOCX, XLSX
"""

import io
from typing import Tuple
from pathlib import Path


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text from PDF file.
    
    Args:
        file_content: Raw bytes of the PDF file
        
    Returns:
        Extracted text as string
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf library is required for PDF support. Install with: pip install pypdf")
    
    try:
        pdf_reader = PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")


def extract_text_from_docx(file_content: bytes) -> str:
    """
    Extract text from DOCX file.
    
    Args:
        file_content: Raw bytes of the DOCX file
        
    Returns:
        Extracted text as string
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx library is required for DOCX support. Install with: pip install python-docx")
    
    try:
        doc = Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Also extract from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + " "
                text += "\n"
        
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading DOCX file: {str(e)}")


def extract_text_from_xlsx(file_content: bytes) -> str:
    """
    Extract text from Excel (XLSX) file.
    
    Args:
        file_content: Raw bytes of the XLSX file
        
    Returns:
        Extracted text as string
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas library is required for Excel support. Install with: pip install pandas")
    
    try:
        excel_file = io.BytesIO(file_content)
        xl_file = pd.ExcelFile(excel_file)
        
        text = ""
        # Extract from all sheets
        for sheet_name in xl_file.sheet_names:
            text += f"--- Sheet: {sheet_name} ---\n"
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            text += df.to_string() + "\n\n"
        
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading XLSX file: {str(e)}")


def extract_text_from_xls(file_content: bytes) -> str:
    """
    Extract text from Excel (XLS) file.
    
    Args:
        file_content: Raw bytes of the XLS file
        
    Returns:
        Extracted text as string
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas library is required for Excel support. Install with: pip install pandas")
    
    try:
        excel_file = io.BytesIO(file_content)
        xl_file = pd.ExcelFile(excel_file)
        
        text = ""
        # Extract from all sheets
        for sheet_name in xl_file.sheet_names:
            text += f"--- Sheet: {sheet_name} ---\n"
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            text += df.to_string() + "\n\n"
        
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading XLS file: {str(e)}")


def extract_text_from_txt(file_content: bytes) -> str:
    """
    Extract text from plain text file.
    
    Args:
        file_content: Raw bytes of the text file
        
    Returns:
        Extracted text as string
    """
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return file_content.decode('latin-1')
        except UnicodeDecodeError:
            raise ValueError("Could not decode text file. Ensure it's UTF-8 or Latin-1 encoded.")


def extract_text_from_md(file_content: bytes) -> str:
    """
    Extract text from Markdown file.
    
    Args:
        file_content: Raw bytes of the markdown file
        
    Returns:
        Extracted text as string
    """
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return file_content.decode('latin-1')
        except UnicodeDecodeError:
            raise ValueError("Could not decode markdown file. Ensure it's UTF-8 or Latin-1 encoded.")


def extract_text_from_file(file_content: bytes, file_extension: str) -> Tuple[str, bool]:
    """
    Extract text from any supported file format.
    
    Args:
        file_content: Raw bytes of the file
        file_extension: File extension (e.g., 'pdf', 'docx', 'xlsx')
        
    Returns:
        Tuple of (extracted_text, success_flag)
    """
    file_extension = file_extension.lower().strip('.')
    
    extractors = {
        'pdf': extract_text_from_pdf,
        'docx': extract_text_from_docx,
        'doc': extract_text_from_docx,  # Treat .doc as .docx (limited support)
        'xlsx': extract_text_from_xlsx,
        'xls': extract_text_from_xls,
        'xl': extract_text_from_xlsx,  # Alias for xlsx
        'txt': extract_text_from_txt,
        'md': extract_text_from_md,
    }
    
    if file_extension not in extractors:
        return None, False
    
    try:
        text = extractors[file_extension](file_content)
        return text, True
    except Exception as e:
        return str(e), False


def get_supported_file_types() -> list:
    """
    Get list of supported file extensions.
    
    Returns:
        List of supported file extensions
    """
    return ['txt', 'md', 'pdf', 'docx', 'doc', 'xlsx', 'xls', 'xl']
