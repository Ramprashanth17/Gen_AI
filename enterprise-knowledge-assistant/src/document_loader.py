"""Document loading utilities for multiple formats"""

from pathlib import Path
from PyPDF2 import PdfReader

"""Document loading utilities for multiple formats"""

from pathlib import Path
from PyPDF2 import PdfReader

def load_document(file_path):
    """
    Load a document and extract its text.
    
    Args:
        file_path: Path to document file
        
    Returns:
        dict with 'text' and 'metadata' keys
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    text = ""
    doc_type = ""
    
    if ext == '.pdf':
        doc_type = "pdf"
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text()
    elif ext == '.txt':
        doc_type = "txt"
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return {
        'text': text,
        'metadata': {
            'source': file_path.name,
            'type': doc_type,
            'path': str(file_path),
            'chars': len(text)
        }
    }

def load_all_documents(folder_path):
    """Load all supported documents from a folder"""
    folder = Path(folder_path)
    all_files = list(folder.rglob("*.pdf")) + list(folder.rglob("*.txt"))
    
    documents = []
    for file_path in all_files:
        doc = load_document(file_path)
        documents.append(doc)
    
    return documents