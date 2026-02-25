"""
Document processing utilities for loading and chunking documents
"""

import io
from typing import List, Dict
from PyPDF2 import PdfReader
from docx import Document
import config


class DocumentProcessor:
    """Handles loading and processing of various document types"""
    
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, 
                 chunk_overlap: int = config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, file) -> str:
        """Extract text from PDF file using PyPDF2"""
        try:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def load_docx(self, file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def load_txt(self, file) -> str:
        """Extract text from TXT file"""
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
    
    def load_document(self, file, filename: str) -> str:
        """Load document based on file extension"""
        if filename.endswith('.pdf'):
            return self.load_pdf(file)
        elif filename.endswith('.docx'):
            return self.load_docx(file)
        elif filename.endswith('.txt'):
            return self.load_txt(file)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Create chunk with metadata
            chunk_data = {
                'text': chunk,
                'metadata': metadata or {}
            }
            chunks.append(chunk_data)
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            # Avoid infinite loop if chunk_overlap >= chunk_size
            if start <= end - self.chunk_size:
                start = end
        
        return chunks
    
    def process_document(self, file, filename: str) -> List[Dict]:
        """Load and chunk a document"""
        text = self.load_document(file, filename)
        metadata = {
            'filename': filename,
            'source': filename
        }
        chunks = self.chunk_text(text, metadata)
        return chunks
