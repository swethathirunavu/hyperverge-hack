import PyMuPDF  # fitz
import docx
import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

class DocumentProcessor:
    def __init__(self, chunk_size: int = 8000, overlap: int = 500):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process uploaded file and return chunks with metadata"""
        file_extension = file_path.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self._process_pdf(file_path)
        elif file_extension in ['docx', 'doc']:
            return self._process_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF and create chunks"""
        chunks = []
        
        try:
            doc = PyMuPDF.open(file_path)
            
            # Check document size
            total_pages = len(doc)
            if total_pages > 20:
                st.warning(f"Large PDF detected ({total_pages} pages). Processing first 20 pages only.")
                total_pages = 20
            
            full_text = ""
            page_texts = {}
            
            # Extract text from each page
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_texts[page_num] = page_text
                full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            doc.close()
            
            # Split into chunks
            text_chunks = self.text_splitter.split_text(full_text)
            
            # Create chunks with metadata
            for i, chunk_text in enumerate(text_chunks):
                # Find which pages this chunk covers
                pages_covered = self._find_pages_in_chunk(chunk_text, page_texts)
                
                chunk = {
                    'index': i,
                    'text': chunk_text,
                    'source_type': 'pdf',
                    'pages_covered': pages_covered,
                    'chunk_size': len(chunk_text),
                    'metadata': {
                        'total_pages': total_pages,
                        'chunk_id': f"chunk_{i}",
                        'file_type': 'pdf'
                    }
                }
                chunks.append(chunk)
                
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
        
        return chunks
    
    def _process_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from DOCX and create chunks"""
        chunks = []
        
        try:
            doc = docx.Document(file_path)
            full_text = ""
            paragraph_texts = []
            
            # Extract text from paragraphs
            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    paragraph_texts.append(paragraph.text.strip())
                    full_text += paragraph.text.strip() + "\n\n"
            
            # Check document size (approximate)
            estimated_pages = len(full_text) // 2000  # Rough estimate
            if estimated_pages > 20:
                st.warning(f"Large document detected (~{estimated_pages} pages). Processing first portion only.")
                # Limit text to approximately 20 pages worth
                full_text = full_text[:40000]
            
            # Split into chunks
            text_chunks = self.text_splitter.split_text(full_text)
            
            # Create chunks with metadata
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    'index': i,
                    'text': chunk_text,
                    'source_type': 'docx',
                    'paragraphs_covered': self._find_paragraphs_in_chunk(chunk_text, paragraph_texts),
                    'chunk_size': len(chunk_text),
                    'metadata': {
                        'total_paragraphs': len(paragraph_texts),
                        'chunk_id': f"chunk_{i}",
                        'file_type': 'docx'
                    }
                }
                chunks.append(chunk)
                
        except Exception as e:
            raise Exception(f"Error processing DOCX: {str(e)}")
        
        return chunks
    
    def _find_pages_in_chunk(self, chunk_text: str, page_texts: Dict[int, str]) -> List[int]:
        """Find which pages are represented in a chunk"""
        pages = []
        
        # Look for page markers in chunk
        for page_num, page_text in page_texts.items():
            # If chunk contains significant portion of page text, include page
            if len(page_text.strip()) > 0:
                # Check if chunk contains at least 100 characters from this page
                common_text = self._find_common_text(chunk_text, page_text)
                if len(common_text) > 100:
                    pages.append(page_num + 1)  # 1-indexed for display
        
        return pages if pages else [1]  # Default to page 1 if no matches found
    
    def _find_paragraphs_in_chunk(self, chunk_text: str, paragraph_texts: List[str]) -> List[int]:
        """Find which paragraphs are represented in a chunk"""
        paragraphs = []
        
        for i, para_text in enumerate(paragraph_texts):
            if len(para_text.strip()) > 0:
                # Check if chunk contains this paragraph
                if para_text[:100] in chunk_text:  # Check first 100 chars
                    paragraphs.append(i + 1)  # 1-indexed
        
        return paragraphs if paragraphs else [1]
    
    def _find_common_text(self, text1: str, text2: str, min_length: int = 50) -> str:
        """Find common text between two strings"""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Find longest common substring
        common = ""
        for i in range(len(text1)):
            for j in range(i + min_length, len(text1) + 1):
                substring = text1[i:j]
                if substring in text2:
                    if len(substring) > len(common):
                        common = substring
        
        return common
    
    def get_chunk_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics about chunks"""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        avg_chunk_size = sum(chunk['chunk_size'] for chunk in chunks) / total_chunks
        total_text_length = sum(chunk['chunk_size'] for chunk in chunks)
        
        # Coverage analysis
        if chunks[0]['source_type'] == 'pdf':
            pages_covered = set()
            for chunk in chunks:
                pages_covered.update(chunk.get('pages_covered', []))
            coverage = len(pages_covered)
        else:
            paragraphs_covered = set()
            for chunk in chunks:
                paragraphs_covered.update(chunk.get('paragraphs_covered', []))
            coverage = len(paragraphs_covered)
        
        return {
            'total_chunks': total_chunks,
            'average_chunk_size': round(avg_chunk_size, 2),
            'total_text_length': total_text_length,
            'source_type': chunks[0]['source_type'],
            'coverage': coverage,
            'estimated_tokens': total_text_length // 4  # Rough token estimate
        }
    
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Validate that chunks are properly formatted"""
        if not chunks:
            return False
        
        required_fields = ['index', 'text', 'source_type', 'chunk_size', 'metadata']
        
        for chunk in chunks:
            for field in required_fields:
                if field not in chunk:
                    return False
            
            # Check text content
            if not chunk['text'] or len(chunk['text'].strip()) < 50:
                return False
        
        return True
