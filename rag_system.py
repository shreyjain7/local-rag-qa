"""
Local RAG-Based Document Q&A System using Ollama
Ingests PDFs, creates embeddings with nomic-embed-text, and answers queries with Llama 3
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalRAGSystem:
    def __init__(self, 
                 vector_store_path: str = "vector_store",
                 embedding_model: str = "nomic-embed-text",
                 query_model: str = "llama2"):
        """
        Initialize the Local RAG System
        
        Args:
            vector_store_path: Path to store Chroma DB
            embedding_model: Ollama model for embeddings (nomic-embed-text)
            query_model: Ollama model for Q&A (llama2, llama3, etc.)
        """
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.query_model = query_model
        
        # Initialize Chroma DB with persistent storage
        self.chroma_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=vector_store_path,
            anonymized_telemetry=False
        )
        
        self.client = chromadb.Client(self.chroma_settings)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized RAG System with embedding model: {embedding_model}, query model: {query_model}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page_num, page in enumerate(reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            logger.info(f"Extracted text from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i:i + chunk_size])
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Ollama"""
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def ingest_pdf(self, pdf_path: str, document_name: str = None) -> bool:
        """Ingest a PDF document into the vector store"""
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            if document_name is None:
                document_name = Path(pdf_path).stem
            
            logger.info(f"Ingesting PDF: {document_name}")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(f"No text extracted from {pdf_path}")
                return False
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Generate embeddings and store
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.get_embedding(chunk)
                    chunk_id = f"{document_name}_chunk_{i}"
                    
                    self.collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "source": document_name,
                            "chunk_index": i,
                            "document_path": pdf_path
                        }]
                    )
                    
                    if (i + 1) % 5 == 0:
                        logger.info(f"Processed {i + 1}/{len(chunks)} chunks from {document_name}")
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    continue
            
            logger.info(f"Successfully ingested {document_name}")
            return True
        except Exception as e:
            logger.error(f"Error ingesting PDF: {str(e)}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query"""
        try:
            query_embedding = self.get_embedding(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=num_results
            )
            
            relevant_chunks = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    relevant_chunks.append({
                        'content': doc,
                        'source': results['metadatas'][0][i]['source'] if results['metadatas'] else 'Unknown',
                        'distance': results['distances'][0][i] if results['distances'] else 0
                    })
            
            return relevant_chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    def answer_query(self, query: str, num_context_chunks: int = 3) -> str:
        """Answer a query using retrieval and generation"""
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query, num_context_chunks)
            
            if not relevant_chunks:
                return "No relevant information found in the documents."
            
            # Build context
            context = "\n---\n".join([
                f"Source: {chunk['source']}\n{chunk['content']}" 
                for chunk in relevant_chunks
            ])
            
            # Generate answer using Ollama
            prompt = f"""Based on the following context, answer the user's question:

Context:
{context}

Question: {query}

Answer:"""
            
            logger.info(f"Generating answer for query: {query}")
            
            response = ollama.generate(
                model=self.query_model,
                prompt=prompt,
                stream=False
            )
            
            return response['response']
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def load_documents_from_folder(self, folder_path: str) -> Dict[str, bool]:
        """Load all PDF files from a folder"""
        results = {}
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
        
        for pdf_file in pdf_files:
            success = self.ingest_pdf(str(pdf_file), pdf_file.stem)
            results[pdf_file.name] = success
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection.name,
                'embedding_model': self.embedding_model,
                'query_model': self.query_model
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}


def main():
    """Main function to demonstrate RAG system"""
    # Initialize system
    rag_system = LocalRAGSystem(
        vector_store_path="vector_store",
        embedding_model="nomic-embed-text",
        query_model="llama2"
    )
    
    # Load documents from the documents folder
    documents_folder = "documents"
    if os.path.exists(documents_folder):
        logger.info(f"Loading documents from {documents_folder}")
        results = rag_system.load_documents_from_folder(documents_folder)
        
        for filename, success in results.items():
            status = "✓ Ingested" if success else "✗ Failed"
            logger.info(f"{status}: {filename}")
    else:
        logger.warning(f"Documents folder not found: {documents_folder}")
    
    # Print collection stats
    stats = rag_system.get_collection_stats()
    logger.info(f"Collection stats: {stats}")
    
    # Interactive Q&A loop
    print("\n" + "="*60)
    print("Local RAG-Based Document Q&A System")
    print("="*60)
    print("Type 'exit' to quit, 'stats' to see collection info\n")
    
    while True:
        query = input("Ask a question: ").strip()
        
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        if query.lower() == 'stats':
            stats = rag_system.get_collection_stats()
            print(f"\nCollection Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
            continue
        
        if not query:
            continue
        
        print("\nRetrieving relevant documents...")
        answer = rag_system.answer_query(query)
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
