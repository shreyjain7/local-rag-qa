"""
Simplified Local RAG-Based Document Q&A System using Ollama
Works without chromadb - stores embeddings in JSON files
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleRAGSystem:
    def __init__(self, 
                 vector_store_path: str = "vector_store",
                 embedding_model: str = "nomic-embed-text",
                 query_model: str = "llama2"):
        """
        Initialize the Simple Local RAG System
        
        Args:
            vector_store_path: Path to store embeddings as JSON
            embedding_model: Ollama model for embeddings
            query_model: Ollama model for Q&A
        """
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.query_model = query_model
        self.embeddings_file = os.path.join(vector_store_path, "embeddings.json")
        self.metadata_file = os.path.join(vector_store_path, "metadata.json")
        
        # Create vector store directory
        Path(vector_store_path).mkdir(parents=True, exist_ok=True)
        
        # Load or create embeddings
        self.embeddings = self._load_embeddings()
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized RAG System with embedding model: {embedding_model}, query model: {query_model}")
    
    def _load_embeddings(self) -> Dict[str, List[float]]:
        """Load embeddings from JSON file"""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load embeddings: {e}")
        return {}
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        return {}
    
    def _save_embeddings(self):
        """Save embeddings to JSON file"""
        with open(self.embeddings_file, 'w') as f:
            json.dump(self.embeddings, f, indent=2)
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
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
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
        magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
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
                    
                    self.embeddings[chunk_id] = embedding
                    self.metadata[chunk_id] = {
                        "source": document_name,
                        "chunk_index": i,
                        "document_path": pdf_path,
                        "text": chunk[:200] + "..." if len(chunk) > 200 else chunk
                    }
                    
                    if (i + 1) % 5 == 0:
                        logger.info(f"Processed {i + 1}/{len(chunks)} chunks from {document_name}")
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    continue
            
            # Save to disk
            self._save_embeddings()
            self._save_metadata()
            
            logger.info(f"Successfully ingested {document_name}")
            return True
        except Exception as e:
            logger.error(f"Error ingesting PDF: {str(e)}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query"""
        try:
            if not self.embeddings:
                return []
            
            query_embedding = self.get_embedding(query)
            
            # Calculate similarities
            similarities = []
            for chunk_id, embedding in self.embeddings.items():
                similarity = self._cosine_similarity(query_embedding, embedding)
                similarities.append({
                    'chunk_id': chunk_id,
                    'similarity': similarity,
                    'metadata': self.metadata.get(chunk_id, {})
                })
            
            # Sort by similarity and get top results
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = similarities[:num_results]
            
            relevant_chunks = []
            for result in top_results:
                # Reconstruct the text from metadata
                chunk_id = result['chunk_id']
                # Get the actual text by re-extracting or storing it
                relevant_chunks.append({
                    'content': result['metadata'].get('text', ''),
                    'source': result['metadata'].get('source', 'Unknown'),
                    'similarity': result['similarity'],
                    'chunk_id': chunk_id
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
            return {
                'total_chunks': len(self.embeddings),
                'collection_name': 'documents',
                'embedding_model': self.embedding_model,
                'query_model': self.query_model,
                'storage_path': self.vector_store_path
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}


def main():
    """Main function to demonstrate RAG system"""
    # Initialize system
    rag_system = SimpleRAGSystem(
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
