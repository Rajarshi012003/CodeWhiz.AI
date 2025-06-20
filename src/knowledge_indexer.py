import json
import os
from typing import Dict, List, Any, Optional
import logging

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeIndexer:
    """
    Handles indexing and retrieval of DSA knowledge.
    """
    
    def __init__(self, knowledge_base_dir: str, db_directory: str):
        """
        Initialize the knowledge indexer.
        
        Args:
            knowledge_base_dir: Directory containing knowledge base JSON files
            db_directory: Directory to store the vector database
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.db_directory = db_directory
        
        # Initialize embedding function
        try:
            # Use sentence-transformers for better semantic understanding
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='all-MiniLM-L6-v2'
            )
        except Exception as e:
            logger.error(f"Error initializing embedding function: {str(e)}")
            raise
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(path=db_directory)
            logger.info(f"ChromaDB client initialized with database at {db_directory}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
        
        # Create or get collections
        self.algorithms_collection = self._get_or_create_collection("algorithms")
        self.concepts_collection = self._get_or_create_collection("concepts")
        self.problems_collection = self._get_or_create_collection("problems")
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        try:
            return self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            logger.error(f"Error creating collection {name}: {str(e)}")
            raise
    
    def index_knowledge_base(self) -> None:
        """Index all knowledge base files."""
        # Index algorithms
        algorithms_file = os.path.join(self.knowledge_base_dir, "algorithms.json")
        if os.path.exists(algorithms_file):
            self._index_algorithms(algorithms_file)
        
        # Index concepts
        concepts_file = os.path.join(self.knowledge_base_dir, "concepts.json")
        if os.path.exists(concepts_file):
            self._index_concepts(concepts_file)
        
        # Index problems
        problems_file = os.path.join(self.knowledge_base_dir, "problems.json")
        if os.path.exists(problems_file):
            self._index_problems(problems_file)
            
        logger.info("Knowledge base indexing completed")
    
    def _index_algorithms(self, file_path: str) -> None:
        """
        Index algorithms from a JSON file.
        
        Args:
            file_path: Path to the algorithms JSON file
        """
        try:
            with open(file_path, 'r') as f:
                algorithms = json.load(f)
            
            # Clear existing data
            try:
                # Get all existing IDs first
                existing_data = self.algorithms_collection.get()
                if existing_data and 'ids' in existing_data and existing_data['ids']:
                    self.algorithms_collection.delete(ids=existing_data['ids'])
                else:
                    # If collection is empty, no need to delete
                    pass
            except Exception as e:
                logger.warning(f"Could not clear existing algorithms data: {str(e)}")
            
            # Prepare data for indexing
            ids = []
            documents = []
            metadatas = []
            
            for algo_id, algo_data in algorithms.items():
                # Create a comprehensive document for each algorithm
                doc = f"Algorithm: {algo_data.get('name', algo_id)}\n"
                doc += f"Category: {algo_data.get('category', '')}\n"
                doc += f"Description: {algo_data.get('description', '')}\n"
                doc += f"Time Complexity: {algo_data.get('time_complexity', '')}\n"
                doc += f"Space Complexity: {algo_data.get('space_complexity', '')}\n"
                
                if "pseudocode" in algo_data:
                    doc += f"Pseudocode:\n{algo_data['pseudocode']}\n"
                
                if "cpp_implementation" in algo_data:
                    doc += f"C++ Implementation:\n{algo_data['cpp_implementation']}\n"
                
                # Add to lists
                ids.append(algo_id)
                documents.append(doc)
                metadatas.append({
                    "type": "algorithm",
                    "name": algo_data.get("name", algo_id),
                    "category": algo_data.get("category", "")
                })
                
                # Also index key components separately for more granular retrieval
                if "pseudocode" in algo_data and algo_data["pseudocode"]:
                    ids.append(f"{algo_id}_pseudocode")
                    documents.append(f"Pseudocode for {algo_data.get('name', algo_id)}:\n{algo_data['pseudocode']}")
                    metadatas.append({
                        "type": "algorithm_pseudocode",
                        "name": algo_data.get("name", algo_id),
                        "category": algo_data.get("category", "")
                    })
                
                if "cpp_implementation" in algo_data and algo_data["cpp_implementation"]:
                    ids.append(f"{algo_id}_cpp")
                    documents.append(f"C++ Implementation of {algo_data.get('name', algo_id)}:\n{algo_data['cpp_implementation']}")
                    metadatas.append({
                        "type": "algorithm_cpp",
                        "name": algo_data.get("name", algo_id),
                        "category": algo_data.get("category", "")
                    })
            
            # Add to collection
            if documents:
                self.algorithms_collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"Indexed {len(algorithms)} algorithms with {len(documents)} total documents")
            
        except Exception as e:
            logger.error(f"Error indexing algorithms: {str(e)}")
            raise
    
    def _index_concepts(self, file_path: str) -> None:
        """
        Index concepts from a JSON file.
        
        Args:
            file_path: Path to the concepts JSON file
        """
        try:
            with open(file_path, 'r') as f:
                concepts = json.load(f)
            
            # Clear existing data
            try:
                # Get all existing IDs first
                existing_data = self.concepts_collection.get()
                if existing_data and 'ids' in existing_data and existing_data['ids']:
                    self.concepts_collection.delete(ids=existing_data['ids'])
                else:
                    # If collection is empty, no need to delete
                    pass
            except Exception as e:
                logger.warning(f"Could not clear existing concepts data: {str(e)}")
            
            # Prepare data for indexing
            ids = []
            documents = []
            metadatas = []
            
            for concept_id, concept_data in concepts.items():
                # Create a comprehensive document for each concept
                doc = f"Concept: {concept_data.get('name', concept_id)}\n"
                doc += f"Description: {concept_data.get('description', '')}\n"
                
                if "key_points" in concept_data:
                    doc += "Key Points:\n"
                    for point in concept_data["key_points"]:
                        doc += f"- {point}\n"
                
                if "examples" in concept_data:
                    doc += f"Examples:\n{concept_data['examples']}\n"
                
                # Add to lists
                ids.append(concept_id)
                documents.append(doc)
                metadatas.append({
                    "type": "concept",
                    "name": concept_data.get("name", concept_id)
                })
                
                # Also index examples separately if they exist
                if "examples" in concept_data and concept_data["examples"]:
                    ids.append(f"{concept_id}_examples")
                    documents.append(f"Examples of {concept_data.get('name', concept_id)}:\n{concept_data['examples']}")
                    metadatas.append({
                        "type": "concept_examples",
                        "name": concept_data.get("name", concept_id)
                    })
            
            # Add to collection
            if documents:
                self.concepts_collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"Indexed {len(concepts)} concepts with {len(documents)} total documents")
            
        except Exception as e:
            logger.error(f"Error indexing concepts: {str(e)}")
            raise
    
    def _index_problems(self, file_path: str) -> None:
        """
        Index problems from a JSON file.
        
        Args:
            file_path: Path to the problems JSON file
        """
        try:
            with open(file_path, 'r') as f:
                problems = json.load(f)
            
            # Clear existing data
            try:
                # Get all existing IDs first
                existing_data = self.problems_collection.get()
                if existing_data and 'ids' in existing_data and existing_data['ids']:
                    self.problems_collection.delete(ids=existing_data['ids'])
                else:
                    # If collection is empty, no need to delete
                    pass
            except Exception as e:
                logger.warning(f"Could not clear existing problems data: {str(e)}")
            
            # Prepare data for indexing
            ids = []
            documents = []
            metadatas = []
            
            for problem_id, problem_data in problems.items():
                # Create a comprehensive document for each problem
                doc = f"Problem: {problem_data.get('name', problem_id)}\n"
                doc += f"Difficulty: {problem_data.get('difficulty', '')}\n"
                doc += f"Description: {problem_data.get('description', '')}\n"
                
                if "constraints" in problem_data:
                    doc += "Constraints:\n"
                    for constraint in problem_data["constraints"]:
                        doc += f"- {constraint}\n"
                
                if "examples" in problem_data:
                    doc += "Examples:\n"
                    for example in problem_data["examples"]:
                        doc += f"Input: {example.get('input', '')}\n"
                        doc += f"Output: {example.get('output', '')}\n"
                        if "explanation" in example:
                            doc += f"Explanation: {example['explanation']}\n"
                
                if "related_concepts" in problem_data:
                    doc += f"Related Concepts: {', '.join(problem_data['related_concepts'])}\n"
                
                if "solution_approach" in problem_data:
                    doc += f"Solution Approach: {problem_data['solution_approach']}\n"
                
                if "cpp_solution" in problem_data:
                    doc += f"C++ Solution:\n{problem_data['cpp_solution']}\n"
                
                # Add to lists
                ids.append(problem_id)
                documents.append(doc)
                metadatas.append({
                    "type": "problem",
                    "name": problem_data.get("name", problem_id),
                    "difficulty": problem_data.get("difficulty", ""),
                    "related_concepts": ", ".join(problem_data.get("related_concepts", []))
                })
                
                # Also index solution approach and code separately
                if "solution_approach" in problem_data and problem_data["solution_approach"]:
                    ids.append(f"{problem_id}_approach")
                    documents.append(f"Solution Approach for {problem_data.get('name', problem_id)}:\n{problem_data['solution_approach']}")
                    metadatas.append({
                        "type": "problem_approach",
                        "name": problem_data.get("name", problem_id),
                        "difficulty": problem_data.get("difficulty", "")
                    })
                
                if "cpp_solution" in problem_data and problem_data["cpp_solution"]:
                    ids.append(f"{problem_id}_cpp")
                    documents.append(f"C++ Solution for {problem_data.get('name', problem_id)}:\n{problem_data['cpp_solution']}")
                    metadatas.append({
                        "type": "problem_cpp",
                        "name": problem_data.get("name", problem_id),
                        "difficulty": problem_data.get("difficulty", "")
                    })
            
            # Add to collection
            if documents:
                self.problems_collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info(f"Indexed {len(problems)} problems with {len(documents)} total documents")
            
        except Exception as e:
            logger.error(f"Error indexing problems: {str(e)}")
            raise
    
    def query_knowledge_base(self, query: str, n_results: int = 3, filters: Optional[Dict] = None) -> Dict:
        """
        Query the knowledge base.
        
        Args:
            query: The query string
            n_results: Number of results to return
            filters: Optional filters to apply
            
        Returns:
            Dictionary containing query results
        """
        try:
            # Query each collection
            algo_results = self._query_collection(self.algorithms_collection, query, n_results, filters)
            concept_results = self._query_collection(self.concepts_collection, query, n_results, filters)
            problem_results = self._query_collection(self.problems_collection, query, n_results, filters)
            
            # Combine results
            all_documents = algo_results["documents"] + concept_results["documents"] + problem_results["documents"]
            all_metadatas = algo_results["metadatas"] + concept_results["metadatas"] + problem_results["metadatas"]
            all_distances = algo_results["distances"] + concept_results["distances"] + problem_results["distances"]
            
            # Sort by distance (lower is better)
            sorted_results = sorted(zip(all_documents, all_metadatas, all_distances), key=lambda x: x[2])
            
            # Take top n_results
            top_documents = []
            top_metadatas = []
            top_distances = []
            
            for doc, meta, dist in sorted_results[:n_results]:
                top_documents.append(doc)
                top_metadatas.append(meta)
                top_distances.append(dist)
            
            return {
                "documents": [top_documents],
                "metadatas": [top_metadatas],
                "distances": [top_distances]
            }
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }
    
    def _query_collection(self, collection, query: str, n_results: int, filters: Optional[Dict] = None) -> Dict:
        """
        Query a specific collection.
        
        Args:
            collection: The ChromaDB collection
            query: The query string
            n_results: Number of results to return
            filters: Optional filters to apply
            
        Returns:
            Dictionary containing query results
        """
        try:
            # Adjust n_results to ensure we get enough results from each collection
            adjusted_n_results = max(n_results, 2)
            
            # Query the collection
            results = collection.query(
                query_texts=[query],
                n_results=adjusted_n_results,
                where=filters
            )
            
            # Extract results
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            return {
                "documents": documents,
                "metadatas": metadatas,
                "distances": distances
            }
            
        except Exception as e:
            logger.error(f"Error querying collection: {str(e)}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": []
            }
    
    def search_by_topic(self, topic: str, content_type: Optional[str] = None, n_results: int = 5) -> Dict:
        """
        Search the knowledge base by topic.
        
        Args:
            topic: The topic to search for
            content_type: Optional content type filter (algorithm, concept, problem)
            n_results: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        # Build filters based on content type
        filters = {}
        if content_type:
            filters = {"type": content_type}
        
        # Create a more comprehensive query
        query = f"{topic} definition explanation examples"
        
        # Query the knowledge base
        results = self.query_knowledge_base(query, n_results, filters)
        
        return results

# For testing purposes
if __name__ == "__main__":
    # Initialize the knowledge indexer
    indexer = KnowledgeIndexer(
        knowledge_base_dir="data/knowledge_base",
        db_directory="chroma_db"
    )
    
    # Index the knowledge base
    indexer.index_knowledge_base()
    
    # Test querying
    query = "How does quicksort work?"
    results = indexer.query_knowledge_base(query)
    
    print(f"Query: {query}")
    print(f"Results: {len(results['documents'][0])}")
    
    for i, doc in enumerate(results['documents'][0]):
        print(f"\nResult {i+1}:")
        print(f"Document: {doc[:200]}...")  # Print first 200 chars
        print(f"Metadata: {results['metadatas'][0][i]}")
        print(f"Distance: {results['distances'][0][i]}") 