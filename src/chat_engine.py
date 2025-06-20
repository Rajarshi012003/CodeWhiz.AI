import os
import time
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union

from src.llm_module import LLMModule
from src.knowledge_indexer import KnowledgeIndexer
from src.quiz_manager import QuizManager
from src.user_data_store import UserDataStore
from src.cpp_executor.executor import CppExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatEngine:
    """
    Main chat engine that handles user queries and integrates all components.
    """
    
    def __init__(self, 
                 llm_module: LLMModule,
                 knowledge_indexer: KnowledgeIndexer,
                 quiz_manager: QuizManager,
                 user_data_store: UserDataStore,
                 cpp_executor: CppExecutor):
        """
        Initialize the chat engine.
        
        Args:
            llm_module: The LLM module for generating responses
            knowledge_indexer: The knowledge indexer for retrieving relevant content
            quiz_manager: The quiz manager for handling quizzes
            user_data_store: The user data store for tracking user progress
            cpp_executor: The C++ executor for running code
        """
        self.llm_module = llm_module
        self.knowledge_indexer = knowledge_indexer
        self.quiz_manager = quiz_manager
        self.user_data_store = user_data_store
        self.cpp_executor = cpp_executor
        
        # Current session state
        self.current_user_id = None
        self.current_session_id = None
        self.current_quiz = None
        self.current_quiz_index = 0
        self.conversation_history = []
    
    def start_session(self, username: str) -> int:
        """
        Start a new chat session.
        
        Args:
            username: The username
            
        Returns:
            The session ID
        """
        # Get or create user
        self.current_user_id = self.user_data_store.get_or_create_user(username)
        
        # Start a new session
        self.current_session_id = self.user_data_store.start_session(self.current_user_id)
        
        # Reset conversation history
        self.conversation_history = []
        
        return self.current_session_id
    
    def end_session(self) -> None:
        """End the current session."""
        if self.current_session_id:
            self.user_data_store.end_session(self.current_session_id)
            self.current_session_id = None
            self.current_quiz = None
            self.current_quiz_index = 0
            self.conversation_history = []
    
    def process_query(self, query: str) -> Dict:
        """
        Process a user query and generate a response.
        
        Args:
            query: The user's query
            
        Returns:
            Response data
        """
        start_time = time.time()
        
        # Ensure we have a session
        if not self.current_session_id:
            return {
                "response": "Please start a session first.",
                "type": "error"
            }
        
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Determine query intent
        intent = self._determine_intent(query)
        logger.info(f"Detected intent: {intent}")
        
        # Process based on intent
        if intent == "quiz":
            response_data = self._handle_quiz_intent(query)
        elif intent == "quiz_answer":
            response_data = self._handle_quiz_answer_intent(query)
        elif intent == "code_execution":
            response_data = self._handle_code_execution_intent(query)
        elif intent == "code_explanation":
            response_data = self._handle_code_explanation_intent(query)
        else:
            # Default to question answering
            response_data = self._handle_question_intent(query)
        
        # Add response to conversation history
        self.conversation_history.append({
            "role": "assistant", 
            "content": response_data["response"]
        })
        
        # Log the interaction
        self.user_data_store.log_interaction(
            self.current_session_id,
            query,
            response_data["response"],
            response_data["type"]
        )
        
        # Add processing time
        response_data["processing_time"] = time.time() - start_time
        
        return response_data
    
    def _determine_intent(self, query: str) -> str:
        """
        Determine the intent of a user query.
        
        Args:
            query: The user's query
            
        Returns:
            The intent type
        """
        # Check if we're in the middle of a quiz
        if self.current_quiz:
            return "quiz_answer"
        
        # Check for code execution intent
        code_pattern = r"```(?:cpp|c\+\+)?(.*?)```"
        if re.search(code_pattern, query, re.DOTALL):
            return "code_execution"
        
        # Check for code explanation intent
        if "explain" in query.lower() and "code" in query.lower():
            return "code_explanation"
        
        # Check for quiz intent
        quiz_keywords = ["quiz", "test", "question", "practice"]
        if any(keyword in query.lower() for keyword in quiz_keywords):
            return "quiz"
        
        # Default to question intent
        return "question"
    
    def _handle_question_intent(self, query: str) -> Dict:
        """
        Handle a question intent.
        
        Args:
            query: The user's query
            
        Returns:
            Response data
        """
        # Check for general conversation patterns
        general_conversation_patterns = [
            "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
            "good evening", "how are you", "what's up", "nice to meet you"
        ]
        
        if query.lower().strip() in general_conversation_patterns or any(pattern in query.lower() for pattern in general_conversation_patterns):
            # Handle general conversation
            response = f"Hello! I'm your DSA (Data Structures and Algorithms) tutor. I can help you with questions about algorithms, data structures, coding problems, and more. What would you like to learn about today?"
            return {
                "response": response,
                "type": "greeting",
                "context": ""
            }
        
        # Try multiple approaches to find relevant content
        # First, try with the original query
        results = self.knowledge_indexer.query_knowledge_base(query, n_results=5)
        
        # Extract context from results
        context = "\n\n".join(results['documents'][0]) if results['documents'] and results['documents'][0] else ""
        
        # If no relevant context found, try with keyword extraction
        if not context.strip():
            # Try to extract key DSA terms from the query
            dsa_keywords = self._extract_dsa_keywords(query)
            if dsa_keywords:
                # Search using extracted keywords
                enhanced_query = f"{query} {' '.join(dsa_keywords)}"
                results = self.knowledge_indexer.query_knowledge_base(enhanced_query, n_results=5)
                context = "\n\n".join(results['documents'][0]) if results['documents'] and results['documents'][0] else ""
        
        # If still no context, try direct topic search
        if not context.strip():
            # Try to identify a specific DSA topic in the query
            topic = self._extract_topic(query)
            if topic:
                # Use the topic search which is optimized for finding specific topics
                results = self.knowledge_indexer.search_by_topic(topic, n_results=5)
                context = "\n\n".join(results['documents'][0]) if results['documents'] and results['documents'][0] else ""
        
        # If still no relevant context found, generate a more helpful response
        if not context.strip():
            # Generate a response that acknowledges the query but asks for clarification
            response = self._generate_clarification_response(query)
            return {
                "response": response,
                "type": "clarification",
                "context": ""
            }
        
        # Generate answer using LLM with the context we found
        answer = self.llm_module.generate_answer(query, context)
        
        return {
            "response": answer,
            "type": "question",
            "context": context
        }
    
    def _extract_dsa_keywords(self, query: str) -> List[str]:
        """
        Extract DSA-related keywords from a query.
        
        Args:
            query: The user's query
            
        Returns:
            List of DSA keywords
        """
        # Common DSA terms to look for
        dsa_terms = [
            # Data structures
            "array", "list", "linked list", "stack", "queue", "tree", "binary tree", 
            "binary search tree", "heap", "hash", "hash table", "graph", "trie",
            "segment tree", "fenwick tree", "suffix tree", "suffix array",
            
            # Algorithm categories
            "sort", "sorting", "search", "searching", "recursion", "recursive", 
            "dynamic programming", "dp", "greedy", "divide and conquer", "backtracking",
            "bfs", "dfs", "breadth first", "depth first", "dijkstra", "bellman ford",
            "kruskal", "prim", "floyd warshall", "topological sort",
            
            # Complexity terms
            "complexity", "time complexity", "space complexity", "big o", "big-o",
            "O(n)", "O(log n)", "O(n log n)", "O(n^2)", "O(2^n)", "O(1)", "O(n!)",
            
            # Common algorithms
            "binary search", "merge sort", "quick sort", "bubble sort", "insertion sort",
            "selection sort", "heap sort", "radix sort", "counting sort", "bucket sort",
            "kmp", "rabin karp", "z algorithm", "manacher", "union find", "disjoint set",
            
            # Problem types
            "shortest path", "minimum spanning tree", "mst", "cycle detection", 
            "strongly connected components", "scc", "bipartite", "maximum flow",
            "minimum cut", "longest common subsequence", "lcs", "edit distance",
            "knapsack", "traveling salesman", "tsp"
        ]
        
        # Extract keywords that appear in the query
        found_keywords = []
        query_lower = query.lower()
        
        for term in dsa_terms:
            if term.lower() in query_lower:
                found_keywords.append(term)
        
        return found_keywords
    
    def _generate_clarification_response(self, query: str) -> str:
        """
        Generate a clarification response when no context is found.
        
        Args:
            query: The user's query
            
        Returns:
            A clarification response
        """
        # Generate a custom response using the LLM
        clarification_prompt = f"""You are a helpful DSA (Data Structures and Algorithms) tutor.
        
        The user asked: "{query}"
        
        You don't have specific information about this exact query in your knowledge base.
        
        Generate a helpful response that:
        1. Acknowledges their question
        2. Suggests a few related DSA topics that might be relevant
        3. Asks them to clarify or rephrase their question with more specific DSA terminology
        
        Keep your response concise and focused on DSA topics.
        """
        
        try:
            # Use the LLM to generate a custom response
            response = self.llm_module.generate_custom_response(clarification_prompt)
            return response
        except Exception as e:
            # Fallback response if LLM fails
            return "I'm not sure I understand your question about data structures and algorithms. Could you please rephrase it or specify which DSA concept you're asking about? For example, are you interested in sorting algorithms, graph algorithms, tree structures, or something else?"
    
    def _handle_quiz_intent(self, query: str) -> Dict:
        """
        Handle a quiz intent.
        
        Args:
            query: The user's query
            
        Returns:
            Response data
        """
        # Extract topic and difficulty from query
        topic = self._extract_topic(query)
        difficulty = self._extract_difficulty(query)
        
        if not topic:
            # If no topic specified, suggest topics based on user's weak areas
            weak_topics = self.user_data_store.get_weak_topics(self.current_user_id)
            suggested_topics = self.quiz_manager.get_topic_suggestions(weak_topics)
            
            response = "I'd be happy to give you a quiz! What topic would you like to be quizzed on? Here are some suggestions:\n\n"
            response += "\n".join(f"- {topic}" for topic in suggested_topics)
            
            return {
                "response": response,
                "type": "quiz_suggestion"
            }
        
        # Generate quiz questions
        num_questions = 5
        self.current_quiz = self.quiz_manager.generate_quiz_session(topic, difficulty, num_questions)
        self.current_quiz_index = 0
        
        # Format the first question
        question_data = self.current_quiz[self.current_quiz_index]
        response = f"Let's start your quiz on {topic} ({difficulty} difficulty).\n\n"
        
        # Check if this is a problem-type question (has structured format)
        if isinstance(question_data['question'], dict) or "title:" in question_data['question'].lower():
            # This is a problem-type question, format it properly
            formatted_question = self._format_problem_question(question_data['question'])
            response += f"Question 1 of {len(self.current_quiz)}:\n{formatted_question}"
        else:
            # Regular question format
            response += f"Question 1 of {len(self.current_quiz)}:\n{question_data['question']}"
        
        return {
            "response": response,
            "type": "quiz_question",
            "topic": topic,
            "difficulty": difficulty,
            "question_index": self.current_quiz_index,
            "total_questions": len(self.current_quiz)
        }
    
    def _format_problem_question(self, question) -> str:
        """
        Format a problem question with all its components.
        
        Args:
            question: The question data (either a dict or a string)
            
        Returns:
            Formatted question string
        """
        # If question is already a string but contains title/difficulty markers
        if isinstance(question, str):
            # Try to parse the structured format from the string
            lines = question.split('\n')
            formatted_parts = []
            
            current_section = None
            section_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                lower_line = line.lower()
                
                # Check for section headers
                if lower_line.startswith("title:"):
                    if current_section and section_content:
                        formatted_parts.append("\n".join(section_content))
                        section_content = []
                    current_section = "title"
                    title_text = line.split(":", 1)[1].strip()
                    formatted_parts.append(f"# {title_text}")
                    
                elif lower_line.startswith("difficulty:"):
                    current_section = "difficulty"
                    difficulty_text = line.split(":", 1)[1].strip()
                    formatted_parts.append(f"Difficulty: {difficulty_text}")
                    
                elif lower_line.startswith("description:"):
                    current_section = "description"
                    section_content = []
                    # Add the description header
                    formatted_parts.append("## Description")
                    # If there's content on same line after "description:"
                    desc_text = line.split(":", 1)[1].strip()
                    if desc_text:
                        section_content.append(desc_text)
                        
                elif lower_line.startswith("input format:") or lower_line.startswith("input_format:"):
                    if current_section and section_content:
                        formatted_parts.append("\n".join(section_content))
                    current_section = "input_format"
                    section_content = []
                    formatted_parts.append("## Input Format")
                    input_text = line.split(":", 1)[1].strip()
                    if input_text:
                        section_content.append(input_text)
                        
                elif lower_line.startswith("output format:") or lower_line.startswith("output_format:"):
                    if current_section and section_content:
                        formatted_parts.append("\n".join(section_content))
                    current_section = "output_format"
                    section_content = []
                    formatted_parts.append("## Output Format")
                    output_text = line.split(":", 1)[1].strip()
                    if output_text:
                        section_content.append(output_text)
                        
                elif lower_line.startswith("constraints:"):
                    if current_section and section_content:
                        formatted_parts.append("\n".join(section_content))
                    current_section = "constraints"
                    section_content = []
                    formatted_parts.append("## Constraints")
                    
                elif lower_line.startswith("example:") or lower_line.startswith("examples:"):
                    if current_section and section_content:
                        formatted_parts.append("\n".join(section_content))
                    current_section = "example"
                    section_content = []
                    formatted_parts.append("## Example")
                    
                elif lower_line.startswith("input:") and current_section == "example":
                    input_text = line.split(":", 1)[1].strip()
                    section_content.append(f"Input: {input_text}")
                    
                elif lower_line.startswith("output:") and current_section == "example":
                    output_text = line.split(":", 1)[1].strip()
                    section_content.append(f"Output: {output_text}")
                    
                elif lower_line.startswith("explanation:") and current_section == "example":
                    explanation_text = line.split(":", 1)[1].strip()
                    section_content.append(f"Explanation: {explanation_text}")
                    
                else:
                    # Add to current section
                    if current_section == "constraints" and line.strip():
                        # Format constraint as bullet point
                        section_content.append(f"- {line}")
                    else:
                        section_content.append(line)
            
            # Add the last section if any
            if section_content:
                formatted_parts.append("\n".join(section_content))
                
            return "\n\n".join(formatted_parts)
            
        # If question is a dictionary with problem structure
        elif isinstance(question, dict):
            formatted_parts = []
            
            # Title
            if "title" in question:
                formatted_parts.append(f"# {question['title']}")
            
            # Difficulty
            if "difficulty" in question:
                formatted_parts.append(f"Difficulty: {question['difficulty']}")
            
            # Description
            if "description" in question:
                formatted_parts.append("## Description")
                formatted_parts.append(question["description"])
            
            # Input Format
            if "input_format" in question:
                formatted_parts.append("## Input Format")
                formatted_parts.append(question["input_format"])
            
            # Output Format
            if "output_format" in question:
                formatted_parts.append("## Output Format")
                formatted_parts.append(question["output_format"])
            
            # Constraints
            if "constraints" in question and question["constraints"]:
                formatted_parts.append("## Constraints")
                constraints = []
                for constraint in question["constraints"]:
                    constraints.append(f"- {constraint}")
                formatted_parts.append("\n".join(constraints))
            
            # Example
            if "example" in question:
                formatted_parts.append("## Example")
                example = question["example"]
                if isinstance(example, dict):
                    if "input" in example:
                        formatted_parts.append(f"Input: {example['input']}")
                    if "output" in example:
                        formatted_parts.append(f"Output: {example['output']}")
                    if "explanation" in example:
                        formatted_parts.append(f"Explanation: {example['explanation']}")
                else:
                    formatted_parts.append(str(example))
            
            return "\n\n".join(formatted_parts)
        
        # Fallback to original question if parsing fails
        return question
    
    def _handle_quiz_answer_intent(self, query: str) -> Dict:
        """
        Handle a quiz answer intent.
        
        Args:
            query: The user's answer
            
        Returns:
            Response data
        """
        if not self.current_quiz:
            return {
                "response": "I don't have an active quiz. Would you like to start one?",
                "type": "error"
            }
        
        # Get current question and answer
        question_data = self.current_quiz[self.current_quiz_index]
        user_answer = query.strip()
        
        # Evaluate the answer
        is_correct, explanation = self.quiz_manager.evaluate_answer(
            question_data['question'], 
            user_answer, 
            question_data['answer']
        )
        
        # Log the quiz attempt
        self.user_data_store.log_quiz_attempt(
            self.current_user_id,
            question_data['topic'],
            question_data['question'],
            user_answer,
            question_data['answer'],
            is_correct
        )
        
        # Prepare response
        if is_correct:
            response = f"Correct! {explanation}\n\n"
        else:
            response = f"Not quite. The correct answer is: {question_data['answer']}\n{explanation}\n\n"
        
        # Move to the next question or end the quiz
        self.current_quiz_index += 1
        
        if self.current_quiz_index < len(self.current_quiz):
            # Next question
            next_question = self.current_quiz[self.current_quiz_index]
            
            # Check if this is a problem-type question (has structured format)
            if isinstance(next_question['question'], dict) or "title:" in next_question['question'].lower():
                # This is a problem-type question, format it properly
                formatted_question = self._format_problem_question(next_question['question'])
                response += f"Question {self.current_quiz_index + 1} of {len(self.current_quiz)}:\n{formatted_question}"
            else:
                # Regular question format
                response += f"Question {self.current_quiz_index + 1} of {len(self.current_quiz)}:\n{next_question['question']}"
            
            return {
                "response": response,
                "type": "quiz_question",
                "is_correct": is_correct,
                "explanation": explanation,
                "question_index": self.current_quiz_index,
                "total_questions": len(self.current_quiz)
            }
        else:
            # End of quiz
            response += "That's the end of the quiz! Would you like to try another topic?"
            
            # Reset quiz state
            self.current_quiz = None
            self.current_quiz_index = 0
            
            return {
                "response": response,
                "type": "quiz_end",
                "is_correct": is_correct,
                "explanation": explanation
            }
    
    def _handle_code_execution_intent(self, query: str) -> Dict:
        """
        Handle a code execution intent.
        
        Args:
            query: The user's query containing code
            
        Returns:
            Response data
        """
        # Extract code and input
        code, input_data = self._extract_code_and_input(query)
        
        if not code:
            return {
                "response": "I couldn't find any C++ code to execute. Please provide code wrapped in ```cpp ... ``` blocks.",
                "type": "error"
            }
        
        # Execute the code
        start_time = time.time()
        result = self.cpp_executor.run_code(code, input_data)
        execution_time = time.time() - start_time
        
        # Log the code execution
        self.user_data_store.log_code_execution(
            self.current_user_id,
            code,
            input_data,
            result["execution_stdout"] if result["compile_success"] else result["compile_stderr"],
            result["success"],
            execution_time
        )
        
        # Prepare response
        if not result["compile_success"]:
            response = f"Compilation Error:\n```\n{result['compile_stderr']}\n```"
            return {
                "response": response,
                "type": "code_execution",
                "success": False,
                "execution_time": execution_time,
                "code": code
            }
        
        if not result["execution_success"]:
            response = f"Execution Error:\n```\n{result['execution_stderr']}\n```"
            return {
                "response": response,
                "type": "code_execution",
                "success": False,
                "execution_time": execution_time,
                "code": code
            }
        
        # Success
        response = f"Code executed successfully in {execution_time:.2f} seconds.\n"
        response += f"Output:\n```\n{result['execution_stdout']}\n```"
        
        return {
            "response": response,
            "type": "code_execution",
            "success": True,
            "execution_time": execution_time,
            "code": code,
            "output": result["execution_stdout"]
        }
    
    def _handle_code_explanation_intent(self, query: str) -> Dict:
        """
        Handle a code explanation intent.
        
        Args:
            query: The user's query
            
        Returns:
            Response data
        """
        # Extract code
        code, _ = self._extract_code_and_input(query)
        
        if not code:
            # Try to find code in the query
            code = query.replace("explain", "").replace("code", "").strip()
        
        if not code:
            return {
                "response": "I couldn't find any code to explain. Please provide the code you'd like me to explain.",
                "type": "error"
            }
        
        # Generate explanation
        explanation = self.llm_module.explain_code(code)
        
        return {
            "response": explanation,
            "type": "code_explanation",
            "code": code
        }
    
    def _extract_topic(self, query: str) -> str:
        """
        Extract topic from a query.
        
        Args:
            query: The user's query
            
        Returns:
            The extracted topic
        """
        # Look for "on X" or "about X" patterns
        on_pattern = r"(?:on|about)\s+(\w+(?:\s+\w+){0,2})"
        match = re.search(on_pattern, query, re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Remove quiz-related words
        cleaned_query = query.lower()
        for word in ["quiz", "test", "question", "give", "me", "a", "please", "would", "like"]:
            cleaned_query = cleaned_query.replace(word, "")
        
        # Return the remaining text as the topic
        return cleaned_query.strip()
    
    def _extract_difficulty(self, query: str) -> str:
        """
        Extract difficulty from a query.
        
        Args:
            query: The user's query
            
        Returns:
            The extracted difficulty (easy, medium, or hard)
        """
        if "easy" in query.lower():
            return "easy"
        elif "hard" in query.lower() or "difficult" in query.lower():
            return "hard"
        else:
            return "medium"  # Default
    
    def _extract_code_and_input(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Extract code and input from a query.
        
        Args:
            query: The user's query
            
        Returns:
            Tuple of (code, input_data)
        """
        # Extract code
        code_pattern = r"```(?:cpp|c\+\+)?(.*?)```"
        code_matches = re.findall(code_pattern, query, re.DOTALL)
        
        if not code_matches:
            return "", None
        
        code = code_matches[0].strip()
        
        # Extract input (if any)
        input_pattern = r"input:?\s*(.*?)(?:$|```)"
        input_match = re.search(input_pattern, query, re.IGNORECASE | re.DOTALL)
        
        input_data = input_match.group(1).strip() if input_match else None
        
        return code, input_data
    
    def get_user_progress(self) -> Dict:
        """
        Get the user's progress.
        
        Returns:
            Dictionary with user progress data
        """
        if not self.current_user_id:
            return {}
        
        # Get topic progress
        topic_progress = self.user_data_store.get_user_topic_progress(self.current_user_id)
        
        # Get quiz history
        quiz_history = self.user_data_store.get_user_quiz_history(self.current_user_id)
        
        # Calculate statistics
        total_questions = len(quiz_history)
        correct_answers = sum(1 for attempt in quiz_history if attempt["is_correct"])
        
        if total_questions > 0:
            accuracy = correct_answers / total_questions * 100
        else:
            accuracy = 0
        
        return {
            "topic_progress": topic_progress,
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "accuracy": accuracy
        }

# For testing purposes
if __name__ == "__main__":
    from src.llm_module import LLMModule
    from src.knowledge_indexer import KnowledgeIndexer
    from src.quiz_manager import QuizManager
    from src.user_data_store import UserDataStore
    from src.cpp_executor.executor import CppExecutor
    
    # This assumes you have already set up all the components
    model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"
    
    # Check if model exists, if not, print a message
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please download a GGUF model first.")
    else:
        llm_module = LLMModule(model_path=model_path)
        knowledge_indexer = KnowledgeIndexer(
            knowledge_base_dir="data/knowledge_base",
            db_directory="chroma_db"
        )
        quiz_manager = QuizManager(
            llm_module=llm_module,
            knowledge_indexer=knowledge_indexer,
            knowledge_base_dir="data/knowledge_base"
        )
        user_data_store = UserDataStore("user_data.db")
        cpp_executor = CppExecutor()
        
        chat_engine = ChatEngine(
            llm_module=llm_module,
            knowledge_indexer=knowledge_indexer,
            quiz_manager=quiz_manager,
            user_data_store=user_data_store,
            cpp_executor=cpp_executor
        )
        
        # Start a session
        chat_engine.start_session("test_user")
        
        # Test question
        response = chat_engine.process_query("What is merge sort?")
        print(f"Question: What is merge sort?")
        print(f"Response: {response['response']}")
        print()
        
        # Test quiz
        response = chat_engine.process_query("Give me a quiz on sorting algorithms")
        print(f"Query: Give me a quiz on sorting algorithms")
        print(f"Response: {response['response']}")
        print()
        
        # Test code execution
        code_query = """Run this code:
        ```cpp
        #include <iostream>
        
        int main() {
            std::cout << "Hello, World!" << std::endl;
            return 0;
        }
        ```
        """
        response = chat_engine.process_query(code_query)
        print(f"Query: {code_query}")
        print(f"Response: {response['response']}")
        print()
        
        # End session
        chat_engine.end_session() 