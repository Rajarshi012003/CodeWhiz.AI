import json
import os
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

from src.llm_module import LLMModule
from src.knowledge_indexer import KnowledgeIndexer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuizManager:
    """
    Handles quiz generation, evaluation, and management.
    """
    
    def __init__(self, 
                 llm_module: LLMModule,
                 knowledge_indexer: KnowledgeIndexer,
                 knowledge_base_dir: str):
        """
        Initialize the quiz manager.
        
        Args:
            llm_module: The LLM module for generating quiz questions
            knowledge_indexer: The knowledge indexer for retrieving relevant content
            knowledge_base_dir: Directory containing knowledge base JSON files
        """
        self.llm_module = llm_module
        self.knowledge_indexer = knowledge_indexer
        self.knowledge_base_dir = knowledge_base_dir
        
        # Load knowledge base files
        self.algorithms_data = self._load_json_file(os.path.join(knowledge_base_dir, "algorithms.json"))
        self.concepts_data = self._load_json_file(os.path.join(knowledge_base_dir, "concepts.json"))
        self.problems_data = self._load_json_file(os.path.join(knowledge_base_dir, "problems.json"))
        
        # Cache for quiz questions
        self.quiz_cache = {}
    
    def _load_json_file(self, file_path: str) -> Dict:
        """Load a JSON file and return its contents."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return {}
    
    def generate_quiz_question(self, topic: str, difficulty: str = "medium") -> Dict:
        """
        Generate a quiz question on a specific topic.
        
        Args:
            topic: The topic for the quiz
            difficulty: The difficulty level (easy, medium, hard)
            
        Returns:
            Dictionary containing the quiz question and answer
        """
        # Check if we have a cached question for this topic and difficulty
        cache_key = f"{topic}_{difficulty}"
        if cache_key in self.quiz_cache:
            # Use a cached question with 50% probability, otherwise generate a new one
            if random.random() < 0.5:
                return self.quiz_cache[cache_key]
        
        # First, try to find relevant content in the knowledge base
        results = self.knowledge_indexer.query_knowledge_base(
            query=f"{topic} {difficulty} concept algorithm problem",
            n_results=5  # Increased from 3 to 5 for more context
        )
        
        # Extract context from results
        context = "\n\n".join(results['documents'][0]) if results['documents'] else ""
        
        # If no context found, try to find direct matches in the JSON files
        if not context:
            context = self._find_direct_matches(topic)
        
        # Generate quiz question using LLM with the context
        if context:
            # Use a custom prompt that includes the context
            quiz_text = self.llm_module.generate_custom_response(
                """You are a Data Structures and Algorithms tutor creating a quiz.
                
                Use the following context information to create a detailed quiz question on "{topic}" with {difficulty} difficulty.
                
                Context:
                {context}
                
                For multiple choice questions, include 4-5 options and clearly mark the correct answer.
                For coding questions, provide a clear problem statement with example inputs and outputs.
                For theoretical questions, be specific about what you're asking.
                
                Make sure your question is directly related to the context provided.
                Include a clear ANSWER section at the end that provides the correct answer.
                
                Quiz Question:""",
                topic=topic,
                difficulty=difficulty,
                context=context
            )
        else:
            # Fallback to standard quiz generation
            quiz_text = self.llm_module.generate_quiz(topic, difficulty)
        
        # Parse the quiz text to extract question and answer
        question, answer = self._parse_quiz_text(quiz_text)
        
        # Create quiz data
        quiz_data = {
            "topic": topic,
            "difficulty": difficulty,
            "question": question,
            "answer": answer,
            "context": context
        }
        
        # Cache the quiz question
        self.quiz_cache[cache_key] = quiz_data
        
        return quiz_data
    
    def _find_direct_matches(self, topic: str) -> str:
        """
        Find direct matches for a topic in the knowledge base files.
        
        Args:
            topic: The topic to search for
            
        Returns:
            String containing relevant context
        """
        topic_lower = topic.lower()
        contexts = []
        
        # Search in algorithms
        for algo_id, algo_data in self.algorithms_data.items():
            if (topic_lower in algo_id.lower() or 
                topic_lower in algo_data.get("name", "").lower() or 
                topic_lower in algo_data.get("category", "").lower()):
                
                contexts.append(f"Algorithm: {algo_data.get('name', algo_id)}")
                contexts.append(f"Description: {algo_data.get('description', '')}")
                contexts.append(f"Time Complexity: {algo_data.get('time_complexity', '')}")
                contexts.append(f"Space Complexity: {algo_data.get('space_complexity', '')}")
                if "pseudocode" in algo_data:
                    contexts.append(f"Pseudocode: {algo_data['pseudocode']}")
                if "cpp_implementation" in algo_data:
                    contexts.append(f"C++ Implementation: {algo_data['cpp_implementation']}")
                contexts.append("")  # Add a blank line
        
        # Search in concepts
        for concept_id, concept_data in self.concepts_data.items():
            if (topic_lower in concept_id.lower() or 
                topic_lower in concept_data.get("name", "").lower()):
                
                contexts.append(f"Concept: {concept_data.get('name', concept_id)}")
                contexts.append(f"Description: {concept_data.get('description', '')}")
                contexts.append(f"Key Points: {', '.join(concept_data.get('key_points', []))}")
                if "examples" in concept_data:
                    contexts.append(f"Examples: {concept_data['examples']}")
                contexts.append("")  # Add a blank line
        
        # Search in problems
        for problem_id, problem_data in self.problems_data.items():
            if (topic_lower in problem_id.lower() or
                topic_lower in problem_data.get("name", "").lower() or
                any(topic_lower in concept.lower() for concept in problem_data.get("related_concepts", []))):
                
                contexts.append(f"Problem: {problem_data.get('name', problem_id)}")
                contexts.append(f"Description: {problem_data.get('description', '')}")
                contexts.append(f"Difficulty: {problem_data.get('difficulty', '')}")
                if "solution_approach" in problem_data:
                    contexts.append(f"Solution Approach: {problem_data['solution_approach']}")
                if "cpp_solution" in problem_data:
                    contexts.append(f"C++ Solution: {problem_data['cpp_solution']}")
                contexts.append("")  # Add a blank line
        
        return "\n".join(contexts)
    
    def _parse_quiz_text(self, quiz_text: str) -> Tuple[str, str]:
        """
        Parse quiz text to extract question and answer.
        
        Args:
            quiz_text: The generated quiz text
            
        Returns:
            Tuple of (question, answer)
        """
        # Look for explicit answer section
        if "ANSWER:" in quiz_text:
            parts = quiz_text.split("ANSWER:", 1)
            return parts[0].strip(), parts[1].strip()
        
        if "Answer:" in quiz_text:
            parts = quiz_text.split("Answer:", 1)
            return parts[0].strip(), parts[1].strip()
        
        # Look for correct answer in multiple choice
        lines = quiz_text.strip().split('\n')
        
        question_lines = []
        answer_lines = []
        in_answer = False
        
        for line in lines:
            if in_answer:
                answer_lines.append(line)
            elif line.lower().startswith(("answer:", "correct answer:")):
                in_answer = True
                answer_lines.append(line)
            else:
                question_lines.append(line)
        
        # If no explicit answer section was found, assume it's a multiple choice
        # and the answer is embedded in the question
        if not answer_lines:
            # Try to extract from multiple choice format
            return self._extract_multiple_choice(quiz_text)
        
        question = '\n'.join(question_lines).strip()
        answer = '\n'.join(answer_lines).strip()
        
        # Remove "Answer:" prefix if present
        answer = answer.replace("Answer:", "").replace("Correct Answer:", "").strip()
        
        return question, answer
    
    def _extract_multiple_choice(self, quiz_text: str) -> Tuple[str, str]:
        """
        Extract question and answer from multiple choice format.
        
        Args:
            quiz_text: The quiz text
            
        Returns:
            Tuple of (question, answer)
        """
        # Check for marked correct answer in the format "(correct)" or "[correct]" or "✓"
        for marker in ["(correct)", "[correct]", "✓", "correct"]:
            if marker.lower() in quiz_text.lower():
                lines = quiz_text.strip().split('\n')
                for i, line in enumerate(lines):
                    if marker.lower() in line.lower():
                        # Found the correct answer line
                        answer = line.strip()
                        # Remove the marker from the answer in the question
                        lines[i] = line.replace(marker, "").strip()
                        question = '\n'.join(lines).strip()
                        return question, answer
        
        # For now, just return the whole text as question and ask LLM for answer
        answer = self.llm_module.generate_custom_response(
            """You are given a multiple choice quiz question. Please provide ONLY the correct answer letter (A, B, C, D, etc.)
            and the corresponding text. Format your answer as: "X. Answer text"
            
            Quiz Question:
            {question}
            
            Correct Answer:""",
            question=quiz_text
        )
        
        return quiz_text, answer.strip()
    
    def get_problem_by_topic(self, topic: str, difficulty: Optional[str] = None) -> Optional[Dict]:
        """
        Get a problem from the knowledge base by topic.
        
        Args:
            topic: The topic
            difficulty: Optional difficulty filter
            
        Returns:
            Problem data or None if not found
        """
        # Filter problems by topic
        matching_problems = []
        
        for problem_id, problem_data in self.problems_data.items():
            # Check if the topic is in related concepts
            related_concepts = problem_data.get("related_concepts", [])
            
            if (topic.lower() in [concept.lower() for concept in related_concepts] or
                topic.lower() in problem_id.lower() or
                topic.lower() in problem_data.get("title", "").lower()):
                
                # Apply difficulty filter if provided
                if difficulty and problem_data.get("difficulty", "").lower() != difficulty.lower():
                    continue
                
                # Create a complete problem structure
                complete_problem = {
                    "id": problem_id,
                    "title": problem_data.get("title", problem_id),
                    "difficulty": problem_data.get("difficulty", "medium"),
                    "description": problem_data.get("description", ""),
                    "input_format": problem_data.get("input_format", ""),
                    "output_format": problem_data.get("output_format", ""),
                    "constraints": problem_data.get("constraints", []),
                    "example": problem_data.get("example", {})
                }
                
                matching_problems.append(complete_problem)
        
        # Return a random matching problem
        if matching_problems:
            return random.choice(matching_problems)
        
        return None
    
    def evaluate_answer(self, question: str, user_answer: str, correct_answer: str) -> Tuple[bool, str]:
        """
        Evaluate a user's answer to a quiz question.
        
        Args:
            question: The quiz question
            user_answer: The user's answer
            correct_answer: The correct answer
            
        Returns:
            Tuple of (is_correct, explanation)
        """
        # Use LLM to evaluate the answer with improved prompt
        evaluation_prompt = f"""You are evaluating a student's answer to a Data Structures and Algorithms quiz question.
        
        Question: {question}
        
        Correct Answer: {correct_answer}
        
        Student Answer: {user_answer}
        
        First, determine if the student's answer is correct or incorrect. The answer should be considered correct if it 
        captures the main concepts correctly, even if the wording is different. For multiple choice questions, the answer 
        must match exactly.
        
        Then, provide a detailed explanation (at least 100 words) of why the answer is correct or incorrect. Include:
        1. The key concepts involved
        2. Why the correct answer is right
        3. If incorrect, what misconceptions the student might have
        4. Additional context that helps reinforce the learning
        
        Format your response as:
        CORRECT: True/False
        EXPLANATION: Your explanation here
        """
        
        evaluation = self.llm_module.generate_custom_response(evaluation_prompt)
        
        # Parse the evaluation
        is_correct = False
        explanation = ""
        
        for line in evaluation.split('\n'):
            if line.startswith("CORRECT:"):
                is_correct_text = line.replace("CORRECT:", "").strip().lower()
                is_correct = is_correct_text in ["true", "yes", "1"]
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()
            elif explanation:  # Append to explanation if we've already started it
                explanation += " " + line.strip()
        
        return is_correct, explanation
    
    def generate_quiz_session(self, topic: str, difficulty: str = "medium", num_questions: int = 5) -> List[Dict]:
        """
        Generate a quiz session with multiple questions.
        
        Args:
            topic: The topic for the quiz
            difficulty: The difficulty level
            num_questions: Number of questions to generate
            
        Returns:
            List of quiz questions
        """
        quiz_questions = []
        
        # Try to include at least one problem-based question if available
        problem = self.get_problem_by_topic(topic, difficulty)
        if problem:
            # Create a quiz question from the problem
            quiz_data = {
                "topic": topic,
                "difficulty": difficulty,
                "question": problem,  # Use the entire problem structure
                "answer": "This is a conceptual problem. Provide your solution approach.",
                "context": f"Problem: {problem['title']}"
            }
            quiz_questions.append(quiz_data)
            
            # Reduce the number of remaining questions
            num_questions -= 1
        
        # Generate the remaining questions
        for _ in range(num_questions):
            quiz_data = self.generate_quiz_question(topic, difficulty)
            quiz_questions.append(quiz_data)
            
            # Add a small delay to ensure variety in questions
            # This helps prevent the LLM from generating very similar questions
            import time
            time.sleep(0.5)
        
        # Shuffle the questions to mix problem-based and regular questions
        random.shuffle(quiz_questions)
        
        return quiz_questions
    
    def get_topic_suggestions(self, user_weak_topics: List[str] = None) -> List[str]:
        """
        Get topic suggestions for quizzes.
        
        Args:
            user_weak_topics: Optional list of user's weak topics
            
        Returns:
            List of suggested topics
        """
        # Start with user's weak topics if provided
        suggested_topics = list(user_weak_topics) if user_weak_topics else []
        
        # Add some random topics from the knowledge base
        all_topics = set()
        
        # Extract topics from algorithms
        for algo_id, algo_data in self.algorithms_data.items():
            category = algo_data.get("category", "")
            if category:
                all_topics.add(category)
            name = algo_data.get("name", "")
            if name:
                all_topics.add(name)
        
        # Extract topics from concepts
        for concept_id, concept_data in self.concepts_data.items():
            name = concept_data.get("name", concept_id)
            if name:
                all_topics.add(name)
        
        # Add random topics to suggestions
        remaining_topics = list(all_topics - set(suggested_topics))
        random.shuffle(remaining_topics)
        suggested_topics.extend(remaining_topics[:5])
        
        return suggested_topics[:8]  # Return at most 8 suggestions

# For testing purposes
if __name__ == "__main__":
    from src.llm_module import LLMModule
    from src.knowledge_indexer import KnowledgeIndexer
    
    # This assumes you have already set up the LLM and knowledge indexer
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
        
        # Generate a quiz question
        print("Generating quiz question on sorting algorithms...")
        quiz_data = quiz_manager.generate_quiz_question("sorting algorithms", "medium")
        print(f"Question: {quiz_data['question']}")
        print(f"Answer: {quiz_data['answer']}")
        
        # Evaluate an answer
        user_answer = "O(n log n)"
        is_correct, explanation = quiz_manager.evaluate_answer(
            quiz_data['question'], user_answer, quiz_data['answer']
        )
        print(f"User answer: {user_answer}")
        print(f"Correct: {is_correct}")
        print(f"Explanation: {explanation}") 