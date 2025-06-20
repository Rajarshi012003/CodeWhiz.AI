import os
import logging
from typing import List, Dict, Any, Optional, Union

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.llamacpp import LlamaCpp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMModule:
    """
    Handles loading and interacting with the local LLM.
    """
    
    def __init__(self, 
                 model_path: str,
                 n_ctx: int = 2048,
                 n_batch: int = 512,
                 temperature: float = 0.5,  # Reduced temperature for less randomness
                 verbose: bool = False):
        """
        Initialize the LLM module.
        
        Args:
            model_path: Path to the LLM model file
            n_ctx: Context window size
            n_batch: Batch size for inference
            temperature: Sampling temperature
            verbose: Whether to print verbose output
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.temperature = temperature
        self.verbose = verbose
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model
        logger.info(f"Loading LLM from {model_path}")
        self.llm = self._load_model()
        logger.info("LLM loaded successfully")
        
        # Define improved prompt templates
        self.qa_template = PromptTemplate.from_template(
            """You are a helpful, accurate, and detailed Data Structures and Algorithms tutor.
            
            Use ONLY the following context to answer the question. If the context doesn't contain enough information to answer the question completely, say "I don't have enough information to answer this question fully." and then provide what you can based on the context.
            
            DO NOT make up information or examples that aren't supported by the context.
            
            Context:
            {context}
            
            Question: {question}
            
            Provide a thorough answer with examples where appropriate. Include code snippets if they help explain the concept. Your answer should be at least 150 words when the context contains sufficient information.
            
            Answer:"""
        )
        
        self.quiz_template = PromptTemplate.from_template(
            """You are a Data Structures and Algorithms tutor creating a quiz.
            
            Create a detailed quiz question on the following topic. The question should test the user's understanding of key concepts, not just memorization.
            
            Topic: {topic}
            Difficulty: {difficulty}
            
            For multiple choice questions, include 4-5 options and clearly mark the correct answer.
            For coding questions, provide a clear problem statement with example inputs and outputs.
            For theoretical questions, be specific about what you're asking.
            
            Make sure your question is directly related to Data Structures and Algorithms, specifically focusing on the given topic.
            
            Quiz Question (include a clear answer section at the end):"""
        )
        
        self.code_explanation_template = PromptTemplate.from_template(
            """You are a Data Structures and Algorithms tutor.
            Explain the following C++ code step by step in detail. Include:
            
            1. The overall purpose of the code
            2. The algorithm or data structure being implemented
            3. A line-by-line or section-by-section explanation
            4. The time and space complexity analysis
            5. Any edge cases or limitations
            
            C++ Code:
            {code}
            
            Explanation:"""
        )
        
        # Set up output parser
        self.output_parser = StrOutputParser()
    
    def _load_model(self) -> LlamaCpp:
        """Load the LLM model."""
        try:
            # Configure the model with increased max_tokens
            model = LlamaCpp(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                temperature=self.temperature,
                verbose=self.verbose,
                f16_kv=True,  # Use half-precision for key/value cache
                streaming=False,
                max_tokens=1024  # Increased max tokens for longer responses
            )
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_answer(self, question: str, context: Union[str, List[str]]) -> str:
        """
        Generate an answer to a question using the provided context.
        
        Args:
            question: The user's question
            context: The context to use for answering the question
            
        Returns:
            The generated answer
        """
        # Combine context if it's a list
        if isinstance(context, list):
            context = "\n\n".join(context)
        
        # Prepare inputs
        inputs = {
            "question": question,
            "context": context
        }
        
        try:
            # Generate answer
            chain = self.qa_template | self.llm | self.output_parser
            answer = chain.invoke(inputs)
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I'm sorry, I encountered an error while generating an answer."
    
    def generate_quiz(self, topic: str, difficulty: str = "medium") -> str:
        """
        Generate a quiz question on a specific topic.
        
        Args:
            topic: The topic for the quiz
            difficulty: The difficulty level (easy, medium, hard)
            
        Returns:
            The generated quiz question
        """
        # Prepare inputs
        inputs = {
            "topic": topic,
            "difficulty": difficulty
        }
        
        try:
            # Generate quiz
            chain = self.quiz_template | self.llm | self.output_parser
            quiz = chain.invoke(inputs)
            return quiz
        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return "I'm sorry, I encountered an error while generating a quiz."
    
    def explain_code(self, code: str) -> str:
        """
        Generate an explanation for C++ code.
        
        Args:
            code: The C++ code to explain
            
        Returns:
            The explanation
        """
        # Prepare inputs
        inputs = {
            "code": code
        }
        
        try:
            # Generate explanation
            chain = self.code_explanation_template | self.llm | self.output_parser
            explanation = chain.invoke(inputs)
            return explanation
        except Exception as e:
            logger.error(f"Error explaining code: {str(e)}")
            return "I'm sorry, I encountered an error while explaining the code."
    
    def generate_custom_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using a custom prompt.
        
        Args:
            prompt: The prompt template
            **kwargs: The inputs for the prompt
            
        Returns:
            The generated response
        """
        try:
            # Create prompt template
            prompt_template = PromptTemplate.from_template(prompt)
            
            # Generate response
            chain = prompt_template | self.llm | self.output_parser
            response = chain.invoke(kwargs)
            return response
        except Exception as e:
            logger.error(f"Error generating custom response: {str(e)}")
            return "I'm sorry, I encountered an error while generating a response."

# For testing purposes
if __name__ == "__main__":
    # This assumes you have downloaded a GGUF format model
    # For example, a quantized Llama 2 model
    model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"
    
    # Check if model exists, if not, print a message
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please download a GGUF model first.")
    else:
        llm_module = LLMModule(model_path=model_path)
        
        # Test answer generation
        context = "Merge Sort is a divide-and-conquer algorithm that divides the input array into two halves, recursively sorts them, and then merges the sorted halves. It has a time complexity of O(n log n) in all cases."
        question = "What is the time complexity of Merge Sort?"
        
        print("Generating answer...")
        answer = llm_module.generate_answer(question, context)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        
        # Test quiz generation
        print("\nGenerating quiz...")
        quiz = llm_module.generate_quiz("binary search trees")
        print(f"Quiz: {quiz}") 