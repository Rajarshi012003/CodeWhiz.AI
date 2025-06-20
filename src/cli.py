#!/usr/bin/env python3

import os
import sys
import time
import argparse
import logging
from typing import Optional
import getpass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print

from src.llm_module import LLMModule
from src.knowledge_indexer import KnowledgeIndexer
from src.quiz_manager import QuizManager
from src.user_data_store import UserDataStore
from src.cpp_executor.executor import CppExecutor
from src.chat_engine import ChatEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()

def setup_components(model_path: str, knowledge_base_dir: str) -> ChatEngine:
    """
    Set up all components and return a chat engine.
    
    Args:
        model_path: Path to the LLM model file
        knowledge_base_dir: Path to the knowledge base directory
        
    Returns:
        A configured ChatEngine instance
    """
    console.print("[bold blue]Setting up components...[/bold blue]")
    
    # Check if model exists
    if not os.path.exists(model_path):
        console.print(f"[bold red]Error:[/bold red] Model not found at {model_path}")
        console.print("Please download a GGUF model first.")
        sys.exit(1)
    
    # Check if knowledge base exists
    if not os.path.exists(knowledge_base_dir):
        console.print(f"[bold red]Error:[/bold red] Knowledge base not found at {knowledge_base_dir}")
        sys.exit(1)
    
    # Set up components
    try:
        console.print("Loading LLM module...")
        llm_module = LLMModule(model_path=model_path)
        
        console.print("Setting up knowledge indexer...")
        db_directory = "chroma_db"
        knowledge_indexer = KnowledgeIndexer(
            knowledge_base_dir=knowledge_base_dir,
            db_directory=db_directory
        )
        
        # Check if the collection exists, if not, index the knowledge base
        if not os.path.exists(db_directory) or not os.listdir(db_directory):
            console.print("Indexing knowledge base (this may take a while)...")
            knowledge_indexer.index_knowledge_base()
        
        console.print("Setting up quiz manager...")
        quiz_manager = QuizManager(
            llm_module=llm_module,
            knowledge_indexer=knowledge_indexer,
            knowledge_base_dir=knowledge_base_dir
        )
        
        console.print("Setting up user data store...")
        user_data_store = UserDataStore("user_data.db")
        
        console.print("Setting up C++ executor...")
        cpp_executor = CppExecutor()
        
        console.print("Setting up chat engine...")
        chat_engine = ChatEngine(
            llm_module=llm_module,
            knowledge_indexer=knowledge_indexer,
            quiz_manager=quiz_manager,
            user_data_store=user_data_store,
            cpp_executor=cpp_executor
        )
        
        console.print("[bold green]All components set up successfully![/bold green]")
        return chat_engine
        
    except Exception as e:
        console.print(f"[bold red]Error setting up components:[/bold red] {str(e)}")
        sys.exit(1)

def print_welcome_message() -> None:
    """Print a welcome message."""
    console.print(Panel.fit(
        "[bold cyan]Welcome to the DSA Tutor Chatbot![/bold cyan]\n\n"
        "I can help you learn about data structures and algorithms. You can:\n"
        "- Ask me questions about DSA concepts\n"
        "- Take quizzes on various topics\n"
        "- Execute and explain C++ code\n"
        "- Track your learning progress\n\n"
        "Type 'help' for more information or 'exit' to quit.",
        title="DSA Tutor",
        border_style="cyan"
    ))

def print_help() -> None:
    """Print help information."""
    console.print(Panel.fit(
        "[bold]Available Commands:[/bold]\n\n"
        "- [cyan]help[/cyan]: Show this help message\n"
        "- [cyan]exit[/cyan]: Exit the chatbot\n"
        "- [cyan]quiz <topic> [difficulty][/cyan]: Take a quiz on a topic (e.g., 'quiz sorting algorithms')\n"
        "- [cyan]progress[/cyan]: Show your learning progress\n"
        "- [cyan]run[/cyan]: Run C++ code (paste code between ```cpp and ```)\n"
        "- [cyan]explain[/cyan]: Explain C++ code (paste code between ```cpp and ```)\n\n"
        "For anything else, just ask a question about data structures and algorithms!",
        title="Help",
        border_style="green"
    ))

def run_cli(chat_engine: ChatEngine) -> None:
    """
    Run the CLI interface.
    
    Args:
        chat_engine: The chat engine to use
    """
    print_welcome_message()
    
    # Get username
    username = Prompt.ask("[bold cyan]Please enter your username[/bold cyan]")
    
    # Start session
    session_id = chat_engine.start_session(username)
    console.print(f"[green]Session started for {username} (ID: {session_id})[/green]")
    
    # Main loop
    try:
        while True:
            # Get user input
            query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            # Check for exit command
            if query.lower() in ["exit", "quit", "bye"]:
                break
            
            # Check for help command
            if query.lower() == "help":
                print_help()
                continue
            
            # Check for progress command
            if query.lower() == "progress":
                progress = chat_engine.get_user_progress()
                if not progress:
                    console.print("[yellow]No progress data available yet.[/yellow]")
                else:
                    console.print(Panel.fit(
                        f"[bold]Quiz Statistics:[/bold]\n"
                        f"Total questions: {progress['total_questions']}\n"
                        f"Correct answers: {progress['correct_answers']}\n"
                        f"Accuracy: {progress['accuracy']:.1f}%\n\n"
                        f"[bold]Topic Progress:[/bold]",
                        title="Learning Progress",
                        border_style="blue"
                    ))
                    
                    for topic in progress['topic_progress']:
                        level = topic['proficiency_level']
                        stars = "★" * level + "☆" * (5 - level)
                        console.print(f"{topic['topic']}: {stars}")
                continue
            
            # Process the query
            console.print("[cyan]Processing...[/cyan]")
            start_time = time.time()
            
            try:
                response_data = chat_engine.process_query(query)
                
                # Print response based on type
                if response_data["type"] == "error":
                    console.print(f"[bold red]Error:[/bold red] {response_data['response']}")
                elif response_data["type"] == "code_execution":
                    if response_data["success"]:
                        console.print(f"[bold green]Success:[/bold green] Code executed in {response_data['processing_time']:.2f} seconds")
                        console.print(Markdown(response_data["response"]))
                    else:
                        console.print(f"[bold red]Error:[/bold red] Code execution failed")
                        console.print(Markdown(response_data["response"]))
                else:
                    console.print("\n[bold green]DSA Tutor[/bold green]:")
                    console.print(Markdown(response_data["response"]))
                    
                    # Print processing time
                    console.print(f"[dim](Processed in {response_data['processing_time']:.2f} seconds)[/dim]")
            
            except Exception as e:
                console.print(f"[bold red]Error processing query:[/bold red] {str(e)}")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    
    finally:
        # End session
        chat_engine.end_session()
        console.print("[green]Session ended. Goodbye![/green]")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="DSA Tutor Chatbot CLI")
    parser.add_argument("--model", type=str, default="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                        help="Path to the LLM model file")
    parser.add_argument("--knowledge-base", type=str, default="data/knowledge_base",
                        help="Path to the knowledge base directory")
    
    args = parser.parse_args()
    
    # Set up components
    chat_engine = setup_components(args.model, args.knowledge_base)
    
    # Run CLI
    run_cli(chat_engine)

if __name__ == "__main__":
    main() 