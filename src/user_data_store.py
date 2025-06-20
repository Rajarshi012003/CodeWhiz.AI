import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserDataStore:
    """
    Handles storing and retrieving user data, including progress and quiz results.
    """
    
    def __init__(self, db_path: str = "user_data.db"):
        """
        Initialize the user data store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        try:
            # Connect to database (will create if it doesn't exist)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            ''')
            
            # Create interactions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                interaction_type TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
            ''')
            
            # Create quiz_attempts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                question TEXT NOT NULL,
                user_answer TEXT,
                correct_answer TEXT NOT NULL,
                is_correct BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            ''')
            
            # Create topic_progress table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS topic_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                proficiency_level INTEGER DEFAULT 0,
                last_studied TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, topic)
            )
            ''')
            
            # Create code_executions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                code TEXT NOT NULL,
                input_data TEXT,
                output TEXT,
                success BOOLEAN,
                execution_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            ''')
            
            # Commit changes and close connection
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def get_or_create_user(self, username: str) -> int:
        """
        Get or create a user with the given username.
        
        Args:
            username: The username
            
        Returns:
            The user ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            
            if user:
                user_id = user[0]
            else:
                # Create new user
                cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
                user_id = cursor.lastrowid
                logger.info(f"Created new user: {username} (ID: {user_id})")
            
            conn.commit()
            conn.close()
            
            return user_id
        
        except Exception as e:
            logger.error(f"Error getting or creating user: {str(e)}")
            raise
    
    def start_session(self, user_id: int) -> int:
        """
        Start a new session for the user.
        
        Args:
            user_id: The user ID
            
        Returns:
            The session ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create new session
            cursor.execute("INSERT INTO sessions (user_id) VALUES (?)", (user_id,))
            session_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            logger.info(f"Started new session for user {user_id} (Session ID: {session_id})")
            return session_id
        
        except Exception as e:
            logger.error(f"Error starting session: {str(e)}")
            raise
    
    def end_session(self, session_id: int) -> None:
        """
        End a session.
        
        Args:
            session_id: The session ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update session end time
            cursor.execute(
                "UPDATE sessions SET end_time = CURRENT_TIMESTAMP WHERE id = ?", 
                (session_id,)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Ended session {session_id}")
        
        except Exception as e:
            logger.error(f"Error ending session: {str(e)}")
            raise
    
    def log_interaction(self, session_id: int, query: str, response: str, 
                       interaction_type: str) -> int:
        """
        Log a user interaction.
        
        Args:
            session_id: The session ID
            query: The user's query
            response: The system's response
            interaction_type: The type of interaction (e.g., 'question', 'quiz', 'code')
            
        Returns:
            The interaction ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert interaction
            cursor.execute(
                "INSERT INTO interactions (session_id, query, response, interaction_type) VALUES (?, ?, ?, ?)",
                (session_id, query, response, interaction_type)
            )
            interaction_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return interaction_id
        
        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")
            raise
    
    def log_quiz_attempt(self, user_id: int, topic: str, question: str, 
                        user_answer: str, correct_answer: str, is_correct: bool) -> int:
        """
        Log a quiz attempt.
        
        Args:
            user_id: The user ID
            topic: The quiz topic
            question: The quiz question
            user_answer: The user's answer
            correct_answer: The correct answer
            is_correct: Whether the user's answer is correct
            
        Returns:
            The quiz attempt ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert quiz attempt
            cursor.execute(
                """INSERT INTO quiz_attempts 
                (user_id, topic, question, user_answer, correct_answer, is_correct) 
                VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, topic, question, user_answer, correct_answer, is_correct)
            )
            attempt_id = cursor.lastrowid
            
            # Update topic progress
            self._update_topic_progress(conn, user_id, topic, is_correct)
            
            conn.commit()
            conn.close()
            
            return attempt_id
        
        except Exception as e:
            logger.error(f"Error logging quiz attempt: {str(e)}")
            raise
    
    def _update_topic_progress(self, conn: sqlite3.Connection, user_id: int, 
                              topic: str, is_correct: bool) -> None:
        """
        Update the user's progress on a topic.
        
        Args:
            conn: The database connection
            user_id: The user ID
            topic: The topic
            is_correct: Whether the user's answer was correct
        """
        cursor = conn.cursor()
        
        # Check if topic progress exists
        cursor.execute(
            "SELECT proficiency_level FROM topic_progress WHERE user_id = ? AND topic = ?",
            (user_id, topic)
        )
        result = cursor.fetchone()
        
        if result:
            # Update existing topic progress
            current_level = result[0]
            new_level = min(5, current_level + 1) if is_correct else max(0, current_level - 1)
            
            cursor.execute(
                """UPDATE topic_progress 
                SET proficiency_level = ?, last_studied = CURRENT_TIMESTAMP 
                WHERE user_id = ? AND topic = ?""",
                (new_level, user_id, topic)
            )
        else:
            # Create new topic progress
            initial_level = 1 if is_correct else 0
            cursor.execute(
                "INSERT INTO topic_progress (user_id, topic, proficiency_level) VALUES (?, ?, ?)",
                (user_id, topic, initial_level)
            )
    
    def log_code_execution(self, user_id: int, code: str, input_data: Optional[str],
                          output: str, success: bool, execution_time: float) -> int:
        """
        Log a code execution.
        
        Args:
            user_id: The user ID
            code: The code that was executed
            input_data: The input data for the code
            output: The output of the code execution
            success: Whether the execution was successful
            execution_time: The execution time in seconds
            
        Returns:
            The code execution ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert code execution
            cursor.execute(
                """INSERT INTO code_executions 
                (user_id, code, input_data, output, success, execution_time) 
                VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, code, input_data, output, success, execution_time)
            )
            execution_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return execution_id
        
        except Exception as e:
            logger.error(f"Error logging code execution: {str(e)}")
            raise
    
    def get_user_quiz_history(self, user_id: int, topic: Optional[str] = None) -> List[Dict]:
        """
        Get the user's quiz history.
        
        Args:
            user_id: The user ID
            topic: Optional topic filter
            
        Returns:
            List of quiz attempts
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            if topic:
                cursor.execute(
                    """SELECT * FROM quiz_attempts 
                    WHERE user_id = ? AND topic = ? 
                    ORDER BY timestamp DESC""",
                    (user_id, topic)
                )
            else:
                cursor.execute(
                    "SELECT * FROM quiz_attempts WHERE user_id = ? ORDER BY timestamp DESC",
                    (user_id,)
                )
            
            quiz_attempts = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            return quiz_attempts
        
        except Exception as e:
            logger.error(f"Error getting quiz history: {str(e)}")
            return []
    
    def get_user_topic_progress(self, user_id: int) -> List[Dict]:
        """
        Get the user's progress on all topics.
        
        Args:
            user_id: The user ID
            
        Returns:
            List of topic progress
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM topic_progress WHERE user_id = ? ORDER BY proficiency_level DESC",
                (user_id,)
            )
            
            topic_progress = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            return topic_progress
        
        except Exception as e:
            logger.error(f"Error getting topic progress: {str(e)}")
            return []
    
    def get_weak_topics(self, user_id: int, limit: int = 3) -> List[str]:
        """
        Get the user's weakest topics.
        
        Args:
            user_id: The user ID
            limit: Maximum number of topics to return
            
        Returns:
            List of topic names
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT topic FROM topic_progress 
                WHERE user_id = ? 
                ORDER BY proficiency_level ASC, last_studied ASC 
                LIMIT ?""",
                (user_id, limit)
            )
            
            weak_topics = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return weak_topics
        
        except Exception as e:
            logger.error(f"Error getting weak topics: {str(e)}")
            return []
    
    def get_recent_interactions(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        Get the user's most recent interactions.
        
        Args:
            user_id: The user ID
            limit: Maximum number of interactions to return
            
        Returns:
            List of interactions
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            cursor.execute(
                """SELECT i.* FROM interactions i
                JOIN sessions s ON i.session_id = s.id
                WHERE s.user_id = ?
                ORDER BY i.timestamp DESC
                LIMIT ?""",
                (user_id, limit)
            )
            
            interactions = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            return interactions
        
        except Exception as e:
            logger.error(f"Error getting recent interactions: {str(e)}")
            return []
    
    def export_user_data(self, user_id: int, output_file: str) -> bool:
        """
        Export all data for a user to a JSON file.
        
        Args:
            user_id: The user ID
            output_file: Path to the output file
            
        Returns:
            Success flag
        """
        try:
            # Collect all user data
            user_data = {
                "quiz_attempts": self.get_user_quiz_history(user_id),
                "topic_progress": self.get_user_topic_progress(user_id),
                "recent_interactions": self.get_recent_interactions(user_id, limit=100)
            }
            
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(user_data, f, indent=2, default=str)
            
            logger.info(f"Exported user data to {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting user data: {str(e)}")
            return False

# For testing purposes
if __name__ == "__main__":
    # Create a test instance
    data_store = UserDataStore("test_user_data.db")
    
    # Create a test user
    user_id = data_store.get_or_create_user("test_user")
    print(f"User ID: {user_id}")
    
    # Start a session
    session_id = data_store.start_session(user_id)
    print(f"Session ID: {session_id}")
    
    # Log some interactions
    data_store.log_interaction(
        session_id, 
        "What is merge sort?", 
        "Merge sort is a divide-and-conquer algorithm...", 
        "question"
    )
    
    # Log a quiz attempt
    data_store.log_quiz_attempt(
        user_id,
        "sorting_algorithms",
        "What is the time complexity of merge sort?",
        "O(n log n)",
        "O(n log n)",
        True
    )
    
    # End the session
    data_store.end_session(session_id)
    
    # Get quiz history
    quiz_history = data_store.get_user_quiz_history(user_id)
    print(f"Quiz History: {quiz_history}")
    
    # Get topic progress
    topic_progress = data_store.get_user_topic_progress(user_id)
    print(f"Topic Progress: {topic_progress}") 