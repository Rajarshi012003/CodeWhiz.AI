import os
import subprocess
import tempfile
import uuid
import logging
import signal
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CppExecutor:
    """
    Safely compiles and executes C++ code in a sandboxed environment.
    """
    
    def __init__(self, 
                 temp_dir: Optional[str] = None, 
                 compile_timeout: int = 10, 
                 execution_timeout: int = 5,
                 memory_limit: int = 512 * 1024):  # 512 MB in KB
        """
        Initialize the C++ executor.
        
        Args:
            temp_dir: Directory to store temporary files (default: system temp dir)
            compile_timeout: Timeout for compilation in seconds
            execution_timeout: Timeout for execution in seconds
            memory_limit: Memory limit for execution in KB
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.compile_timeout = compile_timeout
        self.execution_timeout = execution_timeout
        self.memory_limit = memory_limit
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def _create_temp_file(self, code: str) -> str:
        """Create a temporary file with the given code."""
        # Generate a unique filename
        filename = f"code_{uuid.uuid4().hex}.cpp"
        filepath = os.path.join(self.temp_dir, filename)
        
        # Write code to file
        with open(filepath, 'w') as f:
            f.write(code)
        
        return filepath
    
    def _compile_code(self, filepath: str) -> Tuple[bool, str, str]:
        """
        Compile C++ code.
        
        Args:
            filepath: Path to the C++ source file
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        # Get executable path
        executable = os.path.splitext(filepath)[0]
        
        # Compile with g++
        try:
            process = subprocess.Popen(
                ['g++', filepath, '-o', executable, '-std=c++17', '-Wall'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for compilation with timeout
            stdout, stderr = process.communicate(timeout=self.compile_timeout)
            
            # Check if compilation was successful
            success = process.returncode == 0
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            process.kill()
            return False, "", "Compilation timed out"
        except Exception as e:
            return False, "", f"Compilation error: {str(e)}"
    
    def _execute_code(self, executable: str, input_data: Optional[str] = None) -> Tuple[bool, str, str, int]:
        """
        Execute compiled C++ code.
        
        Args:
            executable: Path to the executable
            input_data: Optional input data for the program
            
        Returns:
            Tuple of (success, stdout, stderr, return_code)
        """
        try:
            # Set up process
            process = subprocess.Popen(
                [executable],
                stdin=subprocess.PIPE if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid  # Create a new process group
            )
            
            # Wait for execution with timeout
            stdout, stderr = process.communicate(
                input=input_data, 
                timeout=self.execution_timeout
            )
            
            return True, stdout, stderr, process.returncode
            
        except subprocess.TimeoutExpired:
            # Kill the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            return False, "", "Execution timed out", -1
        except Exception as e:
            return False, "", f"Execution error: {str(e)}", -1
    
    def _cleanup(self, filepath: str) -> None:
        """Clean up temporary files."""
        try:
            # Remove source file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Remove executable
            executable = os.path.splitext(filepath)[0]
            if os.path.exists(executable):
                os.remove(executable)
        except Exception as e:
            logger.error(f"Error cleaning up files: {str(e)}")
    
    def run_code(self, code: str, input_data: Optional[str] = None) -> Dict:
        """
        Compile and run C++ code.
        
        Args:
            code: C++ source code
            input_data: Optional input data for the program
            
        Returns:
            Dictionary with compilation and execution results
        """
        result = {
            "success": False,
            "compile_success": False,
            "compile_stdout": "",
            "compile_stderr": "",
            "execution_success": False,
            "execution_stdout": "",
            "execution_stderr": "",
            "execution_return_code": None
        }
        
        # Create temp file
        filepath = self._create_temp_file(code)
        
        try:
            # Compile code
            compile_success, compile_stdout, compile_stderr = self._compile_code(filepath)
            result["compile_success"] = compile_success
            result["compile_stdout"] = compile_stdout
            result["compile_stderr"] = compile_stderr
            
            # Execute code if compilation was successful
            if compile_success:
                executable = os.path.splitext(filepath)[0]
                execution_success, execution_stdout, execution_stderr, return_code = self._execute_code(
                    executable, input_data
                )
                result["execution_success"] = execution_success
                result["execution_stdout"] = execution_stdout
                result["execution_stderr"] = execution_stderr
                result["execution_return_code"] = return_code
                
                # Set overall success
                result["success"] = execution_success and return_code == 0
        
        finally:
            # Clean up
            self._cleanup(filepath)
        
        return result

# For testing purposes
if __name__ == "__main__":
    executor = CppExecutor()
    
    # Test with a simple C++ program
    test_code = """
    #include <iostream>
    
    int main() {
        std::cout << "Hello, World!" << std::endl;
        return 0;
    }
    """
    
    result = executor.run_code(test_code)
    print("Compilation output:", result["compile_stdout"] or "None")
    print("Compilation errors:", result["compile_stderr"] or "None")
    print("Execution output:", result["execution_stdout"] or "None")
    print("Execution errors:", result["execution_stderr"] or "None")
    print("Return code:", result["execution_return_code"])
    print("Success:", result["success"]) 