# CodeWhiz.AI 🧙‍♀️💬
“Talk to a true algorithm whiz!!”

# 👩🏻‍🏫📚🖳 **DSA Tutor Chatbot**

> *A local-first Data Structures and Algorithms (DSA) tutor chatbot that runs entirely on your machine — no cloud, just power! ⚡*

---

## 📚 Overview

The **DSA Tutor Chatbot** is designed to help users learn **Data Structures and Algorithms** through **interactive conversations**, **quizzes**, and **live C++ code execution**.
It leverages a **Retrieval-Augmented Generation (RAG)** architecture powered by TinyLlama 🦙 to provide accurate and contextually relevant information on DSA concepts.

---

## 🚀 Features

🔹 **Interactive Learning** – Ask questions about any DSA topic and get detailed explanations
🧠 **Quiz System** – Test your knowledge with quizzes on various DSA topics
💻 **Code Execution** – Run and test C++ code snippets directly in the chat
📜 **Code Explanation** – Get clear explanations of C++ code to understand algorithms better
📊 **Progress Tracking** – Monitor your learning journey across different DSA topics

---

## 🧩 Architecture

The **DSA Tutor Chatbot** is built with a **modular architecture**, including:

1. 🗂️ **Knowledge Base Indexer** – Converts JSON files (`algorithms.json`, `concepts.json`, `problems.json`) into a **vector database** for semantic retrieval
2. 🧠 **LLM Module** – Uses open-source models (like **TinyLlama**) for generating responses
3. 📝 **Quiz Manager** – Handles quiz creation and scoring
4. ⚙️ **C++ Executor** – Compiles and runs C++ code in a **sandboxed environment**
5. 🧾 **User Data Store** – Tracks user progress using **SQLite**
6. 💬 **Chat Engine** – Connects everything and responds to user input

---

## 🛠️ Installation

### 📌 Prerequisites

* 🐍 Python **3.8+**
* 🛠️ C++ compiler (**g++** or **clang**)
* ⚡ CUDA-compatible GPU *(optional, for faster LLM inference)*

---

### 📥 Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/DSA_BOT.git
   cd DSA_BOT
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the LLM model**

   ```bash
   mkdir -p models
   # Download TinyLlama model
   wget -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
   ```

4. **Initialize the knowledge base**

   ```bash
   python -c "from src.knowledge_indexer import KnowledgeIndexer; KnowledgeIndexer(knowledge_base_dir='data/knowledge_base', db_directory='chroma_db').index_knowledge_base()"
   ```

---

## 💬 Usage

### ▶️ Running the Chatbot

```bash
python main.py
```

Customize it with your own model or knowledge base:

```bash
python main.py --model models/your-model-name.gguf --knowledge-base path/to/knowledge_base
```

---

### 🧑‍🏫 Interacting with the Chatbot

1. **Ask questions** 🧠

   ```text
   What is the time complexity of quicksort?
   ```

2. **Take quizzes** 📝

   ```text
   quiz sorting algorithms
   ```

3. **Run code** 💻

   ````text
   run
   ```cpp
   #include <iostream>

   int main() {
       std::cout << "Hello, World!" << std::endl;
       return 0;
   }
   ````

4. **Get explanations** 🧾

   ```text
   explain binary search
   ```

5. **Check progress** 📊

   ```text
   progress
   ```

6. **Get help** ❓

   ```text
   help
   ```

7. **Exit chatbot** 🛑

   ```text
   exit
   ```

---

## 🗂️ Project Structure

```
DSA_BOT/
├── data/
│   └── knowledge_base/
│       ├── algorithms.json
│       ├── concepts.json
│       └── problems.json
├── models/
│   └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
├── src/
│   ├── chat_engine.py
│   ├── cli.py
│   ├── cpp_executor/
│   │   └── executor.py
│   ├── knowledge_indexer.py
│   ├── llm_module.py
│   ├── quiz_manager.py
│   └── user_data_store.py
├── chroma_db/
├── main.py
├── requirements.txt
└── user_data.db
```

---

## 🎨 Customization

### ➕ Adding New Knowledge

Expand the knowledge base by editing or adding new JSON files inside the `data/knowledge_base` directory:

* `algorithms.json` – Info about various algorithms
* `concepts.json` – Explanations of DSA topics
* `problems.json` – Practice problems and solutions

Then reindex the knowledge base:

```bash
python -c "from src.knowledge_indexer import KnowledgeIndexer; KnowledgeIndexer(knowledge_base_dir='data/knowledge_base', db_directory='chroma_db').index_knowledge_base()"
```

---

### 🔄 Using Different LLM Models

You can use **any GGUF-compatible model**!
Just download the model to the `models/` directory and start the chatbot like this:

```bash
python main.py --model models/your-model-name.gguf
```

---

## ⚠️ Limitations

* 🧠 Response quality depends on the selected LLM
* 🔒 Code execution is sandboxed and limited to standard libraries
* 🧮 Requires sufficient system memory (minimum **4GB RAM** recommended)

---

## 📄 License

This project is licensed under the **MIT License** – see the `LICENSE` file for full details.

---

## 🙏 Acknowledgments

* 🤖 Uses the **TinyLlama** model for intelligent chat responses
* 💾 Vector database powered by **ChromaDB**
* 🔍 **Sentence Transformers** used for semantic search

---

Let the learning begin! 🌟👨‍💻👩‍💻
