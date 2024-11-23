# Project Overview

This project aims to build a chatbot that responds to user queries using the Retrieval-Augmented Generation (RAG) framework, delivering accurate and contextually relevant answers.

---

## Environment Setup

1. **Python Version**: 3.8.8  
2. **SQLite Version**: > 3.35 (required for ChromaDB, the vector database)  
3. **Dependencies**: Install all required packages using:
    ```bash
    pip install -r requirements.txt
    ```
4. **API Key Configuration**:  
   Update the `config.json` file located in the `code` directory. Add your OpenAI API key to the file before running the code.

---

## Program Description

### 1. **Query Module**
- **File**: `demo_Query.py`  
- **Description**:  
  This script performs simple Q&A operations. All functions and classes used in this script are defined in `QueryAndEvaluation.py`.

### 2. **Query and Evaluation Module**
- **File**: `demo_QueryAndEvaluation.py`  
- **Description**:  
  This script evaluates responses under different parameter settings.  
  - The results are saved as both `.csv` and `.xlsx` files for comparison and analysis.  
  - The goal is to determine the optimal parameter configuration.  
  - All functions and classes used are located in `QueryAndEvaluation.py`.

### 3. **Data Processing Module**
- **File**: `Data_Processing.py`  
- **Description**:  
  This script handles:
  - Storing text chunks in the vector database.
  - Saving data as `.json` files.
  - Creating related lexical index files.
  - By default, it creates a vector database with text chunks of length 500.  
    To test other chunk sizes, use this script to create the corresponding vector database and related files.

---

## Notes

- Ensure your environment meets the requirements before running any scripts.
- Update the `config.json` file with valid API credentials for seamless functionality.
- Experiment with different parameter settings using the `Query and Evaluation` module to refine performance.

---





如何使用此程式？

- 環境設置：
    - Python版本：3.8.8
    - SQLite > 3.35 （這是向量資料庫chromadb的要求）
    - 其他的套件，直接 pip install -r requirements.txt 即可
    - 請至code底下的config.json檔，將您的openai key填進去


- 程式說明：
    - demo_Query.py：單純問答，所有使用到的function/class，皆放在QueryAndEvaluation.py
    - demo_QueryAndEvaluation.py : 對不同參數設定下的回答進行評估（evaluation），並將結果儲存成csv和excel檔，以便進行比較和評估，進而選擇最佳的參數。所有使用到的function/class，皆放在QueryAndEvaluation.py
    - Data_Processing.py : 將文字組塊儲存在向量資料庫、儲存成json檔、儲存成lexical _index的相關檔案。目前只有建立文字組塊長度為500的向量資料庫以及相關檔案，若您要測試其他長度，請使用此程式建立相對應的向量資料庫以及檔案




