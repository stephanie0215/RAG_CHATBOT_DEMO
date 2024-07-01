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




