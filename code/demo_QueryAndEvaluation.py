import json
import os
import chromadb
import time
import pandas as pd
from QueryAndEvaluation import evaluate_RetrievalScore_AnswerQuality
import pickle

with open('code/config.json', 'r') as file:
    data = json.load(file)

os.environ['OPENAI_API_KEY'] = data['OPENAI_API_KEY'] 

# with open('chunks/0521_ChunkSize500Collection.json', 'r') as file:
#     all_chunks = json.load(file)

#lexical_index_path = 'chunks/0627_ChunkSize500LexicalIndex'


embedding_model_name = "text-embedding-3-small"
MAX_CONTEXT_LENGTHS = data['MAX_CONTEXT_LENGTHS']

client_chroma = chromadb.PersistentClient(path='./')
#ChunkSize500Collection = client_chroma.get_collection(name="0627_Test_Collection")
#ChunkSize500Collection = client_chroma.get_collection(name="0521_ChunkSize500Collection")

QA_dataset = pd.read_excel('QA Dataset.xlsx')
#print(type(QA_dataset.loc[:,'page'][0]))





chunk_size_list = ["500"]
num_chunks_list = [5]
lexical_search_k_list = [0,1,3]
llm_answer = "gpt-4-turbo"
llm_evaluate = "gpt-3.5-turbo"




result = []
total_start_time = time.time()
for chunk_size in chunk_size_list :
    chromadb_collection = client_chroma.get_collection(name= data['CHUNKS'][chunk_size]['chromadb_collection'])

    with open(data['CHUNKS'][chunk_size]['json_file'], 'r') as file:
        all_chunks = json.load(file)

    with open(data['CHUNKS'][chunk_size]['lexial_index_file'], 'rb') as bm25result_file:
        lexical_index = pickle.load(bm25result_file)

    for num_chunks in num_chunks_list:
    #for i in range(len(collections)) :
        for k in lexical_search_k_list :


            final_result = evaluate_RetrievalScore_AnswerQuality(
                use_lexical_search = True,
                chunks = all_chunks,
                lexical_index = lexical_index,
                embedding_model_name = embedding_model_name, 
                llm_answer = llm_answer,
                llm_evaluate = llm_evaluate,
                max_context_length = MAX_CONTEXT_LENGTHS[llm_answer],
                system_content = "Answer the query using the context provided. Be succinct. Do not produce any content that does not appear in the context.", 
                queries_dataset = QA_dataset, 
                num_chunks = num_chunks,
                lexical_search_k = k,
                chunk_size = int(chunk_size),
                vectordb_collection = chromadb_collection,
                stream = False
            )

            
            print(f'Test - num_chunks:{num_chunks}, lexical_search_chunk:{k}, chunk_size:{chunk_size} - finished')
            result.append(final_result)
        
total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time




# 將評估的結果儲存
result_df = pd.DataFrame(result)
result_df.drop(['detailed_evaluation'], axis=1).to_csv( path_or_buf = 'Experiment_Result.csv')


for index, row in result_df.iterrows():
    data = pd.DataFrame(row['detailed_evaluation'])
    data.to_excel(excel_writer = 'ChunkSize{}_NumChunks{}_LexicalSearchChunks{}_DetailedEvaluation.xlsx'.format(row['chunk_size'], row['num_chunks'], row['num_lexical_search_chunks']))

