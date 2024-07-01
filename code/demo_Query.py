# -*- coding: utf-8 -*-

from QueryAndEvaluation import QueryAgent
import json
import os
import chromadb
import pandas as pd
import time
import pickle




with open('/Users/stephanie/實習/履歷/潮網/自行車_chatbot/code/config.json', 'r') as file:
    data = json.load(file)

os.environ['OPENAI_API_KEY'] = data['OPENAI_API_KEY'] 



embedding_model_name = "text-embedding-3-small"
use_lexical_search = True 
lexical_search_k = 3
chunk_size = "500"
num_chunks = 5
llm = 'gpt-4-turbo'
MAX_CONTEXT_LENGTHS = data['MAX_CONTEXT_LENGTHS']


with open(data['CHUNKS'][chunk_size]['json_file'], 'r') as file:
    all_chunks = json.load(file)

with open(data['CHUNKS'][chunk_size]['lexial_index_file'], 'rb') as bm25result_file:
    lexical_index = pickle.load(bm25result_file)

client_chroma = chromadb.PersistentClient(path='./')
collection = client_chroma.get_collection(name= data['CHUNKS'][chunk_size]['chromadb_collection'])
#collection = client_chroma.get_collection(name="0521_ChunkSize500Collection")


#query = 'What is “second great inflection point” for mobility ?'
query = '請簡單介紹Ming Cycle'

start_time = time.time()
agent = QueryAgent(
    use_lexical_search=use_lexical_search, 
    chunks = all_chunks,
    lexical_index = lexical_index,
    embedding_model_name=embedding_model_name,
    llm=llm,
    max_context_length= MAX_CONTEXT_LENGTHS[llm],
    system_content= "Answer the query using the context provided. Be succinct. Do not produce any content that does not appear in the context.")



generated_answer_result = agent(
                lexical_search_k = lexical_search_k,
                query=query,
                num_chunks = num_chunks, 
                collection = collection,
                stream=False)

end_time = time.time()



print(f"question :{query}" )

print(f"answer : {generated_answer_result['answer']}")

#generated_answer_result_df = pd.DataFrame(generated_answer_result)
print(' total time spent:', end_time - start_time )



#generated_answer_result_df.to_csv('result.csv')
