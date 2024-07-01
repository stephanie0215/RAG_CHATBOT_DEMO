# -*- coding: utf-8 -*-
import pandas as pd
import json
import numpy as np
import os
import time
import tiktoken
from openai import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
import re
from rank_bm25 import BM25Okapi
import jieba
import pickle


def semantic_search(query, embedding_model, collection, k ):
    embedding = embedding_model.embed_query(query)
    result = collection.query(
        query_embeddings=embedding,
        n_results=k
    )
    #result_df = pd.DataFrame(result)
    semantic_context = []
    for i in range (k) :
        file_name = result['metadatas'][0][i]['source']
        file_page = result['metadatas'][0][i]['page']
        source_file_name = handler_source_format( retrieval_reference = 'file_name', 
                                        file_name = file_name, 
                                        file_page = file_page)
        
        source_file_name_and_page = handler_source_format( retrieval_reference = 'file_name_and_page', 
                                file_name = file_name, 
                                file_page = file_page)
        semantic_context.append({'id' :result['ids'][0][i], 
                                 "text" : result['documents'][0][i], 
                                 "source_file_name" : source_file_name , 
                                 "source_file_name_and_page" : source_file_name_and_page,
                                 "method" : 'semantic_search'})
    return semantic_context



def contains_only_chinese_punctuation(text_list):
    # Define a regular expression pattern to match Chinese punctuation characters
    chinese_punctuation_pattern = re.compile(r'[\u3000-\u303F\uFF00-\uFFEF]+')
    return len(text_list) > 0 and not bool(chinese_punctuation_pattern.sub('', text_list))

def contains_only_newline_characters(text_list) :
    newline_pattern = re.compile(r'^\n+$')
    
    # Filter out elements containing only newline characters
    filtered_list = [text for text in text_list if not newline_pattern.match(text)]

    return filtered_list
    


def chinese_word_preprocessing(text) :
    tokenized_text = list(jieba.cut(text))
    filtered_text = [text for text in tokenized_text if not contains_only_chinese_punctuation(text)]
    filtered_text = contains_only_newline_characters(filtered_text)
    return filtered_text
    







def lexical_search(index, query, chunks, k):
    #query_tokens = query.lower().split()  # preprocess query
    query_tokens = chinese_word_preprocessing(query)
    scores = index.get_scores(query_tokens)  # get best matching (BM) scores
    indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]  # sort and get top k
    #print(indices)

    lexical_context = []
    for i in indices :
        file_name = chunks[i]['metadata']['source']
        file_page = chunks[i]['metadata']['page']
        source_file_name = handler_source_format( retrieval_reference = 'file_name', 
                                        file_name = file_name, 
                                        file_page = file_page)
        source_file_name_and_page = handler_source_format( retrieval_reference = 'file_name_and_page', 
                                file_name = file_name, 
                                file_page = file_page)
        
        lexical_context.append(
            {
                "text": chunks[i]['page_content'], 
                "source_file_name": source_file_name, 
                "source_file_name_and_page" : source_file_name_and_page,
                "score": scores[i], 
                "method" : 'lexical_search'

            }
        )
        

    
    # lexical_context = [{
            
    #         "text": chunks[i]['page_content'], 
    #         "source": chunks[i]['metadata']['source'], 
    #         "score": scores[i], 
    #         "method" : 'lexical_search'} for i in indices]
   
    return lexical_context



def response_stream(chat_completion):
    for chunk in chat_completion:
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content

def prepare_response(chat_completion, stream):
    if stream:
        return response_stream(chat_completion)
    else:
        return chat_completion.choices[0].message.content
def get_client(llm):
    if llm.startswith("gpt"):
        #base_url = os.environ["OPENAI_API_BASE"]
        api_key = os.environ["OPENAI_API_KEY"]
    else:
        base_url = os.environ["ANYSCALE_API_BASE"]
        api_key = os.environ["ANYSCALE_API_KEY"]
    client = OpenAI(api_key=api_key)
    return client



def generate_response(
    llm,
    max_tokens=None,
    temperature=0.0,
    stream=False,
    system_content="",
    assistant_content="",
    user_content="",
    max_retries=1,
    retry_interval=60,
):
    """Generate response from an LLM."""
    retry_count = 0
    client = get_client(llm=llm)
    messages = [
        {"role": role, "content": content}
        for role, content in [
            ("system", system_content),
            ("assistant", assistant_content),
            ("user", user_content),
        ]
        if content
    ]
    while retry_count <= max_retries:
        try:
            chat_completion = client.chat.completions.create(
                model=llm,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                messages=messages,
            )
            return prepare_response(chat_completion, stream=stream)

        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return ""





def get_embedding_model(embedding_model_name, model_kwargs, encode_kwargs):

    embedding_model = OpenAIEmbeddings(
        model=embedding_model_name,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    return embedding_model

def get_num_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def trim(text, max_context_length):   ##if 文本長度過長，超過model的input，則使用trim，只取前面的文本，後面通通忽略
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(text)[:max_context_length])


def handler_source_format( retrieval_reference, file_name, file_page ) :
    file_name = file_name.split("/")[-1]
    if (retrieval_reference == 'file_name') :
        source = file_name
    elif (retrieval_reference == 'file_name_and_page') :
        file_page = int(file_page)
        file_page = file_page + 1
        source = f"{file_name} p.{file_page}" 
        #file_name.append(source.split("/")[-1])
    
    return source



class QueryAgent:
    def __init__(self, 
                 chunks, 
                 lexical_index, 
                 use_lexical_search=True, 
                 embedding_model_name="text-embedding-ada-002",
                 llm="gpt-3.5-turbo-1106", 
                 temperature=0.0, 
                 max_context_length=4096, 
                 system_content="", 
                 assistant_content=""):

        self.chunks = chunks
        self.lexical_index = None
        if use_lexical_search:
            # tokenized_text = []
            # for chunk in chunks :
            #     post = chunk['page_content']
            #     tokenized_text.append(chinese_word_preprocessing(post))
                
            # self.lexical_index = BM25Okapi(tokenized_text)

            # with open(lexical_index, 'rb') as bm25result_file:
            #     self.lexical_index = pickle.load(bm25result_file)

            self.lexical_index = lexical_index



        
        # Embedding model
        self.embedding_model = get_embedding_model(
            embedding_model_name=embedding_model_name, 
            model_kwargs={"device": "cuda"}, 
            encode_kwargs={"device": "cuda", "batch_size": 100})
        
        # Context length (restrict input length to 80% of total context length)
        max_context_length = int(0.8*max_context_length)
        
        # LLM
        self.llm = llm
        self.temperature = temperature
        self.context_length = max_context_length - get_num_tokens(system_content + assistant_content)
        self.system_content = system_content
        self.assistant_content = assistant_content

    def __call__(self, lexical_search_k, query, collection, num_chunks=5, stream=True):
        # Get sources and context
        context_results = semantic_search(
            query=query, 
            embedding_model=self.embedding_model,
            collection = collection,
            k=num_chunks) #k : 相似度最高的前k個文本
        
        if self.lexical_index:
            lexical_context = lexical_search(
                index=self.lexical_index, 
                query=query, 
                chunks=self.chunks, 
                k=lexical_search_k)
            # Insert after <lexical_search_k> worth of semantic results
            context_results[lexical_search_k:lexical_search_k] = lexical_context
            #print('length of lexical content' , len(lexical_context))

        #print('length of context used:', len(context_results))
        #print('context_results' , context_results)
            
        # Generate response
        context = [item["text"] for item in context_results]
        source_file_name = [item["source_file_name"] for item in context_results]
        source_file_name_and_page = [item["source_file_name_and_page"] for item in context_results]
        #sources = handler_source_format(sources = sources)
        methods = [item["method"] for item in context_results]
        user_content = f"query: {query}, context: {context}"
        answer = generate_response(
            llm=self.llm,
            temperature=self.temperature,
            stream=stream,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=trim(user_content, self.context_length))

        # Result
        result = {
            "question": query,
            "source_file_name": source_file_name,
            "source_file_name_and_page": source_file_name_and_page,
            "methods" : methods,
            "answer": answer,
            "llm": self.llm,
            "context" : context
        }
        return result
    


# def get_retrieval_score(references, generated):
#     matches = np.zeros(len(references))
#     for i in range(len(references)):
#         reference_source = references[i]
#         if not reference_source:
#             matches[i] = 1
#             continue
#         for source in generated[i]:
#             # sections don't have to perfectly match
#             if (source == reference_source) :
#                 matches[i] = 1
#                 continue

#     retrieval_score = np.mean(matches)
#     return retrieval_score

def get_retrieval_score(reference, generated):
    #matches = np.zeros(len(references))
    match = 0
    for source in generated:
        # sections don't have to perfectly match
        if (source == reference) :
            match = 1
            break    
    return match


def get_answer_evaluation(query, llm, generated_answer, reference_answer , max_context_length):

    # 提供明確的標準
    evaluation_system_content = """
    Your job is to rate the quality of our generated answer {generated_answer}
    given a query {query} and a reference answer {reference_answer}.
    Your score has to be between 1 and 5.
    You must return your response in a line with only the score.
    Do not return answers in any other format.
    On a separate line provide your reasoning for the score as well.
    Note : your reasoning should be traditional Chinese, rather than English.
    """
    # Generate response
    context_length = int(0.8 * (max_context_length - len(evaluation_system_content)))
    user_content = trim(
            str(
                {
                    "query": query,
                    "generated_answer": generated_answer,
                    "reference_answer": reference_answer,
                }
            ),
            context_length,
        )

    response = generate_response(
    llm=llm,
    temperature=0.0,
    stream=False,
    system_content=evaluation_system_content,
    user_content=user_content)

    score, reasoning = response.split("\n", 1) if "\n" in response else (0, "")
    result = {
        "question": query,
        "generated_answer": generated_answer,
        "reference_answer": reference_answer,
        "evaluation_score": float(score),
        "reasoning": reasoning.lstrip("\n")
        #"sources": sources
    }

    return result



def get_average_score (data, attribute) :
    evaluation_score_accumulation = 0
    for i in range(len(data)) :
        evaluation_score_accumulation += data[i][attribute]
    
    average_evaluation_score = evaluation_score_accumulation/len(data)
    return average_evaluation_score

    
    


def evaluate_RetrievalScore_AnswerQuality(use_lexical_search,
                                          chunks,
                                          lexical_index,
                                          embedding_model_name, 
                                          llm_answer,
                                          llm_evaluate,  
                                          max_context_length, 
                                          system_content, 
                                          queries_dataset, 
                                          num_chunks, 
                                          lexical_search_k,
                                          chunk_size, 
                                          vectordb_collection, 
                                          #retrieval_reference, 
                                          stream = False
                                         ) :
    
    
    start_time = time.time()

    
    agent = QueryAgent(
        use_lexical_search=use_lexical_search, 
        chunks = chunks,
        lexical_index = lexical_index,
        embedding_model_name=embedding_model_name,
        llm=llm_answer,
        max_context_length=max_context_length,
        system_content=system_content, 
        )


    evaluation = []
    for index, row in queries_dataset.iterrows() : 

        start_question_time = time.time()
        generated_answer_result = agent(
                       lexical_search_k = lexical_search_k,
                       query=row['question'],
                       num_chunks = num_chunks, 
                       collection = vectordb_collection,
                       #retrieval_reference = retrieval_reference,
                       stream=False)
        end_question_time = time.time()

        
        
        evaluate_generated_answer_result = get_answer_evaluation(query =row['question'], 
            llm = llm_evaluate,
            generated_answer = generated_answer_result['answer'], 
            reference_answer = row['reference_answer'], 
            max_context_length = max_context_length)

        
        file_name_retrieval_score = get_retrieval_score( reference = row['source'], 
                                                         generated = generated_answer_result['source_file_name'] )

        file_name_and_page_retrieval_score = get_retrieval_score( reference =  row['source'] + " p." + str(row['page']) , 
                                                                  generated = generated_answer_result['source_file_name_and_page'])
        
        
    
        question_time = end_question_time - start_question_time
        evaluate_generated_answer_result['generate_answer_time_spent(seconds)'] = question_time
        evaluate_generated_answer_result['methods'] = generated_answer_result['methods']
        evaluate_generated_answer_result['file_name_retrieval_score'] = file_name_retrieval_score
        evaluate_generated_answer_result['file_name_and_page_retrieval_score'] = file_name_and_page_retrieval_score
        evaluate_generated_answer_result['source_file_name'] = generated_answer_result['source_file_name']
        evaluate_generated_answer_result['source_file_name_and_page'] = generated_answer_result['source_file_name_and_page']
        evaluate_generated_answer_result['context'] = generated_answer_result['context']
        evaluate_generated_answer_result['llm_answer'] = llm_answer
        evaluate_generated_answer_result['llm_evaluate'] = llm_evaluate

        evaluation.append(evaluate_generated_answer_result)


    

 
    average_evaluation_score = get_average_score( data = evaluation , 
                                                  attribute = 'evaluation_score')
    average_file_name_retrieval_score = get_average_score( data = evaluation , 
                                                           attribute = 'file_name_retrieval_score')
    average_file_name_and_page_retrieval_score = get_average_score( data = evaluation , 
                                                                    attribute = 'file_name_and_page_retrieval_score')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    

    final_result = {
        'file_name_retrieval_score' : average_file_name_retrieval_score, 
        'file_name_and_page_retrieval_score' : average_file_name_and_page_retrieval_score,
        'average_evaluation_score' : average_evaluation_score,
        'num_chunks' : num_chunks, 
        'chunk_size' : chunk_size,
        'num_lexical_search_chunks' : lexical_search_k if use_lexical_search else 0,
        'embedding_model' : embedding_model_name,
        'llm_answer' : llm_answer,
        'llm_evaluate' : llm_evaluate,
        'detailed_evaluation' : evaluation,
        'time_spent' : pd.DataFrame(evaluation)['generate_answer_time_spent(seconds)'].sum()
        
    }
    return final_result
    
    





    

