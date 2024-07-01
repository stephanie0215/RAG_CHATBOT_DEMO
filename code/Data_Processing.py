from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
import json
import os
import itertools
from QueryAndEvaluation import chinese_word_preprocessing
from rank_bm25 import BM25Okapi
import pickle


# loader = PyPDFLoader("/Users/stephanie/Downloads/RRPG91070673.pdf", extract_images=True)
# pages = loader.load_and_split()

# print(pages[0].page_content)

# 
# loader = PyMuPDFLoader("files/產業設計創新趨勢與指引_全文.pdf")
# data = loader.load()
# print(data[66])

def list_files_in_directory(directory_path):
    try:
        # List all files in the directory
        files = os.listdir(directory_path)
        # Filter out directories, only keep files
        file_names = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]
        return file_names
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def get_loader(directory_path, file_name) :
    file_type = file_name.split('.')[1]
    full_path = os.path.join(directory_path, file_name)
    if (file_type == 'pdf') :
        return PyMuPDFLoader(full_path)
    else :
        return None
    
def document_to_dict(chunks):
    chunk_list = []
    for chunk in chunks :
        chunk_dict = {}
        chunk_dict['page_content'] = chunk.page_content
        chunk_dict['metadata'] = chunk.metadata
        chunk_list.append(chunk_dict)

    return chunk_list


def check_file_exists(filepath):
    return os.path.isfile(filepath)


def unique_items(lst):
    unique_list = []
    for item in lst:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list







class File_Processing :
    def __init__(self, directory_path, file_name) :
        self.directory_path = directory_path
        self.file_name = file_name

    def Read_File(self):
        #from langchain_community.document_loaders import PyMuPDFLoader
        
        loader = get_loader(directory_path = self.directory_path, file_name = self.file_name )
        if (loader) :
            self.pages = loader.load()
            return self.pages
        else : 
            self.pages = None

        
    
    @staticmethod
    def Embed_Documents(doc, model_name) :
        embedding_model = OpenAIEmbeddings(
                model=model_name,
                #openai_api_base=os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"])
        
        embeddings = embedding_model.embed_documents([doc.page_content])
        #print(len(embeddings))
        return embeddings[0]
    


    
    def Transformer_DocumentFormat(self, split_docs, embedding_model_name, file_id_name ) :
        metadatas = []
        documents = []
        embeddings = []
        ids = []

        i = 1
        for doc in split_docs :
            
            metadatas.append(doc.metadata)
            documents.append(doc.page_content)
            embeddings.append(self.Embed_Documents(doc = doc, model_name = embedding_model_name ))
            ids.append([file_id_name+str(i)])
            i += 1

            #print(doc.page_content)

        ids = [element for sublist in ids for element in sublist]    
        return metadatas, documents, embeddings, ids


    
    def SaveFilesToChromaDB(self, chunk_size, chunk_overlap, collection_name, client_chroma, embedding_model_name, file_id_name ) :
        from langchain_community.document_loaders import PyMuPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        import chromadb
        from chromadb.utils import embedding_functions

        #embeddings = OpenAIEmbeddings()
        #new_client = chromadb.PersistentClient()
  
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ".",
                "\u3002"  # Ideographic full stop
                ]
            ) 
        self.chunks  = text_splitter.split_documents(self.pages)
        
        metadatas, documents, embeddings, ids = self.Transformer_DocumentFormat( 
                                                                        split_docs = self.chunks, 
                                                                        embedding_model_name = embedding_model_name, 
                                                                        file_id_name = file_id_name  )
        
    
        collection = client_chroma.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        
        collection.upsert(
            documents= documents,
            embeddings= embeddings,
            metadatas = metadatas,
            ids = ids
        )
        print(f'Successfully save {self.file_name} into {collection_name}')
        #print(collection.count())

        return self.chunks
    
    def ChunksToJson(self, file_name) :
        self.chunks_json_file_name = file_name
        dict_chunks = document_to_dict(chunks = self.chunks)
        #print(dict_chunks)
        all_chunks = []
        if (check_file_exists(file_name)):
            with open(file_name, 'r') as file:
                exist_chunks = json.load(file)
                all_chunks.append(exist_chunks)
                #print('exist chunks\n' ,exist_chunks)

            all_chunks.append(dict_chunks)
            all_chunks = list(itertools.chain(*all_chunks))
            
        
        else :
            all_chunks = dict_chunks
        
        #all_chunks = list(set(all_chunks)) #only unique items would be saved into json file
        #all_chunks = unique_items(all_chunks)

        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(all_chunks,file,ensure_ascii=False)

        print(f"Chunks saved to {file_name}")

    def JsonToLexicalIndex(self, file_name) :
        with open(self.chunks_json_file_name, 'r') as file:
            all_chunks = json.load(file)
        #print(all_chunks)
        tokenized_text = []
        for chunk in all_chunks :
            post = chunk['page_content']
            tokenized_text.append(chinese_word_preprocessing(post))
            
        lexical_index = BM25Okapi(tokenized_text)

        #To save bm25 object
        with open(file_name, 'wb') as bm25result_file:
            pickle.dump(lexical_index, bm25result_file)

        print(f'Successfully save Lexical Index to {file_name}')








if __name__ == "__main__":
    with open('code/config.json', 'r') as file:
        data = json.load(file)

    os.environ['OPENAI_API_KEY'] = data['OPENAI_API_KEY'] 

        
    embedding_model_name = "text-embedding-3-small"
    client_chroma = chromadb.PersistentClient(path='./')
    chroma_db_collection_name = 'ChunkSize500Collection' #ChunkSize100
    json_file_name = 'chunks/ChunkSize500.json'
    lexical_index_file_name = 'chunks/ChunkSize500_LexicalIndex'
    #file_id_name = "RoadTrafficSafetyRules"
    FILE_ID = data['FILE_ID']
    chunk_size=500
    chunk_overlap=50
    directory_path = 'files'
    files = list_files_in_directory(directory_path)
 
    for file_name in files:
        #print(file_name)
        file_processor = File_Processing( directory_path = directory_path , file_name=file_name )
        file_processor.Read_File()
        if (file_processor.pages) :
            file_processor.SaveFilesToChromaDB(chunk_size=chunk_size, 
                                chunk_overlap=chunk_overlap,
                                collection_name = chroma_db_collection_name, 
                                client_chroma = client_chroma, 
                                embedding_model_name = embedding_model_name, 
                                file_id_name= FILE_ID[file_name] )
            
            file_processor.ChunksToJson(file_name = json_file_name)
            file_processor.JsonToLexicalIndex(file_name = lexical_index_file_name)

        

    collection = client_chroma.get_collection(name=chroma_db_collection_name)
    print(collection.count())




        
    





    











        

