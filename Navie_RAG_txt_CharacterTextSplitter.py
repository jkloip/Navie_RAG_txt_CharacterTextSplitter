#-------------------------------------------------------------------------------
# Name:        基礎檢索增強生成 (Navie RAG) 範例程式
# Purpose:     展示 RAG 的基本流程、提供實作參考，為後續更進階的 RAG 研究和開發奠定了基礎。
#              
# Author:      jkloip
#
# Created:     07/02/2025
# Copyright:   (c) jkloip 2025
#-------------------------------------------------------------------------------

# 第1步-引用模組，連接 OpenAI、Groq、Google、Ollama 的 LLM 模型
import os  # 引入 Python 系統模組，用來進行與作業系統相關的操作，如讀取環境變數、路徑處理等
# pip install python-dotenv  安裝 python-dotenv 套件，用來讀取 .env 檔案中的環境變數
from dotenv import load_dotenv  # 從 dotenv 套件引入 load_dotenv 函式，此函式可以從 .env 檔案中讀取環境變數，不需將敏感資訊硬編碼在程式中
# pip install rich  安裝 rich 套件，用來格式化輸出、顯示彩色文字等
from rich import print as richprint  # 從 rich 套件匯入 print 函式，並重新命名為 richprint，可用來格式化輸出、顯示彩色文字等

# pip install langchain_openai  langchain_groq  langchain_google_genai  langchain_ollama 安裝 langchain_openai、langchain_groq、langchain_google_genai、langchain_ollama 套件
from langchain_openai import ChatOpenAI  # 從 langchain_openai 模組引入 ChatOpenAI 類別，此類別可用於與 OpenAI 的聊天模型做互動
from langchain_groq import ChatGroq  # 從 langchain_groq 模組引入 ChatGroq 類別，此類別用於操作 Groq 平台上的聊天模型做互動
from langchain_google_genai import ChatGoogleGenerativeAI  # 從 langchain_google_genai 模組引入 ChatGoogleGenerativeAI 類別，可用於跟 Google 的聊天模型做互動
from langchain_ollama import ChatOllama  # 從 langchain_ollama 模組引入 ChatOllama 類別，此類別可用於與 Ollama 的聊天模型做互動

load_dotenv()  # 呼叫 load_dotenv() 函式以從 .env 檔案中載入環境變數，讓程式可以使用設定好的 API 金鑰或其他設定值

# OpenAI GPT-4o-mini
def openai_generate_response(prompt):
    # 定義 openai_generate_response 函式，使用 OpenAI 的 GPT-4o-mini 模型生成回應。
    # 輸入參數 prompt 為使用者或應用程式提供的提示文字。
    
    chat_openai = ChatOpenAI(
        model="gpt-4o-mini", 
        api_key=os.getenv("OPENAI_API_KEY"), 
        temperature=0
    )  
    # 建立一個 ChatOpenAI 物件，設定:
    # 1. 模型名稱為 "gpt-4o-mini"
    # 2. 使用 os.getenv("OPENAI_API_KEY") 從環境變數中取得 API 金鑰
    # 3. temperature=0 表示生成回應時採用低溫度，使生成結果較穩定
    
    response = chat_openai.invoke(prompt)  
    # 呼叫 chat_openai 物件的 invoke 方法，傳入提示文字 prompt，開始生成回應
    # 此方法運作方式為非串流模式，將完整回應一次性返回
    
    return response  
    # 將模型生成的回應返回給呼叫方

# Groq Llama3-8b-8192
def groq_generate_response(prompt):
    # 定義 groq_generate_response 函式，使用 Groq 平台的 Llama3-8b-8192 模型生成回應
    # 輸入參數 prompt 為提示文字
    
    chat_groq = ChatGroq(
        model="llama3-8b-8192", 
        api_key=os.getenv("GROQ_API_KEY"), 
        temperature=0
    )  
    # 建立一個 ChatGroq 物件，設定:
    # 1. 模型名稱為 "llama3-8b-8192"
    # 2. 從環境變數中讀取 GROQ_API_KEY 作為 API 金鑰
    # 3. temperature=0 同樣保證生成回應的穩定性
    
    response = chat_groq.invoke(prompt)  
    # 呼叫 chat_groq 物件的 invoke 方法，傳入提示文字 prompt，生成相應的回應    
    # 此方法運作方式為非串流模式，將完整回應一次性返回
    
    return response  
    # 將模型生成的回應返回給呼叫方

# Google gemini-1.5-flash
def google_generate_response(prompt):
    # 定義 google_generate_response 函式，使用 Google AI Studio 平台的 gemini-1.5-flash-8b 模型生成回應
    # 輸入參數 prompt 為提示文字

    chat_google = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        api_key=os.getenv("GEMINI_API_KEY"), 
        temperature=0
    )
    # 建立一個 ChatGoogleGenerativeAI 物件，設定:
    # 1. 模型名稱為 "gemini-1.5-flash-8b"
    # 2. 從環境變數中讀取 GEMINI_API_KEY 作為 API 金鑰
    # 3. temperature=0 同樣保證生成回應的穩定性

    response=chat_google.invoke(prompt) # 啟用非串流模式
    # 呼叫 chat_google 物件的 invoke 方法，傳入提示文字 prompt，生成相應的回應    
    # 此方法運作方式為非串流模式，將完整回應一次性返回

    return response
    # 將模型生成的回應返回給呼叫方

# Ollama Llama3.2 1B
def ollama_generate_response(prompt):
    # 定義 ollama_generate_response 函式，使用 local ollama 平台的 llama3.2:1b 模型生成回應
    # 輸入參數 prompt 為提示文字

    chat_ollama = ChatOllama(
        base_url='http://localhost:11434',
        model="llama3.2:1b", 
        temperature=0
    )
    # 建立一個 ChatOllama 物件，設定:
    # 1. 模型名稱為 "llama3.2:1b"
    # 2. base_url='http://localhost:11434' 指定 Ollama 服務的 URL
    # 3. temperature=0 同樣保證生成回應的穩定性

    response=chat_ollama.invoke(prompt) # 啟用非串流模式
    # 呼叫 chat_ollama 物件的 invoke 方法，傳入提示文字 prompt，生成相應的回應    
    # 此方法運作方式為非串流模式，將完整回應一次性返回

    return response 
    # 將模型生成的回應返回給呼叫方

# 測試 LLM 模型回應
# richprint(groq_generate_response("What is the capital of France?"))  # 使用 LLM 模型回答「法國的首都是哪裡？」

# 第2步-引用文字切割器模組，將長文切割成多個短文區塊
# pip install langchain-text-splitters  安裝 langchain-text-splitters 套件，用來進行文字切割
from langchain_text_splitters import CharacterTextSplitter  # 匯入 CharacterTextSplitter 類別

# 讀取「2024美國國情咨文」檔案內容，該檔案內容為整篇美國國情咨文的文字
with open("2024_state_of_the_union.txt", encoding='utf-8') as f:  # 使用 UTF-8 編碼開啟檔案
    state_of_the_union = f.read()  # 將檔案中的內容讀取為一個大字串變數

# 測試列印國情咨文的前 1000 個字元，以確認讀取是否成功
# richprint(state_of_the_union[:1000])  # 此行原先印出國情咨文的前 1000 個字元
# print(type(state_of_the_union))  # 列印 state_of_the_union 變數的資料型別

# 初始化文字切割器，設定切割參數
text_splitter = CharacterTextSplitter(
    chunk_size=512,    # 每個切割後的區塊最大包含 512 個字元
    chunk_overlap=100,  # 區塊之間有 100 個字元的重疊，讓上下文得以連貫
    length_function=len # 使用內建的 len 函式計算字串長度
)

# 測試列印文字切割器的資料型別
# print(type(text_splitter))  # 印出 text_splitter 變數的資料型別

# 使用 text_splitter 文字切割器將整篇國情咨文切割成多個區塊
# 補充： create_documents 方法傳入的參數為一個列表，裡面可以包含多個文件。用 `[...]` 將字串包起來，可將單一字串轉換成含有一個元素的列表
texts = text_splitter.create_documents([state_of_the_union])  # 傳入包含全文的串列來產生多個文檔區塊

# 測試列印切割後的文檔區塊，以確認切割是否成功
# richprint(texts)  # 此行原先印出所有切割後的區塊
# print(len(texts))  # 列印切割後的文檔區塊數量
# print(type(texts))  # 列印切割後的文檔區塊資料型別

# 第3步-引用向量資料庫模組 Chroma 與 OpenAI、Hugging Face 的向量嵌入模組 Embeddings，將文檔區塊轉換為向量表示
# 匯入向量資料庫模組 Chroma 與 OpenAI 的向量嵌入（Embeddings）模組
# pip install langchain-chroma  安裝 langchain-chroma 套件，用來建立向量資料庫
from langchain_chroma import Chroma  # 匯入 Chroma，作為向量資料庫管理工具
from langchain_openai import OpenAIEmbeddings  # 匯入 OpenAIEmbeddings，用來將文字轉換成向量表示

# 建立 OpenAI 嵌入模型，指定使用 "text-embedding-3-large" 模型來轉換文字為數值向量
# openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # 效果佳，但要花錢

# 使用 Hugging Face 的模型來進行嵌入
# pip install sentence-transformers  安裝 sentence-transformers 套件，用來進行 Hugging Face 模型的嵌入
# pip install langchain-huggingface  安裝 langchain-huggingface 套件，用來進行 Hugging Face 模型的嵌入
from langchain_huggingface import HuggingFaceEmbeddings

# model_name = "sentence-transformers/all-mpnet-base-v2" # 效果不佳
model_name="intfloat/multilingual-e5-large-instruct" # 效果佳，但要花時間除非有 GPU
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 利用 Chroma 建立向量資料庫，並指定資料集合名稱與嵌入函式
vector_store = Chroma(
    collection_name="test_collection",  # 設定文件集合名稱為 "test_collection"
    embedding_function=hf         # 使用上面建立的 embeddings 物件作為向量轉換函式
)

# 測試列印向量資料庫的資料型別
# print(type(vector_store))  # 列印向量資料庫的資料型別

# 將前面切割好的文檔區塊 texts 存入向量資料庫 vector_store 中，並將回傳的文件 ID 儲存於 ids 變數
# 補充：add_documents(...)方法會將這些文檔處理過程中，進行向量嵌入計算，將文字轉成數值向量，並將結果儲存到向量資料庫中
# 補充：ID 在日後可透過相似度來搜尋或其他操作
# 補充：目前文件只存放在記憶體中，尚未寫入硬碟 - 實際應用中，需將向量資料庫寫入到硬碟
ids = vector_store.add_documents(texts)

# 測試列印文檔區塊的 ID，以確認文檔已經成功存入向量資料庫
# richprint(ids)  # 列印文檔區塊的 ID，以確認文檔已經成功存入向量資料庫
# print(type(ids))  # 列印文檔區塊的 ID 的資料型別
# print(len(ids))  # 列印文檔區塊的 ID 的數量

# 測試在向量資料庫中進行相似度搜尋，查詢問題為「Who invaded Ukraine?」
# 並要求回傳最相似的前 2 筆結果
'''
results = vector_store.similarity_search(
    'Who invaded Ukraine?',  # 查詢內容：詢問「誰入侵了烏克蘭？」
    k=2                      # 回傳最相似的前 2 筆結果
)
# 測試列印相似度搜尋結果-完整內容
richprint("相似度搜尋結果：", results)  # 列印相似度搜尋結果

# 測試列印相似度搜尋結果-依需求
for res in results:
    print(f" 😄 {res.id} 😎 [{res.metadata}] 🚀 {res.page_content} \n\n")
'''

# 第4步-將 Chroma 向量資料庫轉換為檢索器，用於後續的資訊檢索處理
retriever = vector_store.as_retriever()

# 測試使用檢索器進行查詢，問題為「Who invaded Ukraine?」，並要求回傳最相似的前 2 筆結果
# query = 'Who invaded Ukraine?'  # 查詢問題
# retriever_results = retriever.invoke(query, k=2)
# richprint("檢索結果：", retriever_results)  # 列印檢索結果

# 第5步-定義一個文件格式化函式，將檢索到的文檔整合為一個字串
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)  # 每個文檔內容以兩個換行符號連接

# 第6步-引用提示模板模組，建立生成模型需要的提示訊息
# pip install langchain-core  安裝 langchain_core 套件，用來建立提示模板
from langchain_core.prompts import PromptTemplate  # 匯入 PromptTemplate 類別

# 建立用來引導語言模型回答問題的提示模板
prompt_template = """
請根據提供的上下文回答用戶提出的問題。如果您根據提供的上下文不知道答案，
請告知用戶"根據提供的上下文無法回答問題"，並向用戶致歉。

上下文內容: {context}

用戶的問題: {query}

答案: """
# 此多行字串模板規定：若基於給定的上下文無法回答問題，模型需回應「不知道」
# 並以明確格式呈現上下文、問題與答案區塊

# 利用上面的模板字串建立提示物件
custom_rag_prompt = PromptTemplate.from_template(prompt_template)

# 測試列印提示模板
# richprint(custom_rag_prompt)  # 列印提示模板

# 第7步-引用輸出解析器與通過運行器模組，建立 RAG 鏈-整合檢索、提示模板、生成及解析等流程
# StrOutputParser 用於解析語言模型回傳的純文字答案，
# RunnablePassthrough 用於直接傳遞查詢文字，不做任何轉換
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 建立 RAG（檢索與生成）鏈，整合檢索、提示模板、生成及解析等流程
rag_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}  # 將「context」欄位設為由檢索器取得後再格式化文件，
    | custom_rag_prompt                                               # 將查詢與上下文資訊套用到自定提示模板上
    | groq_generate_response                                          # 呼叫 openai_generate_response 生成模型回答
    | StrOutputParser()                                                 # 解析生成模型回應為純文字答案
)

# 測試使用 RAG 鏈進行查詢，問題內容根據 2024 國情咨文，且要求模型以中文回答
# 利用 RAG 鏈進行查詢，問題內容根據 2024 美國國情咨文，且要求模型以中文回答
ans1 = rag_chain.invoke("根據2024年的國情咨文演講，誰入侵了烏克蘭？ 請用繁體中文台灣用語回答")
print(ans1)  # 將第一個查詢的結果印出

# 進行第二個查詢，問題為「生命的意義為何？」程式預期模型會回答「不知道」
ans2 = rag_chain.invoke("生命的意義是什麼？ 請用繁體中文台灣用語回答")
print(ans2)  # 將第二個查詢的結果印出
