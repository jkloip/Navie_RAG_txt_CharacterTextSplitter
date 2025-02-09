# pip install streamlit  安裝 streamlit 套件，用於建立網頁式使用者介面
import streamlit as st  # 匯入 Streamlit 套件，用於建立網頁式使用者介面
from Navie_RAG_txt_CharacterTextSplitter import rag_chain  # 從 Navie_RAG_txt_CharacterTextSplitter_persist 模組匯入 rag_chain，此物件包含檢索與生成的工作流程
import datetime  # 匯入 datetime 模組以取得當下時間

st.title("LLM 文件(txt)諮詢系統")  # 設定網頁標題，建立主標題文字
st.subheader("用 Python 與 Langchain 編寫基礎型檢索增強生成 (Navie RAG)系統")  # 設定副標題，說明系統技術背景與目的
st.write("此系統提供基於 TXT 文件的問答諮詢，您輸入的問題，系統將根據 TXT 文件內容來提供答案。") # 輸出系統說明文字，告知使用者如何使用此查詢系統

query = st.text_input("請輸入您的問題：") # 建立一個文字輸入框，讓使用者輸入查詢問題，輸入後字串儲存在變數 query 中

status_placeholder = st.empty() # 建立一個 placeholder 區域，方便後續動態更新或清除狀態訊息

if st.button("送出問題"):
    if query:  # 使用者有輸入文字時
        status_placeholder.info("系統查詢中...")  # 顯示查詢中訊息
                
        answer = rag_chain.invoke(query)  # 呼叫 RAG 鏈，取得回答
                
        status_placeholder.empty()  # 取得回答後清除查詢中提示
        
        st.success("查詢結果：")  # 顯示查詢結果

        st.write(answer)  # 輸出回答

        # 取得當前時間，並格式化為 YYYY-MM-DD HH:MM:SS
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 將查詢記錄寫入 CSV 檔案
        with open("query_history.csv", "a", encoding="utf-8") as log_file:
            log_file.write(f"{current_time},\"{query}\",\"{answer}\"\n")
            
    else:
        st.error("請輸入查詢文字！")  # 若無輸入則提示錯誤
