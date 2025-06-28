import os
import sys
import openai
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain import PromptTemplate, OpenAI, LLMChain
import streamlit as st
import uuid
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from pyhive import hive
import pandas as pd
from datetime import datetime, date
from defineprompts import classPrompts
# from ChatHistory2Server import insert_query, update_query, run_query
from Embeddings import classEmbdding
from langchain.vectorstores.chroma import Chroma
from Get_answers import answer_obj
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT

load_dotenv(find_dotenv())

openai.api_version = "2020-10-01"

# print(openai.api_key)
embeddings = OpenAIEmbeddings(openai_api_key = os.getenv("openai_api_key"))
# print("embeddings done")


# model_name = "gpt-4"
llm_model = OpenAI(openai_api_key = os.getenv("openai_api_key"))


filtered_df = pd.DataFrame()
st.set_page_config(page_title="Chatbot")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'state' not in st.session_state:
    st.session_state['state'] = []

if 'source' not in st.session_state:
    st.session_state['source'] = []

if 'prompt' not in st.session_state:
    st.session_state['prompt'] = []

if 'time_generated' not in st.session_state:
    st.session_state['time_generated'] = []

if 'date_generated' not in st.session_state:
    st.session_state['date_generated'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [("", ""),]

if 'MetaData_source' not in st.session_state:
    st.session_state['MetaData_source'] = []



emb_obj = classEmbdding(chroma_url="./Chroma", data_url="./", subfolder="Data")
promptObj = classPrompts(prompt_file_loc="./Prompts/Prompts.xlsx")
promptObj.def_prompt_file()
file_list_Chroma = os.listdir(emb_obj.chroma_url)
st.title('PDF Chatbot')
st.sidebar.header('Please select the file that you want to use')
    
filename=st.sidebar.selectbox("Filename", file_list_Chroma)
promptObj.search_prompt_file(filename) # creates obj.prompt_template
promptObj.Func_Prompt() # creates obj.prompt
if filename:
    st.session_state['filename']=filename

clear_chat_history = st.sidebar.button("Clear Chat History")
if clear_chat_history:
    st.session_state['chat_history'] = [("", ""),]


chain_type_kwargs = {"prompt": promptObj.Prompt}



user_input=st.chat_input()
vectordb = emb_obj.find_Vectordb(st.session_state['filename'])
Answer_obj = answer_obj(st.session_state['chat_history'], promptObj.Prompt, CONDENSE_QUESTION_PROMPT, llm_model,vectordb,st.session_state['filename'])



if user_input:
    # result1, chat_history,similiarity, source_vect = Answer_obj.get_ans_list(1,100,user_input)
    result1, chat_history,similiarity, source_vect = Answer_obj.find_answers(user_input)
    
    response = result1
    st.session_state['chat_history'] = Answer_obj.chat_history
    
    # print(st.session_state['chat_history'])
    st.session_state['MetaData_source'] = source_vect
    # print(st.session_state['MetaData_source'][0][0:-1])
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)
    st.session_state.source.append(filename)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    d1 = today.strftime("%d-%m-%Y")
    st.session_state.time_generated.append(current_time)
    st.session_state.date_generated.append(d1)
    

if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated'])-1, -1, -1)):
        with st.chat_message("user"):
            st.write(st.session_state['past'][i])
        with st.chat_message("assistant"):
            st.write(st.session_state["generated"][i]+"\n\n\n This query was answered from "+st.session_state["source"][i]+".")
    dislike_button = st.button(":thumbsdown:")
    if len(st.session_state.state)<len(st.session_state.generated):
        st.session_state.state.append("Like")
    if dislike_button:
        st.session_state.state[-1]="Dislike"

    question = st.session_state.past[-1]
    answer = st.session_state.generated[-1]
    answer_status = st.session_state.state[-1]
    filename =st.session_state.source[-1]

    # query1 = insert_query(question,answer,answer_status,filename,st.session_state.time_generated[-1],st.session_state.date_generated[-1])
    # query2 = update_query(question,answer,filename,st.session_state.time_generated[-1],st.session_state.date_generated[-1])
