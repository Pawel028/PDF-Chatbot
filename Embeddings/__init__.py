from email.mime.application import MIMEApplication
import os
import sys
import subprocess
import openai
from dotenv import load_dotenv
import tiktoken
from langchain.document_loaders import PyPDFLoader
# import PyPDF2
# from PyPDF2 import PdfMerger
from pypdf import PdfMerger
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain import PromptTemplate, OpenAI, LLMChain
import streamlit as st
import uuid
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from pyhive import hive
import platform
import pandas as pd
from datetime import datetime, date
# import PyMuPDF
# import fitz
import numpy as np
from langchain.schema import Document
from pymongo import MongoClient
import dotenv
dotenv.load_dotenv()

string = os.getenv("mongodb_str")
client = MongoClient(string)  # Use your MongoDB connection string
db = client['pdf_bot_embeddings']  # Database name
# embedded_movies = db['embedded_movies']
# collection = db['embedded_movies_1']



class classEmbdding:
    def __init__(self, chroma_url=[], data_url=[], subfolder=[]):
        self.chroma_url = chroma_url
        self.data_url = data_url
        self.subfolder=subfolder


    def find_Vectordb(self, filename):
        persist_directory = self.chroma_url+"/"+filename
        embeddings = OpenAIEmbeddings(openai_api_key = os.getenv("openai_api_key"))

        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectordb

    def create_vector_db_Indvdl_files(self):
        load_dotenv(find_dotenv())
        # openai.api_type = "azure"
        # openai.api_base = os.getenv('OPENAI_API_BASE', "")
        # openai.api_version = "2022-12-01"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(openai_api_key = os.getenv("openai_api_key"))
        
        file_list_data=os.listdir(self.data_url+"/"+self.subfolder)
        for j in file_list_data:
            collection = db[j]
            # print(j)
            loader_url = self.data_url+"/"+self.subfolder+"/"+j
            loader = PyPDFLoader(loader_url)
            document = loader.load()


            # document1 = [Document(page_content = '', metadata = {'source':'', 'page':0})]
            for i in range(len(document)):
                dictionary = {'page_content':document[i].page_content,
                              'embedding_vector':embeddings.embed_query(document[i].page_content),
                              'metadata':{'source':document[i].metadata['source'], 'page':i+1}
                }
                collection.insert_one(dictionary)

            #     n_tokens = len(document[i].page_content)
            #     n_tokens1=n_tokens-n_tokens%3
            #     source = document[i].metadata['source']
            
            #     text1 = Document(page_content=document[i].page_content[0:int(n_tokens1/3)], metadata={'source':source, 'page':i+1},)
            #     text2 = Document(page_content=document[i].page_content[int(n_tokens1/3):int(2*n_tokens1/3)], metadata={'source':source, 'page':i+1},)
            #     text3 = Document(page_content=document[i].page_content[int(2*n_tokens1/3):int(n_tokens)], metadata={'source':source, 'page':i+1},)
                
            #     document1.append(text1)
            #     document1.append(text2)
            #     document1.append(text3)

            # print(j, len(document1))
            # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            # texts = text_splitter.split_documents(document1)
            # persist_directory=self.chroma_url+"/"+j
            # vector_db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
            # vector_db.persist()

    def combine_pdfs_merger(self,address,file_array,state,year):
        merger = PdfMerger()
        for i in file_array:
            filename=address+"/"+i
            merger.append(filename)

        if os.path.exists(address+"/"+self.subfolder):
            write_adress = address+"/"+self.subfolder+"/"+year+"_"+state+".pdf"
        else:
            os.mkdir(address+"/"+self.subfolder)
            write_adress = address+"/"+self.subfolder+"/"+year+"_"+state+".pdf"
        merger.write(write_adress)
        merger.close()

    def combine_pdfs(self):
        list = ['AP','AR','AS','AZ','DE','FL','FM','GA','IL','IN','KS','KY','MH','MI','MN','MO','ND','NE','NH','NJ','OK','OR','PA','PR','TN','TX','UT','VA','WV','WY','AA','AE','AK','AL','CA','CO','CT','DC','GU','HI','IA','ID','LA','MA','MD','ME','MP','MS','MT','NC','NM','NV','NY','OH','PW','RI','SC','SD','VI','VT','WA','WI']
        file_list_data=os.listdir(self.data_url)
        year_list = range(2021,2030,1)
        for k in year_list:
            for i in list:
                array=np.array([])
                for j in file_list_data:
                    print(i,j,k)
                    if (os.path.isfile(self.data_url+"/"+j) and i == j.split("-")[3][1:3]) and (str(k) == j.split("-")[0]):
                        array=np.append(array,j)

                if len(array)>0:
                    print(i)
                    address = self.data_url
                    file_array=array
                    self.combine_pdfs_merger(address,file_array,i,str(k))

    def download_blobs(blobAddress):
        "Function to download blobs to self.data_url, when the files are updated at the backend"

