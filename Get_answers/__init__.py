import os
import math
import re
from collections import Counter
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain.embeddings import OpenAIEmbeddings
from langchain import LLMChain
import openai
import pandas as pd
import numpy as np
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore")

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()

class answer_obj:
    def __init__(self, chat_history, prompt, question_prompt, llm_model, filename):
        self.chat_history = chat_history
        self.prompt = prompt
        self.question_prompt = question_prompt
        self.llm_model = llm_model

        # MongoDB setup
        self.client = MongoClient(os.getenv("mongodb_str"))
        db = self.client["pdf_bot_embeddings"]
        self.collection = db[filename]
        print(filename)

        # Embedding + vector store
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("openai_api_key"))
        self.vectorstore = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=embeddings,
            index_name=filename,
            text_key="text",
        )

        # Chain setup
        # self.question_generator = LLMChain(llm=llm_model, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
        # self.doc_chain = load_qa_with_sources_chain(
        #     llm_model, chain_type="stuff", verbose=True, prompt=self.prompt
        # )

        # self.chain = ConversationalRetrievalChain(
        #     retriever=self.vectorstore.as_retriever(),
        #     question_generator=self.question_generator,
        #     combine_docs_chain=self.doc_chain,
        #     verbose=True,
        #     return_generated_question=True,
        #     return_source_documents=True,
        # )







    def find_closest_vect(self,query):
        embed_query = embeddings.embed_query(query)
        ls = pd.DataFrame(list(self.collection.find()))
        arr = [np.dot(embed_query,ls.iloc[i]['embedding_vector']) for i in range(len(ls))]
        arr1 = arr.copy()
        arr1.sort(reverse=True)
        ls1=[arr.index(arr1[i]) for i  in range(6)]
        docs = [{'page_content':ls.iloc[i]['page_content'],'metadata':ls.iloc[i]['metadata']} for i in ls1]
        text = "\n".join([ls.iloc[i]['page_content'] for i in ls1])
        return text, docs
    
    @staticmethod
    def run_llm(query,text):
        template = """
        You are an agent to answer the user query based on the retrieved text. if the information is not in the retrieved text, simply
        reply: "I can't find the answer in the text"
        User Query: {query}

        Retrieved Text: {text}
            """
        prompt = PromptTemplate(template=template, input_variables=["query","text"])
        business_analyst_prompt = prompt.format(query=query,text=text)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": business_analyst_prompt}
        ]
        from openai import OpenAI
        client = OpenAI()
        # Generate response
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message.content

    def find_answers(self,query1):
        text1,docs1 = self.find_closest_vect(query1)
        print(query1)
        ans = self.run_llm(query=query1,text=text1)
        return ans,"",0,docs1
    











    def find_answer(self,query):
        self.chain({"question": query, "chat_history": ""})


    def find_result(self, query, k):
        chat_history_subset = self.chat_history[-k:] if k > 0 else []
        result = self.chain({"question": query, "chat_history": chat_history_subset})
        # print(result.get("source_documents", []))

        if "The query can't be answered" not in result["answer"]:
            self.chat_history.append((query, result["answer"]))

        return result, self

    @staticmethod
    def clear_chat_history():
        return [("", "")]

    def get_ans_list(self, nloops, n_chat_vect, query):
        results = []
        similarities = []
        source_docs = []
        prev_answer = ""

        for i in range(nloops):
            result, _ = self.find_result(query, n_chat_vect)
            # print(result)
            current_answer = result["answer"]
            results.append(current_answer)
            source_docs.append(result["source_documents"])

            if i > 0:
                sim = self.get_cosine(
                    self.text_to_vector(prev_answer), self.text_to_vector(current_answer)
                )
                similarities.append(sim)

            prev_answer = current_answer

        return results, self, similarities, source_docs

    @staticmethod
    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)
        denominator = math.sqrt(sum(v ** 2 for v in vec1.values())) * math.sqrt(sum(v ** 2 for v in vec2.values()))
        return float(numerator) / denominator if denominator else 0.0

    @staticmethod
    def text_to_vector(text):
        words = re.findall(r"\w+", text)
        return Counter(words)
