import math
import re
from collections import Counter
from langchain import PromptTemplate, OpenAI, LLMChain
import warnings
warnings.filterwarnings('ignore')
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

class answer_obj:
    def __init__(self,chat_history, PROMPT, question_prompt, llm_model,vector_db):
        self.chat_history = chat_history
        self.PROMPT = PROMPT
        self.question_prompt = question_prompt
        self.llm_model = llm_model
        self.question_generator = LLMChain(llm=llm_model, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
        self.doc_chain = load_qa_with_sources_chain(llm_model, chain_type="stuff", verbose=True, prompt=PROMPT)
        self.chain = ConversationalRetrievalChain(
            retriever=vector_db.as_retriever(search_type='similarity_score_threshold', search_kwargs={'score_threshold': 0.6, "k": 5}),
            question_generator=self.question_generator,
            combine_docs_chain=self.doc_chain,
            verbose = True,
            return_generated_question = True,
            return_source_documents=True
        )

    def find_result(self,query, k):
        l = len(self.chat_history)
        chat_history1 = self.chat_history[max(l-k,0):l]
        result = self.chain({"question": query,"chat_history": chat_history1})
        if result['answer'].find("The query can't be answered")<0:
            self.chat_history.append((query, result['answer']))
        return result, self

    def clear_chat_hist():
        return [("", ""),]

    def get_ans_list(self,nloops,n_chat_vect,query,chat_history):
        result1 = []
        answer_prev = ""
        t=0
        similiarity = []
        answer_list = []
        result2 = []

        for i in range(0,nloops):
            result, chat_history = self.find_result(query, n_chat_vect)
            # print(result)
            result2.append(result['source_documents'])
            if i==0:
                answer_prev = result['answer']
                answer_list.append(answer_prev)
                if 1>0:
                    result1.append(answer_prev)
            else:
                sim = self.get_cosine(self.text_to_vector(answer_prev),self.text_to_vector(result['answer']))
                similiarity.append(sim)
                answer_prev = result['answer']
                answer_list.append(answer_prev)
                if 1>0:
                    result1.append(answer_prev)

        return result1, chat_history,similiarity,result2


    
    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def text_to_vector(text):
        WORD = re.compile(r"\w+")
        words = WORD.findall(text)
        return Counter(words)