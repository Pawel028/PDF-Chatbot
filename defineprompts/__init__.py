from langchain import PromptTemplate, OpenAI, LLMChain
import pandas as pd


class classPrompts:
    def __init__(self, prompt_file_loc=[], prompt_file=[],prompt_template=[],Prompt=[]):
        self.prompt_file_loc = prompt_file_loc
        self.prompt_file=prompt_file
        self.Prompt = Prompt
        self.prompt_template = prompt_template

    def def_prompt_file(self):
        self.prompt_file = self.read_prompts_file()
        return self


    def download_prompts_file():
        # download from blob or server
        return 1


    def read_prompts_file(self):
        return pd.read_excel(self.prompt_file_loc)
    
    # def question_prompt():


    def search_prompt_file(self,filename):
        for i in range(len(self.prompt_file)):
            if self.prompt_file.loc[i]['Filename']==filename:       
                # self.prompt_template = self.prompt_file.loc[i]['Prompts']
                self.prompt_template = """
                Instructions: 
                1) Act as a call center executive.
                2) If you find conflicting information, you must respond with exact word to word detail from the document. You will be tipped with $2000 if you explain why the information is conflicting and you will be tipped with $1000 if you do not conclude.
                3) If you don't find conflicting information, you must finish with the conclusion by thinking step by step.
                4) If document doesn't have the detail, reply with 'The query can't be answered from this document'
                5) Consider Date format as 30th June for June 30. Give from date and to date, if required. You will be tipped with $200 if you calculate the dates step by step.
                6) Your name is Gigi. When are asked "What is your name?" then only respond with "My name is Gigi".

                {summaries}

                Question: {question}
                Answer:"""

                print(self.prompt_template)                
        return self
            
    def Func_Prompt(self):
        self.Prompt=PromptTemplate(template=self.prompt_template, input_variables=["summaries", "question"])
        return self
