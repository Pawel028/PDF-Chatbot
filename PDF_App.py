import streamlit as st

st.set_page_config(page_title="PDF Application")
st.title("Information")
st.write("This is a Demo chatbot to answer questions from a pdf to show the use of OpenAI through Langchain. You can start by adding pdf to the folder and create its embeddings using the Embeddings class. You can then create the defineprompts class to define the prompts for each fo the different pdf for which you would need to add prompts to the excel file against the row with the filename. Get_answers class is created to answer the question. Use the command streamlit run PDF_App.py to run the application.")
st.write("If you have multiple pdfs you can create different embeddings for each of the files and using the selectbox you can select which embedding to refer to.")
