import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
import streamlit as st 
import pandas as pd

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


########## Remeber: enable this for DEV
load_dotenv()
key_value = os.getenv("OPENAI_API_KEY")
if key_value is None or key_value == "":
    print("OPENAI_API_KEY is not set in DEV")
    exit(1)
else:
    print("OPENAI_API_KEY is set in DEV")
 
########## Remeber: enable this for PROD
# key_value = os.environ.get("OPENAI_API_KEY") 
# if key_value is None or key_value == "":
#     print("OPENAI_API_KEY is not set in PROD")
#     exit(1)
# else:
#     print("OPENAI_API_KEY is set in PROD")

########## Remove default html markups
st.set_page_config(page_title="openai-pdf-read")
hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;} 
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def main():
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        print(uploaded_file)
         
        with open(os.path.join("", uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getvalue()) # Save uploaded contents to file
            print(f.name)

        
        doc_reader = PdfReader(uploaded_file.name)
        print(doc_reader)

        # read data from the file and put them into a variable called raw_text
        raw_text = ''
        for i, page in enumerate(doc_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        print(len(raw_text))
        print(raw_text[:100])

        # Splitting up the text into smaller chunks for indexing
        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200, #striding over the text
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)
        print(len(texts)) 

        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings) 

 
        # STARTS: user input section
        def get_text():
            input_text = st.text_input("Enter question here:", key="input")
            return input_text

        user_input = get_text()
        if(user_input != ""):
            response = load_result(user_input, docsearch)

        submit = st.button('get answer')  
        #If generate button is clicked
        if submit:
            st.subheader("Answer:")
            st.write(response)

        # ENDS: user input section

def load_result(query, docsearch):
    # query = "how does GPT-4 change social media?"
    docs = docsearch.similarity_search(query)
    len(docs)
    print(docs[0])
    

    chain = load_qa_chain(OpenAI(), chain_type="stuff") # we are going to stuff all the docs in at once
    final_result = chain.run(input_documents=docs, question=query)
    return final_result

    # check the prompt
    # chain.llm_chain.prompt.template

    # query = "who is the author of the book?"
    # query_02 = "has it rained this week?"
    # docs = docsearch.similarity_search(query_02)
    # chain.run(input_documents=docs, question=query)

    # query = "who is the book authored by?"
    # docs = docsearch.similarity_search(query,k=4)
    # chain.run(input_documents=docs, question=query)

if __name__ == "__main__":
    main()

 
