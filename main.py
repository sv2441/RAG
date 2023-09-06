### Lib 
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import streamlit as st
import os
import base64
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

loader = CSVLoader(file_path='extracted_data2.csv',encoding="utf8")
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])


chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")



def generator(df):
    
    result_df = pd.DataFrame(columns=['OP ref', 'OP', 'Similar'])
    for index ,row in df.iterrows():
        query =f""" Provide the "OP Name" and its "OP ref" that has same meaning as or has higher similarity ranking to "{row['OP']}". 
            
            """
        response = chain({"question": query})
        result_df = result_df.append({'OP ref': row['OP ref'], 'OP': row['OP'], 'Similar': response['result']}, ignore_index=True)
    return result_df
    

 
### main
st.title("👨‍💻 Prodago RAG")



# Streamlit UI
st.write("OP Similarity Search")

uploaded_file = st.file_uploader("Upload a file", type=["xlsx", "xls","csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,encoding="latin-1")
    st.write(df)

    if st.button("Generate Similarities", key="generate"):
        # data=pd.read_csv("fdata.csv")
        result_df = generator(df)
        st.write(result_df)

        # Download button for the generated result
        st.markdown('### Download Result')
        result_csv = result_df.to_csv(index=False)
        b64 = base64.b64encode(result_csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="result.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)