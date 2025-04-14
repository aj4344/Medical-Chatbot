# import os
# from huggingface_hub import InferenceClient
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
# from langchain_community.vectorstores import FAISS  # Updated import
# from langchain_core.runnables import RunnableLambda  # ✅ Fix for LLM compatibility

# # Step 1: Setup Hugging Face LLM with InferenceClient
# HF_TOKEN = os.getenv("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# # Initialize Hugging Face Inference Client
# client = InferenceClient(model=HUGGINGFACE_REPO_ID, token=HF_TOKEN)

# def query_llm(prompt: str):
#     """ Function to call the LLM API """
#     response = client.text_generation(
#         prompt, max_new_tokens=512, temperature=0.5
#     )
#     return response

# # ✅ Wrap query_llm inside RunnableLambda to fix LLMChain issue
# query_llm_runnable = RunnableLambda(query_llm)

# # Step 2: Define Custom Prompt Template
# CUSTOM_PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer. Don't provide any additional information.

# Context: {context}
# Question: {question}

# Start answering directly. No small talk please.
# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

# def set_custom_prompt(custom_prompt_template):
#     """ Prompt template for QA retrieval """
#     return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

# # Step 3: Load FAISS Vector Database
# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# # Step 4: Create the QA Chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=query_llm_runnable,  # ✅ Wrapped in RunnableLambda
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k': 3}),
#     return_source_documents=True,  # ✅ Corrected argument name
#     chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# # Step 5: Take User Input and Get Response
# user_query = input("Write Query here: ")
# response = qa_chain.invoke({'query': user_query})

# # Step 6: Print Results
# print("\n✅ RESULT: \n", response['result  \n'])
# print("\n📄 SOURCE DOCUMENTS:", response['source_documents'])


















import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # ✅ Updated Import

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("❌ ERROR: HF_TOKEN not found! Make sure your .env file is loaded properly.")

# Explicitly set the token for Hugging Face API authentication
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# Hugging Face Model
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Step 1: Load LLM from Hugging Face
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )
    return llm

# Step 2: Define Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide any additional information.

Context: {context}
Question: {question}

Start answering directly. No Small talk, please.
Only return the helpful answer below and nothing else.

Helpful answer:
"""

def set_custom_prompt(custom_prompt_template):
    """Prompt template for QA retrieval for each vector store"""
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Step 3: Load FAISS Database
DB_FAISS_PATH = 'vectorstore/db_faiss'
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Run Query
user_query = input("Write Query here: ")
response = qa_chain.invoke({'query': user_query})

# Step 6: Print Result
print("✅ RESULT: \n", response['result'] ,end="\n\n")
print("📄 SOURCE DOCUMENTS: \n", response['source_documents'])














