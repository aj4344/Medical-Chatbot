import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Load LLM
def load_llm(hf_token):
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.1,
        model_kwargs={"token": hf_token, "max_length": 2048}
    )

# Strict prompt
def set_custom_prompt():
    return PromptTemplate(
        template="""
        You are MediBot, a medical AI assistant. Answer the question using ONLY the context from the user-uploaded PDFs. 
        Do not use any external knowledge or make assumptions beyond the provided context. 
        If the context does not contain the answer, respond ONLY with: "I don’t have enough data from the uploaded documents to answer fully. Consult a healthcare professional."
        Include causes, symptoms, treatments, and examples if they are in the context. Keep it clear and concise.

        Context: {context}
        Question: {question}

        Answer:
        """,
        input_variables=['context', 'question']
    )

# Process PDFs
def create_vectorstore_from_pdfs(pdf_files, progress_bar):
    if not pdf_files:
        st.warning("⚠️ Please upload at least one PDF.")
        return None

    try:
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        total_files = len(pdf_files)

        for i, pdf_file in enumerate(pdf_files):
            progress_bar.progress((i + 1) / total_files, f"Processing PDF {i+1}/{total_files}: {pdf_file.name}")
            with open(pdf_file.name, "wb") as f:
                f.write(pdf_file.read())
            loader = PyPDFLoader(pdf_file.name)
            docs = loader.load()
            split_docs = text_splitter.split_documents(docs)
            documents.extend(split_docs)
            os.remove(pdf_file.name)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embedding_model)
        return vectorstore

    except Exception as e:
        st.error(f"⚠️ Error processing PDFs: {str(e)}")
        return None

# Main app
def main():
    st.set_page_config(page_title="MediBot 💊", page_icon="🤖", layout="centered")
    st.title("💊 MediBot - Your Trusted Medical AI")
    st.markdown("👩‍⚕️ **Welcome!** Upload medical PDFs to ask questions based on their content.")

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'pdfs_uploaded' not in st.session_state:
        st.session_state.pdfs_uploaded = False

    st.subheader("📤 Upload Medical PDFs")
    uploaded_pdfs = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    process_button = st.button("Process PDFs", disabled=not uploaded_pdfs)

    # Track if PDFs are uploaded but not processed
    if uploaded_pdfs and not st.session_state.pdfs_uploaded:
        st.session_state.pdfs_uploaded = True

    if process_button and uploaded_pdfs:
        progress_bar = st.progress(0, "Starting PDF processing...")
        st.session_state.vectorstore = create_vectorstore_from_pdfs(uploaded_pdfs, progress_bar)
        st.session_state.pdfs_uploaded = False  # Reset after processing
        progress_bar.empty()
        st.success("✅ PDFs processed successfully!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask your medical question here...")

    if prompt:
        with st.chat_message("user"):
            st.markdown(f"🧑‍💻 **You:** {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})

        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            st.error("🔑 Hugging Face API token not found.")
            st.markdown("ℹ️ **How to fix:** Set 'HF_TOKEN' in System/User Environment Variables.")
            return

        if not uploaded_pdfs:
            st.error("❌ No PDFs uploaded yet. Please upload PDFs to proceed.")
            return
        elif st.session_state.pdfs_uploaded and not st.session_state.vectorstore:
            st.warning("⚠️ Reminder: You’ve uploaded PDFs but haven’t processed them. Please click 'Process PDFs' to continue.")
            return
        elif not st.session_state.vectorstore:
            st.error("❌ PDFs not processed yet. Please upload and click 'Process PDFs' to proceed.")
            return

        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HF_TOKEN),
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt()}
            )

            with st.spinner("🤖 MediBot is generating your answer..."):
                response = qa_chain.invoke({'query': prompt})
            answer = response['result'].strip()
            sources = response.get('source_documents', [])

            formatted_answer = f"🤖 **MediBot:**\n\n{answer.replace('. ', '.\n- ')}"
            with st.chat_message("MediBot"):
                st.markdown(formatted_answer)
            st.session_state.messages.append({"role": "MediBot", "content": answer})

            no_data_message = "I don’t have enough data from the uploaded documents to answer fully. Consult a healthcare professional."
            if sources and answer != no_data_message:
                with st.expander("📚 References from Uploaded PDFs"):
                    for i, doc in enumerate(sources, 1):
                        metadata = doc.metadata
                        source_file = metadata.get('source', 'Unknown PDF')
                        page_num = metadata.get('page', 'Unknown Page')
                        author = metadata.get('author', 'Unknown Author') if 'author' in metadata else 'Unknown Author'
                        st.markdown(
                            f"**{i}.** {doc.page_content}\n\n"
                            f"   - **Source:** {os.path.basename(source_file)}\n"
                            f"   - **Page:** {page_num + 1}\n"
                            f"   - **Author:** {author}"
                        )

        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
            st.markdown("ℹ️ **Support:** Try again or contact the MediBot team.")

    st.markdown("⚠️ **Disclaimer:** MediBot provides information based on uploaded PDFs, not medical advice. Consult a doctor for diagnosis or treatment.")

if __name__ == "__main__":
    main()














# workinggg 
# import os
# import streamlit as st
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Constants
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# # Load LLM
# def load_llm(hf_token):
#     return HuggingFaceEndpoint(
#         repo_id=HUGGINGFACE_REPO_ID,
#         temperature=0.2,  # Lowered to reduce hallucination
#         model_kwargs={"token": hf_token, "max_length": 2048}
#     )

# # Strict prompt to enforce PDF-only answers
# def set_custom_prompt():
#     return PromptTemplate(
#         template="""
#         You are MediBot, a medical AI assistant. Answer the question using ONLY the context from the user-uploaded PDFs. 
#         Do not use any external knowledge or make assumptions beyond the provided context. 
#         If the context does not contain the answer, respond ONLY with: "I don’t have enough data from the uploaded documents to answer fully. Consult a healthcare professional."
#         Include causes, symptoms, treatments, and examples if they are in the context. Keep it clear and concise.

#         Context: {context}
#         Question: {question}

#         Answer:
#         """,
#         input_variables=['context', 'question']
#     )

# # Process PDFs and create FAISS vector store
# def create_vectorstore_from_pdfs(pdf_files, progress_bar):
#     if not pdf_files:
#         st.warning("⚠️ Please upload at least one PDF.")
#         return None

#     try:
#         documents = []
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         total_files = len(pdf_files)

#         for i, pdf_file in enumerate(pdf_files):
#             progress_bar.progress((i + 1) / total_files, f"Processing PDF {i+1}/{total_files}: {pdf_file.name}")
#             with open(pdf_file.name, "wb") as f:
#                 f.write(pdf_file.read())
#             loader = PyPDFLoader(pdf_file.name)
#             docs = loader.load()
#             split_docs = text_splitter.split_documents(docs)
#             documents.extend(split_docs)
#             os.remove(pdf_file.name)

#         embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         vectorstore = FAISS.from_documents(documents, embedding_model)
#         return vectorstore

#     except Exception as e:
#         st.error(f"⚠️ Error processing PDFs: {str(e)}")
#         return None

# # Main app
# def main():
#     st.set_page_config(page_title="MediBot 💊", page_icon="🤖", layout="centered")
#     st.title("💊 MediBot - Your Trusted Medical AI")
#     st.markdown("👩‍⚕️ **Welcome!** Upload medical PDFs to ask questions based on their content.")

#     # Session state
#     if 'vectorstore' not in st.session_state:
#         st.session_state.vectorstore = None

#     # PDF Upload Section
#     st.subheader("📤 Upload Medical PDFs")
#     uploaded_pdfs = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
#     process_button = st.button("Process PDFs", disabled=not uploaded_pdfs)

#     if process_button and uploaded_pdfs:
#         progress_bar = st.progress(0, "Starting PDF processing...")
#         st.session_state.vectorstore = create_vectorstore_from_pdfs(uploaded_pdfs, progress_bar)
#         progress_bar.empty()
#         st.success("✅ PDFs processed successfully!")

#     # Chat history
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # User input
#     prompt = st.chat_input("Ask your medical question here...")

#     if prompt:
#         with st.chat_message("user"):
#             st.markdown(f"🧑‍💻 **You:** {prompt}")
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         HF_TOKEN = os.getenv("HF_TOKEN")
#         if not HF_TOKEN:
#             st.error("🔑 Hugging Face API token not found.")
#             st.markdown("ℹ️ **How to fix:** Set 'HF_TOKEN' in System/User Environment Variables.")
#             return

#         if not st.session_state.vectorstore:
#             st.error("❌ No PDFs uploaded yet. Please upload and process PDFs to proceed.")
#             return

#         try:
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=load_llm(HF_TOKEN),
#                 chain_type="stuff",
#                 retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": set_custom_prompt()}
#             )

#             with st.spinner("🤖 MediBot is generating your answer..."):
#                 response = qa_chain.invoke({'query': prompt})
#             answer = response['result'].strip()
#             sources = response.get('source_documents', [])

#             # Debug: Show context (remove this later if not needed)
#             # st.write("Debug - Context sent to model:", [doc.page_content for doc in sources])

#             formatted_answer = f"🤖 **MediBot:**\n\n{answer.replace('. ', '.\n- ')}"
#             with st.chat_message("MediBot"):
#                 st.markdown(formatted_answer)
#             st.session_state.messages.append({"role": "MediBot", "content": answer})

#             if sources:
#                 with st.expander("📚 References from Uploaded PDFs"):
#                     for i, doc in enumerate(sources, 1):
#                         metadata = doc.metadata
#                         source_file = metadata.get('source', 'Unknown PDF')
#                         page_num = metadata.get('page', 'Unknown Page')
#                         author = metadata.get('author', 'Unknown Author') if 'author' in metadata else 'Unknown Author'
#                         st.markdown(
#                             f"**{i}.** {doc.page_content}\n\n"
#                             f"   - **Source:** {os.path.basename(source_file)}\n"
#                             f"   - **Page:** {page_num + 1}\n"
#                             f"   - **Author:** {author}"
#                         )

#         except Exception as e:
#             st.error(f"⚠️ An error occurred: {str(e)}")
#             st.markdown("ℹ️ **Support:** Try again or contact the MediBot team.")

#     st.markdown("⚠️ **Disclaimer:** MediBot provides information based on uploaded PDFs, not medical advice. Consult a doctor for diagnosis or treatment.")

# if __name__ == "__main__":
#     main()





# Working with pre FAISS
# import os
# import streamlit as st
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Constants
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# FAISS_SAVE_PATH = "vectorstore/db_faiss"  # For pre-saved FAISS

# # Load LLM
# def load_llm(hf_token):
#     return HuggingFaceEndpoint(
#         repo_id=HUGGINGFACE_REPO_ID,
#         temperature=0.7,
#         model_kwargs={"token": hf_token, "max_length": 2048}
#     )

# # Custom prompt
# def set_custom_prompt():
#     return PromptTemplate(
#         template="""
#         You are MediBot, a professional medical AI assistant. Provide a detailed, accurate answer using the context from uploaded PDFs. 
#         Include causes, symptoms, treatments, and examples where applicable. 
#         If the answer isn’t in the PDFs, say: "I don’t have enough data from the uploaded documents to answer fully. Consult a healthcare professional."
#         Keep responses clear, concise, and medically sound—no fluff.

#         Context: {context}
#         Question: {question}

#         Answer:
#         """,
#         input_variables=['context', 'question']
#     )

# # Optimized vectorstore creation
# def create_vectorstore_from_pdfs(pdf_files, progress_bar):
#     if not pdf_files:
#         st.warning("⚠️ Please upload at least one PDF.")
#         return None

#     try:
#         documents = []
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Smaller chunks
#         total_files = len(pdf_files)

#         for i, pdf_file in enumerate(pdf_files):
#             progress_bar.progress((i + 1) / total_files, f"Processing PDF {i+1}/{total_files}: {pdf_file.name}")
#             with open(pdf_file.name, "wb") as f:
#                 f.write(pdf_file.read())
#             loader = PyPDFLoader(pdf_file.name)
#             docs = loader.load()
#             split_docs = text_splitter.split_documents(docs)
#             documents.extend(split_docs)
#             os.remove(pdf_file.name)  # Clean up

#         # Faster embeddings
#         embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         vectorstore = FAISS.from_documents(documents, embedding_model)
#         vectorstore.save_local(FAISS_SAVE_PATH)  # Save for reuse
#         return vectorstore

#     except Exception as e:
#         st.error(f"⚠️ Error processing PDFs: {str(e)}")
#         return None

# # Load pre-saved FAISS if available
# @st.cache_resource
# def load_vectorstore():
#     if os.path.exists(FAISS_SAVE_PATH):
#         try:
#             embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#             return FAISS.load_local(FAISS_SAVE_PATH, embedding_model, allow_dangerous_deserialization=True)
#         except Exception as e:
#             st.error(f"⚠️ Error loading saved FAISS: {str(e)}")
#             return None
#     return None

# # Main app
# def main():
#     st.set_page_config(page_title="MediBot 💊", page_icon="🤖", layout="centered")
#     st.title("💊 MediBot - Your Trusted Medical AI")
#     st.markdown("👩‍⚕️ **Welcome!** Upload medical PDFs or use pre-loaded data to ask questions.")

#     # Session state
#     if 'vectorstore' not in st.session_state:
#         st.session_state.vectorstore = load_vectorstore()
#     if 'pdfs_processed' not in st.session_state:
#         st.session_state.pdfs_processed = bool(st.session_state.vectorstore)

#     # PDF Upload Section
#     st.subheader("📤 Upload Medical PDFs (Optional)")
#     uploaded_pdfs = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
#     process_button = st.button("Process PDFs" if uploaded_pdfs else "Use Pre-Loaded Data", disabled=not uploaded_pdfs and not st.session_state.pdfs_processed)

#     if process_button:
#         if uploaded_pdfs:
#             progress_bar = st.progress(0, "Starting PDF processing...")
#             st.session_state.vectorstore = create_vectorstore_from_pdfs(uploaded_pdfs, progress_bar)
#             st.session_state.pdfs_processed = True
#             progress_bar.empty()
#             st.success("✅ PDFs processed successfully!")
#         elif st.session_state.pdfs_processed:
#             st.info("ℹ️ Using pre-loaded FAISS data.")

#     # Chat history
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # User input
#     prompt = st.chat_input("Ask your medical question here...")

#     if prompt:
#         with st.chat_message("user"):
#             st.markdown(f"🧑‍💻 **You:** {prompt}")
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         HF_TOKEN = os.getenv("HF_TOKEN")
#         if not HF_TOKEN:
#             st.error("🔑 Hugging Face API token not found.")
#             st.markdown("ℹ️ **How to fix:** Set 'HF_TOKEN' in System/User Environment Variables.")
#             return

#         if not st.session_state.vectorstore:
#             st.error("❌ No data available. Upload PDFs or ensure pre-loaded FAISS exists.")
#             return

#         try:
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=load_llm(HF_TOKEN),
#                 chain_type="stuff",
#                 retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": set_custom_prompt()}
#             )

#             with st.spinner("🤖 MediBot is generating your answer..."):
#                 response = qa_chain.invoke({'query': prompt})
#             answer = response['result'].strip()
#             sources = response.get('source_documents', [])

#             formatted_answer = f"🤖 **MediBot:**\n\n{answer.replace('. ', '.\n- ')}"
#             with st.chat_message("MediBot"):
#                 st.markdown(formatted_answer)
#             st.session_state.messages.append({"role": "MediBot", "content": answer})

#             if sources:
#                 with st.expander("📚 References from PDFs"):
#                     for i, doc in enumerate(sources, 1):
#                         st.markdown(f"**{i}.** {doc.page_content}")

#         except Exception as e:
#             st.error(f"⚠️ An error occurred: {str(e)}")
#             st.markdown("ℹ️ **Support:** Try again or contact the MediBot team.")

#     st.markdown("⚠️ **Disclaimer:** MediBot provides information based on uploaded PDFs, not medical advice. Consult a doctor for diagnosis or treatment.")

# if __name__ == "__main__":
#     main()



















# RUNNINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
# import os
# import streamlit as st
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Constants
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# # Load LLM with token
# def load_llm(hf_token):
#     """Load Hugging Face model with higher capacity"""
#     return HuggingFaceEndpoint(
#         repo_id=HUGGINGFACE_REPO_ID,
#         temperature=0.7,
#         model_kwargs={"token": hf_token, "max_length": 2048}  # Increased for detailed answers
#     )

# # Custom prompt for detailed answers
# def set_custom_prompt():
#     """Prompt template for detailed, professional responses"""
#     return PromptTemplate(
#         template="""
#         You are MediBot, a professional medical AI assistant. Provide a detailed, accurate answer using the context from uploaded PDFs. 
#         Include causes, symptoms, treatments, and relevant examples where applicable. 
#         If the answer isn’t in the PDFs, say: "I don’t have enough data from the uploaded documents to answer fully. Consult a healthcare professional."
#         Keep responses clear, concise, and medically sound—no fluff.

#         Context: {context}
#         Question: {question}

#         Answer:
#         """,
#         input_variables=['context', 'question']
#     )

# # Process PDFs and create FAISS vector store
# @st.cache_resource
# def create_vectorstore_from_pdfs(pdf_files):
#     """Create FAISS vector store from uploaded PDFs"""
#     if not pdf_files:
#         st.warning("⚠️ Please upload at least one PDF to proceed.")
#         return None

#     try:
#         # Load and split PDFs
#         documents = []
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         for pdf_file in pdf_files:
#             # Save uploaded file temporarily
#             with open(pdf_file.name, "wb") as f:
#                 f.write(pdf_file.read())
#             loader = PyPDFLoader(pdf_file.name)
#             docs = loader.load()
#             split_docs = text_splitter.split_documents(docs)
#             documents.extend(split_docs)
#             os.remove(pdf_file.name)  # Clean up temp file

#         # Create embeddings and FAISS store
#         embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         vectorstore = FAISS.from_documents(documents, embedding_model)
#         return vectorstore

#     except Exception as e:
#         st.error(f"⚠️ Error processing PDFs: {str(e)}")
#         return None

# # Main app
# def main():
#     st.set_page_config(page_title="MediBot 💊", page_icon="🤖", layout="centered")
#     st.title("💊 MediBot - Your Trusted Medical AI")
#     st.markdown("👩‍⚕️ **Welcome!** Upload medical PDFs and ask questions based on them.")

#     # PDF Upload Section
#     st.subheader("📤 Upload Medical PDFs")
#     uploaded_pdfs = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

#     # Store vectorstore in session state
#     if 'vectorstore' not in st.session_state:
#         st.session_state.vectorstore = None

#     # Process PDFs when uploaded
#     if uploaded_pdfs and st.session_state.vectorstore is None:
#         with st.spinner("🔄 Processing PDFs and building knowledge base..."):
#             st.session_state.vectorstore = create_vectorstore_from_pdfs(uploaded_pdfs)

#     # Chat history
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # User input
#     prompt = st.chat_input("Ask your medical question here...")

#     if prompt:
#         with st.chat_message("user"):
#             st.markdown(f"🧑‍💻 **You:** {prompt}")
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         # Get token
#         HF_TOKEN = os.getenv("HF_TOKEN")
#         if not HF_TOKEN:
#             st.error("🔑 Hugging Face API token not found.")
#             st.markdown("ℹ️ **How to fix:** Set 'HF_TOKEN' in System/User Environment Variables.")
#             return

#         # Check if vectorstore is ready
#         if not st.session_state.vectorstore:
#             st.error("❌ No PDFs uploaded yet. Please upload medical PDFs to proceed.")
#             return

#         try:
#             # Build QA chain
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=load_llm(HF_TOKEN),
#                 chain_type="stuff",
#                 retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": set_custom_prompt()}
#             )

#             # Get response
#             with st.spinner("🤖 MediBot is generating your answer..."):
#                 response = qa_chain.invoke({'query': prompt})
#             answer = response['result'].strip()
#             sources = response.get('source_documents', [])

#             # Format answer with bullets
#             formatted_answer = f"🤖 **MediBot:**\n\n{answer.replace('. ', '.\n- ')}"

#             with st.chat_message("MediBot"):
#                 st.markdown(formatted_answer)
#             st.session_state.messages.append({"role": "MediBot", "content": answer})

#             # Show sources
#             if sources:
#                 with st.expander("📚 References from PDFs"):
#                     for i, doc in enumerate(sources, 1):
#                         st.markdown(f"**{i}.** {doc.page_content}")

#         except Exception as e:
#             st.error(f"⚠️ An error occurred: {str(e)}")
#             st.markdown("ℹ️ **Support:** Try again or contact the MediBot team.")

#     # Disclaimer for commercial use
#     st.markdown("⚠️ **Disclaimer:** MediBot provides information based on uploaded PDFs, not medical advice. Consult a doctor for diagnosis or treatment.")

# if __name__ == "__main__":
#     main()





















# from langchain_huggingface import HuggingFaceEndpoint
# import streamlit as st
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS  # ✅ Updated Import
# from langchain_core.prompts import PromptTemplate
# import os

# DB_FAISS_PATH = "vectorstore/db_faiss"
# @st.cache_resource
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# def set_custom_prompt(custom_prompt_template):
#     """Prompt template for QA retrieval for each vector store"""
#     prompt = PromptTemplate(template=custom_prompt_template,
#                             input_variables=['context', 'question'])
#     return prompt

# def load_llm(huggingface_repo_id,HF_TOKEN):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token": HF_TOKEN, "max_length": 512}
#     )
#     return llm




# def main():
#     st.title("Ask MediBot")

#     if'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     prompt = st.chat_input("pass ur prompt here")


#     if prompt:
#         st.chat_message('user').markdown(prompt)
#         st.session_state.messages.append({'role': 'user', 'content': prompt})


#         CUSTOM_PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
#             If you don't know the answer, just say that you don't know, don't try to make up an answer.
#             Don't provide any additional information.

#             Context: {context}
#             Question: {question}

#             Start answering directly. No Small talk, please.
#             Only return the helpful answer below and nothing else.

#             Helpful answer:
#             """


#         HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
#         HF_TOKEN = os.environ.get("HF_TOKEN")
        
        
#         try:
#             vectorstore = get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store.")

#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID,HF_TOKEN=HF_TOKEN),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#                 )

#             response = qa_chain.invoke({'query': prompt})

#             result = response['result']
#             source_documents = response['source_documents']
            
#             result_to_show = result+"\n\n\nSource Docs: \n" +str(source_documents)
           










#             # response = "Hi I am a MediBot!"
#             st.chat_message('MediBot').markdown(response)
#             st.session_state.messages.append({'role': 'MediBot', 'content': response})

#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")

# #(message['role']).markdown(message['response'])

# if __name__ == "__main__":
#     main()