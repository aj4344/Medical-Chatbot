from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def set_custom_prompt():
    return PromptTemplate(
        template="""
        You are MediBot. Answer the question using ONLY the context from the uploaded PDFs. 
        Do not use any external knowledge, make assumptions, or add extra information beyond the provided context. 
        If the context does not contain the answer, respond EXACTLY with: "I don’t have enough data from the uploaded documents to answer fully" and NOTHING ELSE.
        Context: {context}
        Question: {question}
        Answer:
        """,
        input_variables=['context', 'question']
    )

@app.route('/api/ask', methods=['POST'])
def ask_question():
    logger.info("Entering /api/ask endpoint")
    try:
        question = request.form.get('question')
        pdfs = request.files.getlist('pdfs')
        logger.info(f"Question: {question}, PDFs: {[pdf.filename for pdf in pdfs if pdf]}")

        if not question or not pdfs:
            logger.error("Missing question or PDFs")
            return jsonify({'answer': 'Missing question or PDFs!'}), 400

        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        for pdf in pdfs:
            if not pdf:
                continue
            pdf_path = f"temp_{pdf.filename}"
            pdf.save(pdf_path)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            split_docs = text_splitter.split_documents(docs)
            documents.extend(split_docs)
            os.remove(pdf_path)
            logger.info(f"Processed PDF: {pdf.filename}")

        if not documents:
            logger.error("No valid PDF content extracted")
            return jsonify({'answer': 'No valid PDF content extracted!'}), 400

        logger.info("Building FAISS vector store")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embedding_model)

        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            logger.error("HF_TOKEN not set")
            return jsonify({'answer': 'Hugging Face API token missing! Set HF_TOKEN.'}), 500

        logger.info("Setting up QA chain")
        qa_chain = RetrievalQA.from_chain_type(
            llm=HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                temperature=0.01,  # Lowered for less creativity
                model_kwargs={"token": HF_TOKEN, "max_length": 2048}
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": set_custom_prompt()}
        )

        logger.info(f"Querying with: {question}")
        response = qa_chain.invoke({'query': question})
        answer = response['result'].strip()
        logger.info(f"Answer: {answer}")
        return jsonify({'answer': answer})

    except Exception as e:
        logger.error(f"Error in /api/ask: {str(e)}", exc_info=True)
        return jsonify({'answer': f'Error: {str(e)}'}), 500

@app.route('/test', methods=['GET'])
def test():
    logger.info("Test endpoint hit")
    return jsonify({'message': 'Flask is alive!'})

if __name__ == '__main__':
    try:
        from waitress import serve
        logger.info("Starting Flask with Waitress on http://localhost:5000")
        serve(app, host='localhost', port=5000)
    except ImportError as e:
        logger.error(f"Waitress import failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to start Flask: {str(e)}")
        raise




# from flask import Flask, request, jsonify
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# import os
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# # Global variable to store the processed vector store
# vector_store = None

# def set_custom_prompt():
#     return PromptTemplate(
#         template="""
#         You are MediBot. Answer the question using ONLY the context from the uploaded PDFs. 
#         Do not use any external knowledge, make assumptions, or add extra information beyond the provided context. 
#         If the context does not contain the answer, respond EXACTLY with: "I don’t have enough data from the uploaded documents to answer fully" and NOTHING ELSE.
#         Context: {context}
#         Question: {question}
#         Answer:
#         """,
#         input_variables=['context', 'question']
#     )

# @app.route('/api/process-pdfs', methods=['POST'])
# def process_pdfs():
#     global vector_store
#     logger.info("Entering /api/process-pdfs endpoint")
#     try:
#         pdfs = request.files.getlist('pdfs')
#         logger.info(f"PDFs received: {[pdf.filename for pdf in pdfs if pdf]}")

#         if not pdfs:
#             logger.error("No PDFs uploaded")
#             return jsonify({'message': 'No PDFs uploaded!'}), 400

#         documents = []
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         for pdf in pdfs:
#             if not pdf:
#                 continue
#             pdf_path = f"temp_{pdf.filename}"
#             pdf.save(pdf_path)
#             loader = PyPDFLoader(pdf_path)
#             docs = loader.load()
#             split_docs = text_splitter.split_documents(docs)
#             documents.extend(split_docs)
#             os.remove(pdf_path)
#             logger.info(f"Processed PDF: {pdf.filename}")

#         if not documents:
#             logger.error("No valid PDF content extracted")
#             return jsonify({'message': 'No valid PDF content extracted!'}), 400

#         logger.info("Building FAISS vector store")
#         embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         vector_store = FAISS.from_documents(documents, embedding_model)
#         logger.info("FAISS vector store built successfully")
#         return jsonify({'message': 'PDFs processed successfully'})

#     except Exception as e:
#         logger.error(f"Error in /api/process-pdfs: {str(e)}", exc_info=True)
#         return jsonify({'message': f'Error: {str(e)}'}), 500

# @app.route('/api/ask', methods=['POST'])
# def ask_question():
#     global vector_store
#     logger.info("Entering /api/ask endpoint")
#     try:
#         data = request.get_json()
#         if not data or 'question' not in data:
#             logger.error("No question provided in JSON payload")
#             return jsonify({'answer': 'No question provided!'}), 400
#         question = data['question']
#         logger.info(f"Question received: {question}")

#         if not vector_store:
#             logger.error("No PDFs processed yet")
#             return jsonify({'answer': 'Please process PDFs first!'}), 400

#         HF_TOKEN = os.getenv("HF_TOKEN")
#         if not HF_TOKEN:
#             logger.error("HF_TOKEN not set")
#             return jsonify({'answer': 'Hugging Face API token missing! Set HF_TOKEN.'}), 500

#         logger.info("Setting up QA chain")
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=HuggingFaceEndpoint(
#                 repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#                 temperature=0.01,
#                 model_kwargs={"token": HF_TOKEN, "max_length": 2048}
#             ),
#             chain_type="stuff",
#             retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
#             return_source_documents=True,
#             chain_type_kwargs={"prompt": set_custom_prompt()}
#         )

#         logger.info(f"Querying with: {question}")
#         response = qa_chain.invoke({'query': question})
#         answer = response['result'].strip()
#         logger.info(f"Answer generated: {answer}")
#         return jsonify({'answer': answer})

#     except Exception as e:
#         logger.error(f"Error in /api/ask: {str(e)}", exc_info=True)
#         return jsonify({'answer': f'Error: {str(e)}'}), 500

# @app.route('/test', methods=['GET'])
# def test():
#     logger.info("Test endpoint hit")
#     return jsonify({'message': 'Flask is alive!'})

# if __name__ == '__main__':
#     try:
#         from waitress import serve
#         logger.info("Starting Flask with Waitress on http://localhost:5000")
#         serve(app, host='localhost', port=5000)
#     except ImportError as e:
#         logger.error(f"Waitress import failed: {str(e)}")
#         raise
#     except Exception as e:
#         logger.error(f"Failed to start Flask: {str(e)}")
#         raise