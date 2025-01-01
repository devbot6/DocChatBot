from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import gradio as gr
from langchain_ollama import OllamaLLM
import tempfile
import os
import shutil

class DocumentQA:
    def __init__(self):
        # Initialize embedding model
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Initialize LLM
        self.llm = OllamaLLM(model="mistral")
        self.qa = None
        
        # Define the prompt
        self.prompt = """
        1. Use the following pieces of context to answer the question at the end.
        2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
        3. Keep the answer crisp and limited to 3,4 sentences.

        Context: {context}

        Question: {question}

        Helpful Answer:"""
        
    def setup_qa_chain(self, documents):
        # Split into chunks
        text_splitter = SemanticChunker(self.embedder)
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        vector = FAISS.from_documents(chunks, self.embedder)
        retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Set up QA chain
        QA_CHAIN_PROMPT = PromptTemplate.from_template(self.prompt)
        
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=QA_CHAIN_PROMPT,
            callbacks=None,
            verbose=True
        )
        
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}\nsource:{source}",
        )
        
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            callbacks=None
        )
        
        self.qa = RetrievalQA(
            combine_documents_chain=combine_documents_chain,
            verbose=True,
            retriever=retriever,
            return_source_documents=True
        )

    def process_file(self, file):
        if file is None:
            return "Please upload a PDF file first."
            
        try:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "uploaded.pdf")
            
            # Copy the uploaded file to our temporary file
            shutil.copy2(file.name, temp_path)
            
            # Load the PDF
            loader = PDFPlumberLoader(temp_path)
            documents = loader.load()
            
            # Set up the QA chain with new documents
            self.setup_qa_chain(documents)
            return "Document processed successfully! You can now ask questions."
        except Exception as e:
            return f"Error processing document: {str(e)}"
        finally:
            # Clean up temporary files
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def answer_question(self, message, history):
        if self.qa is None:
            return [[message, "Please upload a document first."]]
        
        answer = self.qa(message)["result"]
        history.append([message, answer])
        return history

# Initialize the DocumentQA class
doc_qa = DocumentQA()

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Document Q&A Chatbot")
    
    with gr.Row():
        file_output = gr.Textbox(label="Upload Status")
        upload_button = gr.UploadButton("Click to Upload a PDF", file_types=[".pdf"])
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about your document", container=False)
    clear = gr.ClearButton([msg, chatbot])
    
    upload_button.upload(
        fn=doc_qa.process_file,
        inputs=[upload_button],
        outputs=[file_output],
    )
    
    msg.submit(
        fn=doc_qa.answer_question,
        inputs=[msg, chatbot],
        outputs=[chatbot],
    )

demo.launch(share=True)