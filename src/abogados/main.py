import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from docx import Document
import pypdf
import io
import csv

# Set page configuration
st.set_page_config(page_title="D'Consulting", page_icon="⚖️")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

def read_pdf(file):
    """Extract text from PDF file"""
    pdf_reader = pypdf.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file):
    """Extract text from DOCX file"""
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_text_file(file):
    """Read text from txt file"""
    return file.read().decode("utf-8")

def load_and_process_document(file):
    """Load and process the document for RAG"""
    # Get file extension
    file_extension = file.name.split(".")[-1].lower()
    
    # Read content based on file type
    try:
        if file_extension == "pdf":
            content = read_pdf(file)
        elif file_extension in ["docx", "doc"]:
            content = read_docx(io.BytesIO(file.read()))
        elif file_extension == "txt":
            content = read_text_file(file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(content)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    return vectorstore

def setup_qa_chain(vectorstore, abogados):
    """Set up the QA chain with custom prompt and recommendation logic"""
    template = """Eres un asistente legal útil. Usa el siguiente contexto para responder la pregunta.
    Si no sabes la respuesta, solo di que no lo sabes. No intentes inventar una respuesta.
    
    Contexto: {context}
    
    Pregunta: {question}
    
    Respuesta: Permíteme ayudarte a entender este asunto legal."""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    
    def qa_chain_with_recommendation(query):
        result = qa_chain({"query": query})
        response = result["result"]
        
        # Check if the query asks for a recommendation
        if "recomienda" in query.lower():
            for abogado in abogados:
                if abogado["especialidad"].lower() in query.lower():
                    response += f"\n\nRecomendación: {abogado['nombre']} es especialista en {abogado['especialidad']}."
                    return {"result": response, "source_documents": result["source_documents"]}
        
        return {"result": response, "source_documents": result["source_documents"]}
    
    return qa_chain_with_recommendation

def load_abogados(file_path):
    """Load abogados data from CSV file"""
    abogados = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            abogados.append(row)
    return abogados

abogados = load_abogados('src/abogados/abogados.csv')

# Sidebar for API key and file upload
with st.sidebar:
    st.title("⚖️ Documentos legales")
    api_key = os.getenv("OPENAI_API_KEY")
    uploaded_file = st.file_uploader(
        "Sube tu documento legal:",
        type=["txt", "pdf", "docx", "doc"],
        help="Formatos soportados: PDF, Word documents (DOC/DOCX), y archivos de texto"
    )

# Main chat interface
st.title("🤖 D'Consulting")

if api_key and uploaded_file:
    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Process document and setup QA chain
    if "vectorstore" not in st.session_state:
        with st.spinner(f"Procesando {uploaded_file.name}..."):
            vectorstore = load_and_process_document(uploaded_file)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.qa_chain = setup_qa_chain(vectorstore, abogados)
                st.success("Documento procesado exitosamente!")
            else:
                st.error("Failed to process document. Please try again.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Escribe tu pregunta legal"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                result = st.session_state.qa_chain(prompt)
                response = result["result"]
                st.markdown(response)
                
                # Display sources
                with st.expander("View Sources"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.write(f"Source {i+1}:")
                        st.write(doc.page_content)
        
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Sube un documento para empezar.")

# Add footer with supported formats
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Formatos soportados
- PDF (*.pdf)
- Word Documents (*.docx, *.doc)
- Text Files (*.txt)
""")