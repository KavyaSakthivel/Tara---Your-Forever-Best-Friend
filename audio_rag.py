import streamlit as st
from time import sleep
from streamlit_mic_recorder import mic_recorder
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings
import chromadb
from gtts import gTTS
import random

# Helper functions
def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return os.path.basename(max(paths, key=os.path.getctime))

def text_to_audio(text):
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_file = "temp_audio.mp3"
    tts.save(mp3_file)
    return mp3_file

def save_uploaded_file(uploaded_file, directory):
    try:
        with open(os.path.join(directory, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        return False

# Initialize environment and settings
os.environ['GROQ_API_KEY'] = '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'
chroma_setting = Settings(anonymized_telemetry=False)
upload_dir = "uploaded_files"
os.makedirs(upload_dir, exist_ok=True)

# Setup LLM and embeddings
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.1,
    max_tokens=1000,
)

model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

def text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
        length_function=len,
    )

# Enhanced answer_question function with empathetic responses
def answer_question(question, vectorstore):
    # Empathetic response components
    empathetic_prefixes = {
        "burnout": "I hear how exhausting this could be for you. ",
        "overwhelm": "It's completely natural to feel overwhelmed. ",
        "stress": "I understand this is a challenging time. ",
        "tired": "It's understandable that you're feeling this way. ",
        "anxiety": "Thank you for sharing these feelings with me. ",
        "worry": "Your concerns are valid. ",
        "guilt": "Please be gentle with yourself. ",
        "help": "I'm here to support you. ",
        "lonely": "It's okay to feel this way, and you're not alone. ",
        "scared": "It takes courage to share these feelings. ",
        "depression": "I'm here to listen without judgment. ",
        "confused": "It's perfectly normal to feel uncertain. "
    }
    
    empathetic_closings = [
        "Remember, taking care of yourself is just as important as taking care of others.",
        "You're doing the best you can, and that's enough.",
        "It's okay to take things one day at a time.",
        "Your feelings matter, and it's important to acknowledge them.",
        "You don't have to carry this burden alone.",
        "Taking small steps toward self-care can make a big difference.",
        "Be patient with yourself as you navigate this journey.",
        "Remember to celebrate even the smallest victories.",
        "You're stronger than you might feel right now.",
        "It's okay to reach out for support when you need it."
    ]
    
    # Get base response from QA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa.invoke({"query": question})
    response = result['result']
    
    # Remove any references to source material or context
    response = response.replace("According to the provided context, ", "")
    response = response.replace("Based on the available information, ", "")
    response = response.replace("The context suggests that ", "")
    response = response.replace("not explicitly mentioned", "")
    response = response.replace("However, it does suggest that ", "")
    response = response.replace("According to the text,", "")
    response = response.replace(" mentioned in the text,", "")
    
    # Add appropriate prefix based on question content
    prefix = ""
    for keyword, empathetic_prefix in empathetic_prefixes.items():
        if keyword.lower() in question.lower():
            prefix = empathetic_prefix
            break
    if not prefix:  # Default prefix if no keywords match
        prefix = "Thank you for sharing with me. "
    
    # Add appropriate closing
    closing = f"\n\n{random.choice(empathetic_closings)}"
    
    # Combine with natural transitions
    full_response = f"{prefix}{response}{closing}"
    
    # Make response more conversational
    replacements = {
        "One should": "You might want to",
        "It is important": "I think it's helpful",
        "must": "could",
        "should": "might consider",
        "need to": "could try to",
        "have to": "might want to",
        "required to": "encouraged to",
        "mandatory": "beneficial",
        "essential": "valuable"
    }
    
    for old, new in replacements.items():
        full_response = full_response.replace(old, new)
    
    return full_response
def transcribe_audio(filename):
    groq_client = Groq()
    with open(filename, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="distil-whisper-large-v3-en",
            response_format="json",
            language="en",
            temperature=0.0
        )
    return transcription.text

# Streamlit UI
st.set_page_config(
    page_title="Mindful Companion",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (your existing CSS remains the same)
st.markdown("""
    <style>
    .stApp {
        background-color: #fdf6f6;
        color: #000000;
    }
    .css-1d391kg {
        padding: 1.5rem;
        color: #000000;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 1rem;
        color: #000000;
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .user-message {
        background-color: #fff5f7;
        color: #000000;
        margin-left: 2rem;
        border-left: 4px solid #feb2b2;
    }
    .assistant-message {
        background-color: #f8f9ff;
        color: #000000;
        margin-right: 2rem;
        border-left: 4px solid #b2b7ff;
    }
    .message-content {
        margin-top: 0.5rem;
        line-height: 1.6;
        font-size: 1.1rem;
        color: #000000;
    }
    .transcript-box {
        background-color: #f0f4ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4f46e5;
        color: #000000;
    }
    .response-box {
        background-color: #f0fff4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #059669;
        color: #000000;
    }
    .upload-section {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .stButton button {
        background-color: #feb2b2;
        border-radius: 2rem;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton button:hover {
        background-color: #fc8181;
    }
    .sidebar .stButton button {
        width: 100%;
    }
    h1, h2, h3 {
        color: #4a5568;
    }
    .welcome-message {
        padding: 2rem;
        color: #000000;
        background-color: white;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with resources
with st.sidebar:
    st.title("🌸 Resources")
    st.markdown("""
    ### Emergency Contacts
    - **National Crisis Line**: 988
    - **Women's Crisis Hotline**: 1-800-799-SAFE
    
    ### Self-Care Tips
    1. Take deep breaths
    2. Stay hydrated
    3. Practice mindfulness
    4. Connect with loved ones
    5. Get adequate rest
    """)
    
    # Hidden file uploader for admin
    with st.expander("Admin Section", expanded=False):
        uploaded_file = st.file_uploader("Upload Knowledge Base", type="pdf", key="pdf_uploader")
        if uploaded_file:
            with st.spinner("Processing document..."):
                if save_uploaded_file(uploaded_file, upload_dir):
                    st.success("Knowledge base updated successfully!")
                    
                    file_name = uploaded_file.name
                    loader = PyPDFLoader(f"{upload_dir}/{file_name}")
                    pages = loader.load_and_split(text_splitter())
                    persist_directory = f"chromanew_{file_name.split('.')[0]}"
                    
                    if not os.path.exists(persist_directory):
                        client = chromadb.PersistentClient(path=persist_directory, settings=chroma_setting)
                        vectorstore = Chroma(
                            embedding_function=embeddings,
                            client=client,
                            persist_directory=persist_directory,
                            collection_name=file_name.split(".")[0],
                            client_settings=chroma_setting
                        )
                        
                        MAX_BATCH_SIZE = 100
                        for i in range(0, len(pages), MAX_BATCH_SIZE):
                            i_end = min(len(pages), i + MAX_BATCH_SIZE)
                            batch = pages[i:i_end]
                            vectorstore.add_documents(batch)

# Main content
st.title("🌸 Mindful Companion")

# Welcome message
st.markdown("""
    <div class="welcome-message">
        <h3>Welcome to Your Safe Space 💝</h3>
        <p>I'm here to listen and support you on your journey. Feel free to share whatever's on your mind - your thoughts, feelings, or concerns. 
        You can either speak to me or type your messages. Everything shared here is confidential and judgment-free.</p>
        <p>Remember: You're not alone, and it's okay to not be okay.</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if "current_transcription" not in st.session_state:
    st.session_state.current_transcription = None
if "current_response" not in st.session_state:
    st.session_state.current_response = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for i, chat in enumerate(st.session_state.chat_history):
    # User message
    st.markdown(f"""
        <div class="chat-message user-message">
            <div><strong>You:</strong></div>
            <div class="message-content">{chat["question"]}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Assistant message
    st.markdown(f"""
        <div class="chat-message assistant-message">
            <div><strong>Mindful Companion:</strong></div>
            <div class="message-content">{chat["response"]}</div>
        </div>
    """, unsafe_allow_html=True)

# Voice input section
st.markdown("### 💝 Share Your Thoughts")
st.markdown("Take your time. I'm here to listen when you're ready.")

audio = mic_recorder(
    start_prompt="Start sharing",
    stop_prompt="Finish sharing",
    just_once=False,
    key='recorder'
)

# Current interaction display
if audio:
    with st.spinner("Processing your message with care..."):
        with open("recorded_audio.wav", "wb") as f:
            f.write(audio['bytes'])
        
        # Get transcription
        transcription = transcribe_audio("recorded_audio.wav")
        st.session_state.current_transcription = transcription
        
        # Display transcription
        st.markdown("### 📝 What You Said:")
        st.markdown(f"""
            <div class="transcript-box">
                {transcription}
            </div>
        """, unsafe_allow_html=True)
        
        # Get and process response
        file_name = newest(upload_dir)
        persist_directory = f"chromanew_{file_name.split('.')[0]}"
        
        client = chromadb.PersistentClient(path=persist_directory, settings=chroma_setting)
        vectorstore = Chroma(
            embedding_function=embeddings,
            client=client,
            persist_directory=persist_directory,
            collection_name=file_name.split(".")[0],
            client_settings=chroma_setting
        )
        
        response = answer_question(transcription, vectorstore)
        st.session_state.current_response = response
        
        # Display response text
        st.markdown("### 💌 My Response:")
        st.markdown(f"""
            <div class="response-box">
                {response}
            </div>
        """, unsafe_allow_html=True)
        
        # Create and display audio response
        audio_response = text_to_audio(response)
        st.markdown("### 🎵 Listen to Response")
        st.audio(audio_response, format='audio/mp3')
        
        # Add to chat history
        st.session_state.chat_history.append({
            "question": transcription,
            "response": response
        })

# Footer with supportive message
st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #718096;">
        Remember to be gentle with yourself. Every small step matters. 🌸
    </div>
""", unsafe_allow_html=True)