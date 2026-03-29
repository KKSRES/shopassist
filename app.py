"""
app.py - Streamlit UI for RAG Healthcare Chatbot
================================================
A beautiful, interactive web interface for your RAG chatbot.

Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import requests
from langchain_community.vectorstores import FAISS

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Page config
st.set_page_config(
    page_title="Healthcare Q&A Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")  # Will use secrets.toml
OPENAI_API_BASE = st.secrets.get("OPENAI_API_BASE", "https://openai.vocareum.com/v1")

# Model Configuration
VECTOR_STORE_PATH = "./vector_store"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K_CHUNKS = 4


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CLASSES (Same as notebooks)
# ══════════════════════════════════════════════════════════════════════════════

class LegacyOpenAIEmbeddings:
    """Direct HTTP requests to OpenAI API for embeddings."""
    
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        self.api_key = OPENAI_API_KEY
        self.api_base = OPENAI_API_BASE.rstrip('/')
        self.embeddings_url = f"{self.api_base}/embeddings"
    
    def _call_api(self, texts):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {"model": self.model, "input": texts}
        response = requests.post(self.embeddings_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return [item['embedding'] for item in result['data']]
    
    def embed_documents(self, texts):
        embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings.extend(self._call_api(batch))
        return embeddings
    
    def embed_query(self, text):
        return self._call_api([text])[0]


class LegacyChatOpenAI:
    """Direct HTTP requests to OpenAI API for chat completions."""
    
    def __init__(self, model="gpt-4o-mini", temperature=0.0):
        self.model = model
        self.temperature = temperature
        self.api_key = OPENAI_API_KEY
        self.api_base = OPENAI_API_BASE.rstrip('/')
        self.chat_url = f"{self.api_base}/chat/completions"
    
    def __call__(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append(msg)
            else:
                formatted_messages.append({"role": "user", "content": str(msg)})
        
        data = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature
        }
        
        response = requests.post(self.chat_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content']


# ══════════════════════════════════════════════════════════════════════════════
# LOAD VECTOR STORE (Cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_vector_store():
    """Load vector store once and cache it."""
    if not Path(VECTOR_STORE_PATH).exists():
        st.error(f"❌ Vector store not found at {VECTOR_STORE_PATH}")
        st.info("Please run `ingest.ipynb` first to create the vector store.")
        st.stop()
    
    embeddings = LegacyOpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
    return vector_store


# ══════════════════════════════════════════════════════════════════════════════
# RAG QUERY FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def rag_query(vector_store, query):
    """Execute RAG query and return answer with sources."""
    
    # 1. Embed query and search
    embeddings_obj = LegacyOpenAIEmbeddings(model=EMBEDDING_MODEL)
    query_embedding = embeddings_obj.embed_query(query)
    docs = vector_store.similarity_search_by_vector(query_embedding, k=TOP_K_CHUNKS)
    
    # 2. Build context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Build prompt
    prompt = f"""You are a helpful healthcare assistant that answers questions based ONLY on the provided context documents.

STRICT RULES:
1. Answer ONLY using information from the context below
2. If the answer is not in the context, respond with: "I don't have enough information in the provided documents to answer that question."
3. Do NOT use your general knowledge
4. Do NOT make up information
5. Be concise but comprehensive

Context from documents:
{context}

Question: {query}

Answer (based only on the context above):"""
    
    # 4. Call LLM
    llm = LegacyChatOpenAI(model=CHAT_MODEL, temperature=TEMPERATURE)
    messages = [{"role": "user", "content": prompt}]
    answer = llm(messages)
    
    # 5. Extract sources
    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        source_name = Path(source).name if source != 'Unknown' else source
        if source_name not in seen:
            seen.add(source_name)
            sources.append(source_name)
    
    return answer, sources


# ══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .source-badge {
        background-color: #e8eaf6;
        color: #3f51b5;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🏥 Healthcare Q&A")
    st.markdown("---")
    
    st.subheader("📚 About")
    st.write("""
    This chatbot answers questions about:
    - 🩺 **Diabetes**
    - ❤️ **Hypertension** 
    - 🫁 **Asthma**
    
    All answers are grounded in verified medical documents.
    """)
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    
    # Model selection
    chat_model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="gpt-4o-mini is faster and cheaper"
    )
    
    # Temperature slider
    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0 = Factual, 1 = Creative"
    )
    
    # Top-k slider
    top_k = st.slider(
        "Context Chunks",
        min_value=2,
        max_value=8,
        value=4,
        help="Number of document chunks to retrieve"
    )
    
    st.markdown("---")
    st.subheader("💡 Example Questions")
    
    example_questions = [
        "What are the symptoms of diabetes?",
        "What blood pressure is considered high?",
        "What triggers asthma attacks?",
        "How is Type 2 diabetes treated?",
        "What lifestyle changes help with hypertension?"
    ]
    
    for q in example_questions:
        if st.button(q, key=f"example_{q}", use_container_width=True):
            st.session_state.example_query = q
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Built with LangChain, OpenAI & FAISS")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.title("🏥 Healthcare Question & Answer Assistant")
st.markdown("Ask me anything about diabetes, hypertension, or asthma. All answers are based on verified medical documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load vector store
try:
    vector_store = load_vector_store()
except Exception as e:
    st.error(f"Error loading vector store: {str(e)}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            st.markdown("**📚 Sources:**")
            sources_html = " ".join([f'<span class="source-badge">{s}</span>' for s in message["sources"]])
            st.markdown(sources_html, unsafe_allow_html=True)

# Handle example query from sidebar
if "example_query" in st.session_state:
    user_input = st.session_state.example_query
    del st.session_state.example_query
else:
    user_input = st.chat_input("Ask a question about diabetes, hypertension, or asthma...")

# Process user input
if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching medical documents..."):
            try:
                # Get answer and sources
                answer, sources = rag_query(vector_store, user_input)
                
                # Display answer
                st.markdown(answer)
                
                # Check if it's a refusal
                refusal_phrases = [
                    "I don't have enough information",
                    "not in the provided documents"
                ]
                
                is_refusal = any(phrase.lower() in answer.lower() for phrase in refusal_phrases)
                
                # Display sources only if not a refusal
                if not is_refusal and sources:
                    st.markdown("**📚 Sources:**")
                    sources_html = " ".join([f'<span class="source-badge">{s}</span>' for s in sources])
                    st.markdown(sources_html, unsafe_allow_html=True)
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources if not is_refusal else []
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })

# Footer
st.markdown("---")
st.caption("⚠️ This chatbot provides educational information only. Always consult healthcare professionals for medical advice.")
