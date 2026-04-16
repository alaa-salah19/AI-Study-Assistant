import streamlit as st
import os
import requests
import json
import random
import re
import PyPDF2
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
# from langchain_classic.chains import RetrievalQA

# ==========================================
# 1. Config & Page Setup
# ==========================================
st.set_page_config(page_title="AI Study Assistant", layout="wide", page_icon="🤖")

st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e; color: #ffffff
    }
    .chat-message.bot {
        background-color: #f0f2f6; color: #000000
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Helper Classes (Refactored from your code)
# ==========================================

# --- Quiz Generator Class ---
class QuizGeneratorHFAPI:
    def __init__(self, hf_token, model="meta-llama/Llama-3.1-8B-Instruct"):
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        self.model = model

    def clean_pdf_text(self, text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    def chunk_text(self, text, chunk_size=1000): 
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def generate_quiz_for_chunk(self, chunk, num_questions=2):
        prompt = f"""
        Generate {num_questions} multiple choice questions based on the text provided below.
        CRITICAL INSTRUCTION: You must return the output purely as a valid JSON list. Do not include markdown formatting like ```json ... ```. 
        
        The format must be:
        [
          {{
            "question": "The question text here?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "Option A",
            "explanation": "Brief explanation here.",
            "difficulty": "Easy",
            "keywords": ["key1"]
          }}
        ] 
        
        Text: {chunk}
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3, 
            "max_tokens": 1000
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            if response.status_code != 200:
                st.error(f"API Error: {response.text}")
                return []
            
            content = response.json()["choices"][0]["message"]["content"]
            # Clean up potential markdown formatting
            content = re.sub(r"```json", "", content)
            content = re.sub(r"```", "", content).strip()
            return json.loads(content)
        except Exception as e:
            st.warning(f"Error parsing JSON from chunk: {e}")
            return []

    def generate_quiz(self, text):
        clean_text = self.clean_pdf_text(text)
        chunks = self.chunk_text(clean_text)
        all_questions = []
        
        # Progress bar
        my_bar = st.progress(0)
        total_chunks = min(len(chunks), 5) # Limit to 5 chunks to save time/tokens for demo
        
        for i in range(total_chunks):
            quiz_part = self.generate_quiz_for_chunk(chunks[i])
            all_questions.extend(quiz_part)
            my_bar.progress((i + 1) / total_chunks)
            
        random.shuffle(all_questions)
        for idx, q in enumerate(all_questions, 1):
            q["id"] = idx
        return all_questions

# --- RAG Chatbot Class ---
class RAGChatbot:
    def __init__(self, hf_token, repo_id="meta-llama/Llama-3.1-8B-Instruct"):
        self.hf_token = hf_token
        self.repo_id = repo_id
        self.vector_db = None

    def create_vector_db(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, length_function=len
        )
        docs = [Document(page_content=text)]
        chunks = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = FAISS.from_documents(chunks, embeddings)
        return True

    def ask(self, query):
        if not self.vector_db:
            return "Please process the document first."

        # Build context from vector DB and call Hugging Face chat completions endpoint directly
        retriever = self.vector_db.as_retriever(search_kwargs={'k': 3})
        # retrieve relevant docs
        try:
            docs = retriever.get_relevant_documents(query)
        except Exception:
            # fallback for different retriever implementations
            try:
                docs = retriever.retrieve(query)
            except Exception:
                docs = []

        context = "\n\n".join([getattr(d, 'page_content', str(d)) for d in docs])

        # Prepare messages: provide context and the user question
        system_msg = "You are a helpful assistant. Use the provided context to answer the question concisely. If the answer is not present in the context, say you don't know."
        user_msg = f"Context:\n{context}\n\nQuestion:\n{query}"

        api_url = "https://router.huggingface.co/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
        payload = {
            "model": self.repo_id,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                # If model is not a chat model, fallback to text-generation/inference endpoint
                try:
                    err = resp.json()
                    err_msg = json.dumps(err)
                except Exception:
                    err_msg = resp.text

                if 'not a chat model' in err_msg or 'model_not_supported' in err_msg:
                    # Fallback: call the Hugging Face Router models inference endpoint
                    inf_url = f"https://router.huggingface.co/v1/models/{self.repo_id}"
                    inf_headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
                    inf_payload = {
                        "inputs": user_msg,
                        "parameters": {"max_new_tokens": 512, "temperature": 0.2}
                    }
                    try:
                        inf_resp = requests.post(inf_url, headers=inf_headers, json=inf_payload, timeout=120)
                        if inf_resp.status_code != 200:
                            return f"Inference API Error: {inf_resp.status_code} - {inf_resp.text}"
                        # Parse various response shapes
                        data = inf_resp.json()
                        if isinstance(data, dict) and data.get('error'):
                            return f"Inference API Error: {data.get('error')}"
                        if isinstance(data, list) and len(data) > 0:
                            first = data[0]
                            # huggingface inference can return [{'generated_text': '...'}]
                            text = first.get('generated_text') if isinstance(first, dict) else None
                            if text:
                                return text.strip()
                        if isinstance(data, dict) and 'generated_text' in data:
                            return data['generated_text'].strip()
                        # fallback to raw text
                        return str(data)
                    except Exception as e2:
                        return f"Fallback inference error: {e2}"
                return f"API Error: {resp.status_code} - {err_msg}"

            # Safely extract assistant content
            choices = resp.json().get("choices") or []
            if not choices:
                return "No response choices from API."
            message = choices[0].get("message") or {}
            content = message.get("content")
            if content is None:
                return "The model returned no content."
            # clean markdown blocks
            content = re.sub(r"```.*?```", "", content, flags=re.S).strip()
            return content
        except Exception as e:
            return f"Error calling HF chat API: {e}"
# --- Notebook reader & Summarizer ---
def read_ipynb(path="TEXT.ipynb"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        texts = []
        for cell in cells:
            if cell.get("cell_type") in ("markdown", "code"):
                source = cell.get("source", "")
                if isinstance(source, list):
                    texts.append("".join(source))
                else:
                    texts.append(source)
        return "\n".join(texts)
    except Exception:
        return ""


def summarize_with_hf(text, hf_token, model="meta-llama/Llama-3.1-8B-Instruct", max_tokens=500):
    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    prompt = (
        "Please provide a concise summary of the following content. Keep it short and clear:\n\n" + text
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": int(max_tokens),
    }
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return f"API Error: {resp.text}"
        content = resp.json()["choices"][0]["message"]["content"]
        content = re.sub(r"```.*?```", "", content, flags=re.S).strip()
        return content
    except Exception as e:
        return f"Error: {e}"


def chunk_text_for_summary(text, max_chars=12000):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
            if len(p) > max_chars:
                # fallback: split big paragraph into fixed-size pieces
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks


def summarize_long_text(text, hf_token, model="meta-llama/Llama-3.1-8B-Instruct", max_tokens_per_call=500):
    # Split large text into safe-sized chunks, summarize each, then combine summaries
    chunks = chunk_text_for_summary(text)
    if not chunks:
        return ""
    if len(chunks) == 1:
        return summarize_with_hf(chunks[0], hf_token, model=model, max_tokens=max_tokens_per_call)

    summaries = []
    for i, c in enumerate(chunks, 1):
        try:
            s = summarize_with_hf(c, hf_token, model=model, max_tokens=max_tokens_per_call)
        except Exception as e:
            s = f"Error summarizing chunk {i}: {e}"
        summaries.append(s)

    combined = "\n\n".join(summaries)
    # Final pass to produce a concise unified summary
    final = summarize_with_hf(combined, hf_token, model=model, max_tokens=max_tokens_per_call)
    return final

# ==========================================
# 3. Main Application UI
# ==========================================

def main():
    st.sidebar.title("⚙️ Settings")

    # Securely input API Token (prefill from HF_TOKEN env var if present)
    hf_token_env = os.getenv("HF_TOKEN")
    hf_token = st.sidebar.text_input(
        "Hugging Face Token",
        value=hf_token_env or "",
        type="password",
        help="Enter your HF Write Token here (or set HF_TOKEN env var)"
    )

    if not hf_token:
        st.warning("Please enter your Hugging Face Token in the sidebar to start.")
        return

    st.sidebar.divider()

    # File Upload
    uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])

    text_content = ""

    if uploaded_file is not None:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text_content += page.extract_text() or ""
            st.sidebar.success(f"PDF Loaded: {len(text_content)} characters")
        except Exception as e:
            st.sidebar.error(f"Error reading PDF: {e}")

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Generate Quiz", "🤖 Chat with PDF", "🗒️ Summarize"])

    # --- TAB 1: Quiz ---
    with tab1:
        st.header("Exam Simulator")

        if uploaded_file and st.button("Generate New Quiz"):
            with st.spinner("Analyzing document and crafting questions..."):
                quiz_gen = QuizGeneratorHFAPI(hf_token)
                st.session_state['quiz'] = quiz_gen.generate_quiz(text_content)

        if 'quiz' in st.session_state and st.session_state['quiz']:
            for q in st.session_state['quiz']:
                with st.expander(f"Question {q['id']}: {q.get('difficulty', 'General')}", expanded=True):
                    st.markdown(f"**{q['question']}**")

                    # Create unique key for radio button
                    user_answer = st.radio("Choose:", q['options'], key=f"q_{q['id']}")

                    if st.button(f"Check Answer {q['id']}", key=f"btn_{q['id']}"):
                        if user_answer == q['answer']:
                            st.success("Correct! ✅")
                        else:
                            st.error(f"Wrong ❌. The correct answer is: {q['answer']}")

                        if 'explanation' in q:
                            st.info(f"💡 Explanation: {q['explanation']}")
        elif not uploaded_file:
            st.info("Upload a PDF to start generating questions.")

    # --- TAB 2: Chat (RAG) ---
    with tab2:
        st.header("Document Assistant")

        # Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize RAG System
        if "rag_system" not in st.session_state and text_content:
            with st.spinner("Indexing document for search..."):
                rag = RAGChatbot(hf_token)
                rag.create_vector_db(text_content)
                st.session_state["rag_system"] = rag

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # Ensure content is a string to avoid showing 'None'
                content = message.get("content")
                st.markdown(str(content or "")) 

        # Chat Input
        if prompt := st.chat_input("Ask something about your file..."):
            if not text_content:
                st.error("Please upload a file first.")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate answer
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        if "rag_system" in st.session_state:
                            response = st.session_state["rag_system"].ask(prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            st.error("System not initialized. Try re-uploading the file.")

    # --- TAB 3: Summarize ---
    with tab3:
        st.header("Summarizer")

        # nb_text = read_ipynb("TEXT.ipynb")

        # Show available sources
        st.write("Choose source to summarize:")
        options = []
        if text_content:
            options.append("Uploaded PDF")
        # if nb_text:
        #     options.append("TEXT.ipynb")
        if not options:
            st.error("No source found — upload a PDF or ensure TEXT.ipynb exists in workspace.")
        else:
            source = st.radio("Source", options)
            max_len = st.slider("Max summary length (tokens approx)", min_value=50, max_value=800, value=300)

            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    if source == "Uploaded PDF":
                        summary = summarize_long_text(text_content, hf_token, max_tokens_per_call=max_len)
                    else:
                        summary = summarize_long_text(text_content, hf_token, max_tokens_per_call=max_len)

                    st.subheader("Summary")
                    st.write(summary)

if __name__ == "__main__":
    main()