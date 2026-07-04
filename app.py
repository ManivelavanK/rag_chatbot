import streamlit as st
import html
import base64
from datetime import datetime
from pathlib import Path
from retriever import HybridRetriever
from database import init_db

init_db()

st.set_page_config(
    page_title="NIT Calicut Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

css = Path("style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

if "msgs" not in st.session_state:
    st.session_state.msgs = []

@st.cache_resource
def load_retriever():
    return HybridRetriever()

retriever = load_retriever()

def esc(v):
    return html.escape(str(v), quote=True)

def ts():
    return datetime.now().strftime("%I:%M %p")

TYPING_HTML = """
<div class="typing-indicator">
  <div class="ai-avatar-sm">🤖</div>
  <div class="typing-bubble">
    <span></span><span></span><span></span>
  </div>
</div>"""

def user_bubble(content, timestamp=""):
    return f"""
<div class="msg-row msg-row-user">
  <div class="msg-col-user">
    <div class="bubble bubble-user">{esc(content)}</div>
    <div class="msg-meta msg-meta-right">{esc(timestamp)}</div>
  </div>
  <div class="avatar avatar-user">You</div>
</div>"""

def ai_bubble(content, timestamp=""):
    return f"""
<div class="msg-row msg-row-assistant">
  <div class="avatar avatar-ai">🤖</div>
  <div class="msg-col-ai">
    <div class="bubble bubble-ai">{content}</div>
    <div class="msg-meta msg-meta-left">{esc(timestamp)}</div>
  </div>
</div>"""

# Logo
with open("lgog/logo.png", "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()
st.markdown(f"""
<div style="text-align:center; padding-top:20px;">
  <img src="data:image/png;base64,{logo_b64}" style="height:80px; object-fit:contain;">
</div>""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align:center; padding: 18px 0 16px;">
  <div style="font-size:1.2rem; font-weight:700; color:#E2E8F0;">NIT Calicut Policy Assistant</div>
  <div style="font-size:0.78rem; color:#94A3B8; margin-top:6px; max-width:560px; margin-left:auto; margin-right:auto; line-height:1.6;">
    This assistant helps <b style="color:#E2E8F0;">NIT Calicut staff &amp; faculty</b> find answers from official policy documents —
    including <b style="color:#E2E8F0;">Leave Rules</b> (EL, CL, HPL, CCL), <b style="color:#E2E8F0;">Recruitment Rules</b>,
    <b style="color:#E2E8F0;">NIT Policies</b>, <b style="color:#E2E8F0;">Ministry of Education</b> directives, and
    <b style="color:#E2E8F0;">Central Civil Services (CCS)</b> rules.
  </div>
  <div style="font-size:0.7rem; color:#475569; margin-top:8px;">Ask a question — relevant policy excerpts will be retrieved instantly.</div>
</div>""", unsafe_allow_html=True)

# Render history
for msg in st.session_state.msgs:
    if msg["role"] == "user":
        st.markdown(user_bubble(msg["content"], msg.get("timestamp", "")), unsafe_allow_html=True)
    else:
        st.markdown(ai_bubble(msg["content"], msg.get("timestamp", "")), unsafe_allow_html=True)

# Chat input
q = st.chat_input("Ask about NIT Calicut...")

if q:
    now = ts()
    st.markdown(user_bubble(q, now), unsafe_allow_html=True)
    st.session_state.msgs.append({"role": "user", "content": q, "timestamp": now})

    typing_ph = st.empty()
    typing_ph.markdown(TYPING_HTML, unsafe_allow_html=True)

    try:
        chunks, _ = retriever.retrieve(q, diagnostics=False)
    except Exception as e:
        chunks = []

    typing_ph.empty()

    GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy"}
    FAREWELLS = {"bye", "goodbye", "see you", "see ya", "thank you", "thanks", "thank you so much", "thanks a lot", "that's all", "thats all", "ok thanks", "okay thanks", "ok thank you", "okay thank you"}
    if q.strip().lower() in GREETINGS:
        answer = (
            "Hello! Welcome to the <b>NIT Calicut Policy Assistant</b>.<br><br>"
            "I can help you find information from official policy documents including "
            "Leave Rules, Recruitment Rules, NIT Policies, Ministry of Education directives, "
            "and Central Civil Services (CCS) rules.<br><br>"
            "Please go ahead and ask your question."
        )
    elif q.strip().lower() in FAREWELLS:
        answer = (
            "Thank you for using the <b>NIT Calicut Policy Assistant</b>.<br><br>"
            "If you have any more questions about policies or rules in the future, feel free to ask. "
            "Have a great day!"
        )
    elif not chunks:
        answer = "No relevant information found in the documents."
    else:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"<b>[{i}] {esc(c.source)}</b> &mdash; <i>{esc(c.section)}</i><br><br>{esc(c.content)}"
            )
        answer = "<br><hr><br>".join(parts)

    now_ai = ts()
    st.session_state.msgs.append({"role": "assistant", "content": answer, "timestamp": now_ai})
    st.markdown(ai_bubble(answer, now_ai), unsafe_allow_html=True)
    st.rerun()
