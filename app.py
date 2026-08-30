import streamlit as st
from google import genai
from pypdf import PdfReader

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="ASH Study Assistant",
    page_icon="logo.png",
    layout="centered"
)

# -----------------------------
# GEMINI CONNECTION
# -----------------------------
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# -----------------------------
# TITLE
# -----------------------------
st.title("🎓 ASH Study Assistant")
st.write("Ask me anything about your studies!")

# -----------------------------
# PDF UPLOAD
# -----------------------------
st.sidebar.header("📄 Study Material")

uploaded_file = st.sidebar.file_uploader(
    "Upload your university notes (PDF)",
    type=["pdf"]
)

pdf_text = ""

if uploaded_file is not None:

    pdf_reader = PdfReader(uploaded_file)

    for page in pdf_reader.pages:
        text = page.extract_text()

        if text:
            pdf_text += text + "\n"

    st.sidebar.success(
        f"✅ PDF loaded! {len(pdf_reader.pages)} pages found."
    )

# -----------------------------
# STUDY MODE
# -----------------------------
mode = st.selectbox(
    "Choose Study Mode",
    [
        "💬 Normal Chat",
        "📚 Explain Topic",
        "📝 Make Notes",
        "❓ Generate MCQs",
        "🎯 Exam Questions"
    ]
)

# -----------------------------
# CHAT HISTORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# USER INPUT
# -----------------------------
prompt = st.chat_input("Ask your study question...")

if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # -----------------------------
    # STUDY MODE INSTRUCTIONS
    # -----------------------------
    instructions = {
        "💬 Normal Chat":
            "Answer the student's question clearly and accurately.",

        "📚 Explain Topic":
            "Explain the topic in extremely simple words. "
            "Use examples and step-by-step explanations.",

        "📝 Make Notes":
            "Convert the student's topic into short, "
            "easy-to-revise study notes with headings and bullet points.",

        "❓ Generate MCQs":
            "Create 10 important multiple-choice questions. "
            "Give four options and clearly identify the correct answer.",

        "🎯 Exam Questions":
            "Create important exam-style questions from this topic. "
            "Include both short and long questions."
    }

    # -----------------------------
    # PDF INSTRUCTIONS
    # -----------------------------
    if pdf_text:

        pdf_instructions = f"""
The student has uploaded university study material.

IMPORTANT:
- Use the uploaded PDF as the main source for your answer.
- Answer the student's question based on the PDF.
- If the answer is not available in the PDF, clearly say:
  "I couldn't find this information in your uploaded PDF."
- Do not invent information and pretend it came from the PDF.

UPLOADED PDF CONTENT:
--------------------
{pdf_text}
--------------------
"""

    else:

        pdf_instructions = """
No PDF has been uploaded.

Answer the student's question using your normal knowledge.
"""

    # -----------------------------
    # SYSTEM PROMPT
    # -----------------------------
    system_prompt = f"""
You are an AI Study Assistant.

Your goal is to help university students learn.

Study mode:
{instructions[mode]}

Rules:
- Use simple English.
- Explain difficult concepts step by step.
- Give examples whenever useful.
- Do not unnecessarily use complicated terminology.
- If the student asks for an exam answer, make it easy to memorize.

{pdf_instructions}
"""

    # -----------------------------
    # AI RESPONSE
    # -----------------------------
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=system_prompt
                + "\n\nStudent question:\n"
                + prompt
            )

            answer = response.text

            st.markdown(answer)

    # -----------------------------
    # SAVE AI RESPONSE
    # -----------------------------
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
