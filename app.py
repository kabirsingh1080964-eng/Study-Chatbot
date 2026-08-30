import streamlit as st
from google import genai

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# GEMINI CONNECTION
# -----------------------------
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# -----------------------------
# TITLE
# -----------------------------
st.title("🎓 AI Study Assistant")
st.write("Ask me anything about your studies!")

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

    # Instructions for AI
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
"""

    # -----------------------------
    # AI RESPONSE
    # -----------------------------
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=system_prompt + "\n\nStudent question:\n" + prompt
            )

            answer = response.text

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
