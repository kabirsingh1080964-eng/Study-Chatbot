import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from huggingface_hub import InferenceClient

from voice_agent import voice_agent


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="ASH Study Assistant",
    page_icon="logo.png",
    layout="centered"
)


# ============================================================
# API CLIENTS
# ============================================================

@st.cache_resource
def get_gemini_client():

    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )


@st.cache_resource
def get_image_client():

    return InferenceClient(
        api_key=st.secrets["HF_TOKEN"]
    )


client = get_gemini_client()
image_client = get_image_client()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "💬 Normal Chat"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main app width */
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    /* Hide unnecessary Streamlit footer */
    footer {
        visibility: hidden;
    }

    /* Voice button area */
    .voice-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.image(
    "logo.png",
    width=130
)

st.title(
    "ASH Study Assistant"
)

st.caption(
    "Your AI-powered university study assistant."
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# VOICE LIVE API TOKEN
# ============================================================

@st.cache_data(ttl=50)
def create_voice_token():

    token = client.auth_tokens.create(
        config={
            "uses": 1,

            "expire_time": (
                __import__("datetime")
                .datetime.now(
                    __import__("datetime").timezone.utc
                )
                + __import__("datetime").timedelta(
                    minutes=30
                )
            ),

            "new_session_expire_time": (
                __import__("datetime")
                .datetime.now(
                    __import__("datetime").timezone.utc
                )
                + __import__("datetime").timedelta(
                    minutes=1
                )
            ),

            "live_connect_constraints": {
                "model": (
                    "gemini-3.1-flash-live-preview"
                ),

                "config": {
                    "response_modalities": [
                        "AUDIO"
                    ]
                }
            }
        }
    )

    return token.name


try:

    voice_token = create_voice_token()

except Exception as e:

    voice_token = None


# ============================================================
# BOTTOM CHAT BAR
#
# [ Ask anything... ] [ 🎤 ] [ ➕ ]
#
# ============================================================

chat_col, voice_col, plus_col = st.columns(
    [8, 0.8, 0.8],
    vertical_alignment="bottom"
)


# ============================================================
# TEXT CHAT
# ============================================================

with chat_col:

    typed_prompt = st.chat_input(
        "Ask anything about your studies..."
    )


# ============================================================
# LIVE VOICE BUTTON
# ============================================================

with voice_col:

    if voice_token:

        voice_agent(
            token=voice_token,

            system_instruction="""
You are ASH Study Assistant.

You are a friendly university study assistant.

Speak naturally and conversationally.

Use simple English.

Help the student understand university subjects.

Keep answers reasonably short unless
the student asks for a detailed explanation.

If the student interrupts you,
stop speaking and listen to the student.

Do not repeat the student's question.
""",

            key="ash_live_voice"
        )

    else:

        st.error(
            "🎤"
        )


# ============================================================
# PLUS MENU
# ============================================================

with plus_col:

    with st.popover(
        "➕",
        use_container_width=True
    ):

        st.subheader(
            "Study Tools"
        )


        # ====================================================
        # STUDY MODES
        # ====================================================

        study_modes = [

            "💬 Normal Chat",

            "📚 Explain Topic",

            "📝 Make Notes",

            "❓ Generate MCQs",

            "🎯 Exam Questions",

            "🎨 Generate Image"
        ]


        selected_mode = st.radio(
            "Choose Study Mode",

            study_modes,

            index=study_modes.index(
                st.session_state.selected_mode
            ),

            label_visibility="collapsed"
        )


        st.session_state.selected_mode = (
            selected_mode
        )


        st.divider()


        # ====================================================
        # PDF UPLOAD
        # ====================================================

        st.write(
            "📄 Upload Study Material"
        )


        uploaded_file = st.file_uploader(
            "Choose PDF",

            type=["pdf"],

            label_visibility="collapsed"
        )


        if uploaded_file is not None:

            try:

                reader = PdfReader(
                    uploaded_file
                )

                extracted_text = ""


                for page in reader.pages:

                    text = page.extract_text()

                    if text:

                        extracted_text += (
                            text + "\n"
                        )


                st.session_state.pdf_text = (
                    extracted_text
                )

                st.session_state.pdf_name = (
                    uploaded_file.name
                )


                st.success(
                    f"✅ {uploaded_file.name} loaded"
                )


                st.caption(
                    f"{len(reader.pages)} pages"
                )


            except Exception as e:

                st.error(
                    "❌ Could not read PDF."
                )

                st.code(
                    str(e)
                )


        elif st.session_state.pdf_name:

            st.success(
                f"📄 "
                f"{st.session_state.pdf_name} loaded"
            )


        # ====================================================
        # REMOVE PDF
        # ====================================================

        if st.session_state.pdf_name:

            if st.button(
                "🗑️ Remove PDF",
                use_container_width=True
            ):

                st.session_state.pdf_text = ""

                st.session_state.pdf_name = ""

                st.rerun()


# ============================================================
# PDF CONTEXT
# ============================================================

def get_pdf_context():

    if not st.session_state.pdf_text:

        return ""


    pdf_text = (
        st.session_state.pdf_text
    )


    # Keep prompts reasonably small
    if len(pdf_text) > 6000:

        pdf_text = pdf_text[:6000]


    return f"""

The student uploaded university study material.

Use this material as the main source.

If the answer cannot be found in the
provided material, say:

"I couldn't find this information
in your uploaded PDF."

PDF CONTENT:

-------------------------
{pdf_text}
-------------------------

"""


# ============================================================
# TEXT AI FUNCTION
# ============================================================

def generate_answer(prompt):

    mode = (
        st.session_state.selected_mode
    )


    instructions = {

        "💬 Normal Chat":
            """
            Answer the question clearly
            and accurately.
            """,

        "📚 Explain Topic":
            """
            Explain the topic in extremely
            simple English.

            Use examples and step-by-step
            explanations.
            """,

        "📝 Make Notes":
            """
            Create short revision notes.

            Use:
            - Headings
            - Bullet points
            - Definitions
            - Examples
            """,

        "❓ Generate MCQs":
            """
            Create 10 important MCQs.

            Each question must contain:

            A
            B
            C
            D

            Clearly show the correct answer.
            """,

        "🎯 Exam Questions":
            """
            Create important university
            exam-style questions.

            Include short and long questions.
            """
    }


    system_prompt = f"""

You are ASH Study Assistant.

{instructions[mode]}

Rules:

- Use simple English.
- Be accurate.
- Be concise.
- Explain difficult concepts step by step.
- Give examples when useful.
- Make exam answers easy to memorize.

{get_pdf_context()}

"""


    response = client.models.generate_content(

        model="gemini-3.7-flash",

        contents=(
            system_prompt
            + "\n\nStudent Question:\n"
            + prompt
        ),

        config=types.GenerateContentConfig(

            max_output_tokens=700
        )
    )


    if response.text:

        return response.text


    return (
        "I couldn't generate an answer."
    )


# ============================================================
# PROCESS TEXT QUESTION
# ============================================================

if typed_prompt:

    prompt = typed_prompt


    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(

        {
            "role": "user",
            "content": prompt
        }
    )


    with st.chat_message("user"):

        st.markdown(
            prompt
        )


    # ========================================================
    # IMAGE GENERATION
    # ========================================================

    if (
        st.session_state.selected_mode
        == "🎨 Generate Image"
    ):

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🎨 Creating image..."
            ):

                try:

                    image = (
                        image_client.text_to_image(

                            prompt=prompt,

                            model=(
                                "black-forest-labs/"
                                "FLUX.1-schnell"
                            )
                        )
                    )


                    st.image(
                        image,

                        caption=(
                            "Generated by "
                            "ASH Study Assistant"
                        ),

                        use_container_width=True
                    )


                    from io import BytesIO

                    image_buffer = BytesIO()


                    image.save(
                        image_buffer,
                        format="PNG"
                    )


                    st.download_button(

                        "⬇️ Download Image",

                        data=(
                            image_buffer.getvalue()
                        ),

                        file_name=(
                            "ash_study_image.png"
                        ),

                        mime="image/png"
                    )


                except Exception as e:

                    st.error(
                        "❌ Image generation failed."
                    )

                    st.code(
                        str(e)
                    )


    # ========================================================
    # NORMAL TEXT AI
    # ========================================================

    else:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🤖 Thinking..."
            ):

                try:

                    answer = (
                        generate_answer(
                            prompt
                        )
                    )


                    st.markdown(
                        answer
                    )


                    st.session_state.messages.append(

                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


                except Exception as e:

                    st.error(
                        "❌ Something went wrong."
                    )

                    st.code(
                        str(e)
                    )
