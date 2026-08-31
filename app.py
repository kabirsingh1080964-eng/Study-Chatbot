import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from huggingface_hub import InferenceClient
from io import BytesIO
import time


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ASH",
    page_icon="logo.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS — CHATGPT STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Center logo */
    .ash-header {
        text-align: center;
        padding-top: 10px;
        padding-bottom: 20px;
    }

    .ash-header img {
        width: 110px;
        border-radius: 20px;
    }

    .ash-name {
        font-size: 34px;
        font-weight: 700;
        margin-top: 5px;
    }

    .ash-tagline {
        color: #777;
        font-size: 15px;
        margin-top: -5px;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 14px;
    }

    /* Bottom area */
    .stChatInput {
        border-radius: 20px;
    }

    /* Buttons */
    button {
        border-radius: 12px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
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
def get_huggingface_client():

    return InferenceClient(
        api_key=st.secrets["HF_TOKEN"]
    )


client = get_gemini_client()
image_client = get_huggingface_client()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""


if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""


if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "💬 Normal Chat"


# ============================================================
# ASH HEADER
# ============================================================

st.markdown(
    """
    <div class="ash-header">
    """,
    unsafe_allow_html=True
)

try:

    st.image(
        "logo.png",
        width=110
    )

except Exception:

    st.markdown(
        "<div style='font-size:70px;'>🤖</div>",
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div class="ash-name">
        ASH
    </div>

    <div class="ash-tagline">
        Your all-in-one AI assistant
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        if message.get("type") == "image":

            st.image(
                message["content"],
                use_container_width=True
            )

        else:

            st.markdown(
                message["content"]
            )


# ============================================================
# PDF CONTEXT
# ============================================================

def get_pdf_context():

    if not st.session_state.pdf_text:

        return ""

    pdf_text = st.session_state.pdf_text

    # Keep prompt small for faster responses
    max_chars = 7000

    if len(pdf_text) > max_chars:

        pdf_text = pdf_text[:max_chars]

    return f"""

The user uploaded a study PDF.

Use the PDF information when it is relevant.

If the requested information is not present
in the PDF, use your general knowledge.

PDF CONTENT:

-------------------------
{pdf_text}
-------------------------
"""


# ============================================================
# AI RESPONSE FUNCTION
# ============================================================

def get_ai_response(
    prompt,
    image=None
):

    mode = st.session_state.selected_mode


    # --------------------------------------------------------
    # MODE INSTRUCTIONS
    # --------------------------------------------------------

    mode_instructions = {

        "💬 Normal Chat":
            """
            Answer the user's question clearly.
            Give useful and accurate information.
            """,

        "📚 Study Assistant":
            """
            Act as a university study assistant.

            Explain concepts in simple language.
            Use examples.
            Break difficult topics into steps.
            Make information easy to remember.
            """,

        "📝 Make Notes":
            """
            Convert the requested topic into
            short, organized study notes.

            Use:
            - Headings
            - Bullet points
            - Definitions
            - Examples
            """,

        "❓ Generate MCQs":
            """
            Generate useful multiple-choice questions.

            Each question should contain:
            A
            B
            C
            D

            Clearly identify the correct answer.
            """,

        "🎯 Exam Preparation":
            """
            Help the student prepare for university exams.

            Focus on:
            - Important concepts
            - Short questions
            - Long questions
            - Examples
            - Exam tips
            """
    }


    instructions = mode_instructions.get(
        mode,
        mode_instructions["💬 Normal Chat"]
    )


    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = f"""
You are ASH, an advanced all-in-one AI assistant.

{instructions}

General rules:

- Answer the user's actual question.
- Use simple and clear language.
- Be accurate.
- Do not unnecessarily repeat information.
- Use headings and bullet points when useful.
- Give examples when helpful.
- If the question requires calculations, show the steps.
- If the user asks for programming help, provide correct code.
- If the user asks about studies, teach instead of just giving
  a one-line answer.
- Keep responses reasonably concise.

{get_pdf_context()}
"""


    # --------------------------------------------------------
    # CONTENT
    # --------------------------------------------------------

    contents = [
        system_prompt
    ]


    # --------------------------------------------------------
    # IMAGE UNDERSTANDING
    # --------------------------------------------------------

    if image is not None:

        contents.append(
            image
        )


        contents.append(
            """
            Analyze the uploaded image carefully
            and answer the user's question about it.
            """
        )


    contents.append(
        f"""

USER QUESTION:

{prompt}
"""
    )


    # --------------------------------------------------------
    # GEMINI REQUEST
    # --------------------------------------------------------

    # A few retries help when Gemini temporarily
    # returns 503 UNAVAILABLE.

    for attempt in range(3):

        try:

            response = client.models.generate_content(

                model="gemini-3.7-flash",

                contents=contents,

                config=types.GenerateContentConfig(

                    max_output_tokens=700,

                    temperature=0.5
                )
            )


            if response.text:

                return response.text.strip()


            return (
                "Sorry, I couldn't generate "
                "an answer."
            )


        except Exception as error:

            error_text = str(error)


            # Retry temporary 503 errors

            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "high demand" in error_text.lower()
            ):

                if attempt < 2:

                    time.sleep(
                        1.5 * (attempt + 1)
                    )

                    continue


            raise error


# ============================================================
# IMAGE GENERATION FUNCTION
# ============================================================

def generate_image(prompt):

    image = image_client.text_to_image(

        prompt=prompt,

        model=(
            "black-forest-labs/"
            "FLUX.1-schnell"
        )
    )

    return image


# ============================================================
# CHAT INPUT + PLUS BUTTON
# ============================================================

chat_col, plus_col = st.columns(
    [9, 1],
    vertical_alignment="bottom"
)


# ============================================================
# CHAT INPUT
# ============================================================

with chat_col:

    prompt = st.chat_input(
        "Message ASH..."
    )


# ============================================================
# PLUS BUTTON
# ============================================================

with plus_col:

    with st.popover(
        "➕",
        use_container_width=True
    ):

        st.markdown(
            "### ASH Tools"
        )


        # ====================================================
        # STUDY MODES
        # ====================================================

        st.markdown(
            "#### 📚 Study Mode"
        )


        modes = [

            "💬 Normal Chat",

            "📚 Study Assistant",

            "📝 Make Notes",

            "❓ Generate MCQs",

            "🎯 Exam Preparation",

            "🎨 Generate Image"
        ]


        selected_mode = st.selectbox(

            "Choose mode",

            modes,

            index=modes.index(
                st.session_state.selected_mode
            )
        )


        st.session_state.selected_mode = (
            selected_mode
        )


        st.divider()


        # ====================================================
        # IMAGE UPLOAD
        # ====================================================

        st.markdown(
            "#### 🖼️ Upload Image"
        )


        uploaded_image = st.file_uploader(

            "Upload an image",

            type=[
                "png",
                "jpg",
                "jpeg",
                "webp"
            ],

            label_visibility="collapsed"
        )


        if uploaded_image is not None:

            try:

                image_bytes = (
                    uploaded_image.getvalue()
                )

                st.session_state.uploaded_image = (
                    image_bytes
                )

                st.image(
                    image_bytes,
                    caption="Image attached",
                    use_container_width=True
                )

            except Exception as error:

                st.error(
                    "Could not load image."
                )


        st.divider()


        # ====================================================
        # PDF UPLOAD
        # ====================================================

        st.markdown(
            "#### 📄 Upload PDF"
        )


        uploaded_pdf = st.file_uploader(

            "Upload study material",

            type=["pdf"],

            label_visibility="collapsed"
        )


        if uploaded_pdf is not None:

            try:

                reader = PdfReader(
                    uploaded_pdf
                )


                extracted_text = []


                for page in reader.pages:

                    text = page.extract_text()

                    if text:

                        extracted_text.append(
                            text
                        )


                final_text = "\n".join(
                    extracted_text
                )


                st.session_state.pdf_text = (
                    final_text
                )


                st.session_state.pdf_name = (
                    uploaded_pdf.name
                )


                st.success(
                    f"✅ {uploaded_pdf.name}"
                )


                st.caption(
                    f"{len(reader.pages)} pages loaded"
                )


            except Exception as error:

                st.error(
                    "Could not read PDF."
                )


        elif st.session_state.pdf_name:

            st.success(
                f"📄 {st.session_state.pdf_name}"
            )


        # ====================================================
        # REMOVE FILES
        # ====================================================

        st.divider()


        if (
            st.session_state.pdf_name
            or st.session_state.uploaded_image
        ):

            if st.button(
                "🗑️ Remove attachments",
                use_container_width=True
            ):

                st.session_state.pdf_text = ""

                st.session_state.pdf_name = ""

                st.session_state.uploaded_image = None

                st.rerun()


# ============================================================
# PROCESS USER PROMPT
# ============================================================

if prompt:

    # ========================================================
    # USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    with st.chat_message("user"):

        st.markdown(prompt)


        # Show uploaded image

        if st.session_state.uploaded_image:

            st.image(
                st.session_state.uploaded_image,
                width=300
            )


    # ========================================================
    # CURRENT MODE
    # ========================================================

    mode = st.session_state.selected_mode


    # ========================================================
    # IMAGE GENERATION
    # ========================================================

    if mode == "🎨 Generate Image":

        with st.chat_message("assistant"):

            with st.spinner(
                "🎨 Creating your image..."
            ):

                try:

                    generated_image = (
                        generate_image(prompt)
                    )


                    st.image(
                        generated_image,
                        caption="Generated by ASH",
                        use_container_width=True
                    )


                    # Save image in chat

                    image_buffer = BytesIO()


                    generated_image.save(
                        image_buffer,
                        format="PNG"
                    )


                    image_data = (
                        image_buffer.getvalue()
                    )


                    st.download_button(

                        "⬇️ Download Image",

                        data=image_data,

                        file_name=(
                            "ash_generated_image.png"
                        ),

                        mime="image/png"
                    )


                except Exception as error:

                    st.error(
                        "❌ Image generation failed."
                    )


                    st.code(
                        str(error)
                    )


    # ========================================================
    # NORMAL AI RESPONSE
    # ========================================================

    else:

        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 ASH is thinking..."
            ):

                try:

                    # -----------------------------------------
                    # IMAGE
                    # -----------------------------------------

                    image_input = None


                    if (
                        st.session_state.uploaded_image
                    ):

                        image_input = (
                            types.Part.from_bytes(
                                data=(
                                    st.session_state
                                    .uploaded_image
                                ),
                                mime_type=(
                                    "image/png"
                                )
                            )
                        )


                    # -----------------------------------------
                    # GET RESPONSE
                    # -----------------------------------------

                    answer = get_ai_response(

                        prompt,

                        image=image_input
                    )


                    # -----------------------------------------
                    # DISPLAY
                    # -----------------------------------------

                    st.markdown(
                        answer
                    )


                    # -----------------------------------------
                    # SAVE
                    # -----------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


                except Exception as error:

                    error_text = str(error)


                    if (
                        "503" in error_text
                        or "UNAVAILABLE"
                        in error_text
                    ):

                        st.error(
                            "⚠️ Gemini is temporarily "
                            "busy. Please try again "
                            "in a few seconds."
                        )

                    else:

                        st.error(
                            "❌ Something went wrong."
                        )


                    st.code(
                        error_text
                    )
