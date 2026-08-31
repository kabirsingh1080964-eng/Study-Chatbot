import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from huggingface_hub import InferenceClient
from io import BytesIO
from pathlib import Path


# ============================================================
# PAGE / FILE SETTINGS
# ============================================================

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "logo.png"

GEMINI_MODEL = "gemini-3.6-flash"
IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# Keep these smaller to reduce free-tier token usage
MAX_PDF_CHARACTERS = 30000
MAX_HISTORY_MESSAGES = 4


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ASH Study Assistant",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📚",
    layout="centered"
)


# ============================================================
# CHECK GEMINI API KEY
# ============================================================

if "GEMINI_API_KEY" not in st.secrets:

    st.error(
        "❌ GEMINI_API_KEY is missing.\n\n"
        "Go to Streamlit Cloud → Manage app → Settings → Secrets "
        "and add your Gemini API key."
    )

    st.stop()


# ============================================================
# GEMINI CONNECTION
# ============================================================

try:

    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

except Exception as e:

    st.error(
        "❌ Could not connect to Gemini."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# HUGGING FACE CONNECTION
# ============================================================

image_client = None

if "HF_TOKEN" in st.secrets:

    try:

        image_client = InferenceClient(
            api_key=st.secrets["HF_TOKEN"]
        )

    except Exception as e:

        st.warning(
            "⚠️ Hugging Face connection failed."
        )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "pdf_text" not in st.session_state:

    st.session_state.pdf_text = ""


if "pdf_name" not in st.session_state:

    st.session_state.pdf_name = ""


if "pdf_pages" not in st.session_state:

    st.session_state.pdf_pages = 0


if "selected_mode" not in st.session_state:

    st.session_state.selected_mode = "💬 Normal Chat"


# ============================================================
# LOGO
# ============================================================

if LOGO_PATH.exists():

    st.image(
        str(LOGO_PATH),
        width=150
    )

else:

    st.warning(
        "⚠️ logo.png was not found."
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "ASH Study Assistant"
)

st.write(
    "📚 Your AI-powered university study assistant."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📚 ASH Study Assistant"
    )

    st.write(
        "Your personal AI study partner."
    )

    st.divider()

    st.subheader(
        "Current Study Mode"
    )

    st.info(
        st.session_state.selected_mode
    )

    st.divider()

    if st.session_state.pdf_name:

        st.success(
            f"📄 {st.session_state.pdf_name}"
        )

        st.caption(
            f"{st.session_state.pdf_pages} pages"
        )

    else:

        st.caption(
            "No PDF uploaded."
        )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# BOTTOM TOOL AREA
# ============================================================

voice_col, chat_col, plus_col = st.columns(
    [1, 7, 1],
    vertical_alignment="bottom"
)


# ============================================================
# VOICE INPUT
# ============================================================

with voice_col:

    audio_value = st.audio_input(
        "🎤",
        label_visibility="collapsed"
    )


# ============================================================
# CHAT INPUT
# ============================================================

with chat_col:

    typed_prompt = st.chat_input(
        "Ask anything about your studies..."
    )


# ============================================================
# PLUS MENU
# ============================================================

with plus_col:

    with st.popover(
        "+",
        use_container_width=True
    ):

        st.subheader(
            "🛠️ Study Tools"
        )


        # ====================================================
        # STUDY MODES
        # ====================================================

        st.write(
            "📚 Choose Study Mode"
        )

        study_modes = [

            "💬 Normal Chat",

            "📚 Explain Topic",

            "📝 Make Notes",

            "❓ Generate MCQs",

            "🎯 Exam Questions",

            "🎨 Generate Image"

        ]


        selected_mode = st.radio(

            "Study Mode",

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

            "Choose your PDF",

            type=["pdf"],

            label_visibility="collapsed"

        )


        if uploaded_file is not None:

            # Process only if this is a new PDF

            if (
                uploaded_file.name
                != st.session_state.pdf_name
            ):

                try:

                    pdf_reader = PdfReader(
                        uploaded_file
                    )


                    pdf_parts = []


                    for page in pdf_reader.pages:

                        try:

                            text = page.extract_text()

                            if text:

                                pdf_parts.append(
                                    text
                                )

                        except Exception:

                            continue


                    pdf_text = "\n\n".join(
                        pdf_parts
                    )


                    # ----------------------------------------
                    # LIMIT PDF SIZE
                    # ----------------------------------------

                    if (
                        len(pdf_text)
                        > MAX_PDF_CHARACTERS
                    ):

                        pdf_text = pdf_text[
                            :MAX_PDF_CHARACTERS
                        ]

                        st.warning(
                            "⚠️ This PDF is large. "
                            "Only the first portion is being "
                            "used to save AI quota."
                        )


                    st.session_state.pdf_text = (
                        pdf_text
                    )


                    st.session_state.pdf_name = (
                        uploaded_file.name
                    )


                    st.session_state.pdf_pages = (
                        len(pdf_reader.pages)
                    )


                    st.success(
                        f"✅ {uploaded_file.name} loaded!"
                    )


                    st.caption(
                        f"{len(pdf_reader.pages)} pages"
                    )


                except Exception as e:

                    st.error(
                        "❌ Could not read this PDF."
                    )

                    st.code(
                        str(e)
                    )


        elif st.session_state.pdf_name:

            st.success(
                f"📄 {st.session_state.pdf_name} "
                "is loaded"
            )


        st.divider()


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

                st.session_state.pdf_pages = 0

                st.rerun()


# ============================================================
# VOICE PROCESSING
# ============================================================

voice_prompt = None


if audio_value is not None:

    with st.spinner(
        "🎧 Understanding your voice..."
    ):

        try:

            audio_file = client.files.upload(
                file=audio_value
            )


            voice_response = (
                client.models.generate_content(

                    model=GEMINI_MODEL,

                    contents=[

                        audio_file,

                        """
                        Listen to the student's
                        spoken question.

                        Convert the spoken question
                        into text.

                        Return ONLY the student's
                        question.

                        Do not answer it.

                        Do not add explanations.

                        Do not use quotation marks.
                        """

                    ]

                )
            )


            if voice_response.text:

                voice_prompt = (
                    voice_response.text.strip()
                )


                st.info(
                    "🎤 Voice question: "
                    + voice_prompt
                )


        except Exception as e:

            error_message = str(e)


            if (
                "quota" in error_message.lower()
                or "429" in error_message
                or "resource_exhausted"
                in error_message.lower()
            ):

                st.warning(
                    "⚠️ Your free Gemini quota "
                    "has been reached. "
                    "Please wait and try again later."
                )

            else:

                st.error(
                    "❌ Could not understand "
                    "the recording."
                )

                st.code(
                    error_message
                )


# ============================================================
# SELECT QUESTION
# ============================================================

prompt = typed_prompt


if voice_prompt:

    prompt = voice_prompt


# ============================================================
# PROCESS QUESTION
# ============================================================

if prompt:

    prompt = prompt.strip()


    if not prompt:

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(

        {
            "role": "user",

            "content": prompt
        }

    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            prompt
        )


    # ========================================================
    # CURRENT MODE
    # ========================================================

    mode = (
        st.session_state.selected_mode
    )


    # ========================================================
    # STUDY MODE INSTRUCTIONS
    # ========================================================

    instructions = {

        "💬 Normal Chat":

        """
        Answer the student's question
        clearly and accurately.

        Give a useful explanation,
        but do not unnecessarily make
        the response very long.
        """,


        "📚 Explain Topic":

        """
        Explain the topic in extremely
        simple words.

        Assume the student is a beginner.

        Use:

        - Simple language
        - Step-by-step explanation
        - Examples
        - Real-life examples when useful
        - Important points
        """,


        "📝 Make Notes":

        """
        Convert the requested topic into
        short and easy-to-revise notes.

        Use:

        - Headings
        - Bullet points
        - Definitions
        - Examples
        - Important points
        - Exam tips
        """,


        "❓ Generate MCQs":

        """
        Create 10 important university-level
        multiple-choice questions.

        Every question must contain:

        A
        B
        C
        D

        Clearly identify the correct answer.

        Give a short explanation after
        each correct answer.
        """,


        "🎯 Exam Questions":

        """
        Create important university
        exam-style questions.

        Include:

        - Short questions
        - Long questions
        - Conceptual questions
        - Important definitions

        Make them suitable for university
        examination preparation.
        """,


        "🎨 Generate Image":

        """
        Generate an educational image,
        diagram, illustration, or
        visualization based on the
        student's request.
        """

    }


    # ========================================================
    # PDF INSTRUCTIONS
    # ========================================================

    if st.session_state.pdf_text:

        pdf_instructions = f"""

The student has uploaded university
study material.

IMPORTANT PDF RULES:

1. Use the uploaded PDF as the
   primary source.

2. Answer using the PDF whenever
   the information is available.

3. Do not claim that information
   came from the PDF if it is not
   present in the PDF.

4. If the requested information
   cannot be found in the PDF,
   clearly say:

"I couldn't find this information
in your uploaded PDF."

5. You can simplify information
   from the PDF.

6. Do not invent page numbers
   or quotations.

UPLOADED PDF:

--------------------------------

{st.session_state.pdf_text}

--------------------------------
"""


    else:

        pdf_instructions = """

No PDF has been uploaded.

Use your general knowledge to
answer the student's question.
"""


    # ========================================================
    # IMAGE GENERATION
    # ========================================================

    if mode == "🎨 Generate Image":

        st.subheader(
            "🎨 AI Image Generator"
        )


        if image_client is None:

            st.error(
                "❌ Image generation is unavailable."
            )

            st.info(
                "Add HF_TOKEN to your "
                "Streamlit Secrets."
            )


        else:

            with st.spinner(
                "🎨 Creating your image..."
            ):

                try:

                    image_prompt = f"""

Create a high-quality educational
image based on this request:

{prompt}

Make it clear, visually attractive,
and suitable for a university student.

If the request is a concept,
create an educational diagram
or visualization.
"""


                    image = (
                        image_client.text_to_image(

                            image_prompt,

                            model=IMAGE_MODEL

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


                    # ----------------------------------------
                    # IMAGE DOWNLOAD
                    # ----------------------------------------

                    image_bytes = BytesIO()


                    image.save(

                        image_bytes,

                        format="PNG"

                    )


                    image_data = (
                        image_bytes.getvalue()
                    )


                    st.download_button(

                        label="⬇️ Download Image",

                        data=image_data,

                        file_name=(
                            "ash_study_image.png"
                        ),

                        mime="image/png",

                        use_container_width=True

                    )


                except Exception as e:

                    error_message = str(e)


                    if (
                        "quota"
                        in error_message.lower()
                        or "429"
                        in error_message
                    ):

                        st.warning(
                            "⚠️ The image generation "
                            "service has reached its "
                            "free usage limit."
                        )

                    else:

                        st.error(
                            "❌ Image generation failed."
                        )

                        st.code(
                            error_message
                        )


    # ========================================================
    # NORMAL AI CHAT
    # ========================================================

    else:

        # ====================================================
        # BUILD SHORT CHAT HISTORY
        # ====================================================

        history_text = ""


        recent_messages = (
            st.session_state.messages[
                -MAX_HISTORY_MESSAGES:
            ]
        )


        for message in recent_messages:

            role = message["role"]

            content = message["content"]


            if role == "user":

                history_text += (
                    f"\nStudent: {content}\n"
                )


            else:

                history_text += (
                    f"\nAssistant: {content}\n"
                )


        # ====================================================
        # SYSTEM PROMPT
        # ====================================================

        system_prompt = f"""

You are ASH Study Assistant,
an AI tutor for university students.

Your goal is to help students:

- Understand difficult subjects
- Prepare for exams
- Create notes
- Practice MCQs
- Understand concepts
- Study from uploaded PDFs


CURRENT STUDY MODE:

{instructions[mode]}


GENERAL RULES:

- Use simple English.
- Explain difficult concepts step by step.
- Give examples whenever useful.
- Be accurate.
- Do not intentionally invent facts.
- If you are uncertain, say so.
- Make exam answers easy to memorize.
- Use headings and bullet points when useful.
- Stay focused on the student's question.
- Avoid unnecessary repetition.


PDF INFORMATION:

{pdf_instructions}


RECENT CONVERSATION:

{history_text}


CURRENT STUDENT QUESTION:

{prompt}
"""


        # ====================================================
        # GEMINI RESPONSE
        # ====================================================

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🤖 Thinking..."
            ):

                try:

                    response = (
                        client.models.generate_content(

                            model=GEMINI_MODEL,

                            contents=system_prompt,

                            config=types.GenerateContentConfig(

                                max_output_tokens=800

                            )

                        )
                    )


                    # ----------------------------------------
                    # GET ANSWER
                    # ----------------------------------------

                    if response.text:

                        answer = (
                            response.text.strip()
                        )

                    else:

                        answer = (
                            "Sorry, I couldn't "
                            "generate an answer."
                        )


                    # ----------------------------------------
                    # DISPLAY ANSWER
                    # ----------------------------------------

                    st.markdown(
                        answer
                    )


                    # ----------------------------------------
                    # SAVE ANSWER
                    # ----------------------------------------

                    st.session_state.messages.append(

                        {
                            "role": "assistant",

                            "content": answer

                        }

                    )


                # =================================================
                # UPDATED ERROR HANDLING
                # =================================================

                except Exception as e:

                    error_message = str(e)


                    # --------------------------------------------
                    # QUOTA ERROR
                    # --------------------------------------------

                    if (
                        "quota"
                        in error_message.lower()

                        or "429"
                        in error_message

                        or "resource_exhausted"
                        in error_message.lower()

                        or "too many requests"
                        in error_message.lower()
                    ):

                        st.warning(
                            "⚠️ Gemini free quota "
                            "has been reached."
                        )

                        st.info(
                            "Please wait until the "
                            "quota resets and then "
                            "try again."
                        )


                    # --------------------------------------------
                    # API KEY ERROR
                    # --------------------------------------------

                    elif (
                        "api key"
                        in error_message.lower()

                        or "permission"
                        in error_message.lower()

                        or "unauthorized"
                        in error_message.lower()
                    ):

                        st.error(
                            "❌ Gemini API key problem."
                        )

                        st.info(
                            "Check GEMINI_API_KEY "
                            "in Streamlit Secrets."
                        )


                    # --------------------------------------------
                    # MODEL ERROR
                    # --------------------------------------------

                    elif (
                        "model"
                        in error_message.lower()
                    ):

                        st.error(
                            "❌ Gemini model error."
                        )

                        st.info(
                            f"Current model: "
                            f"{GEMINI_MODEL}"
                        )

                        st.code(
                            error_message
                        )


                    # --------------------------------------------
                    # OTHER ERROR
                    # --------------------------------------------

                    else:

                        st.error(
                            "❌ Something went wrong."
                        )

                        st.code(
                            error_message
                        )
