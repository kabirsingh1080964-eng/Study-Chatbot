import streamlit as st
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from io import BytesIO
import time


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="ASH",
    page_icon="logo.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CHATGPT STYLE - BLACK UI
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background: #000000 !important;
        color: #ffffff !important;
    }

    .main {
        background: #000000 !important;
    }

    .block-container {
        max-width: 900px;
        padding-top: 20px;
        padding-bottom: 120px;
        background: #000000 !important;
    }


    /* Text */

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    p, label, span {
        color: #ffffff;
    }


    /* Chat messages */

    [data-testid="stChatMessage"] {
        background: transparent !important;
        color: #ffffff !important;
    }


    /* Chat input */

    [data-testid="stChatInput"] {
        background: #1f1f1f !important;
        border: 1px solid #444444 !important;
        border-radius: 24px !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #1f1f1f !important;
        color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #999999 !important;
    }


    /* Buttons */

    button {
        background: #1f1f1f !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
        border-radius: 10px !important;
    }

    button:hover {
        background: #333333 !important;
    }


    /* Popover */

    [data-testid="stPopover"] {
        background: #111111 !important;
    }


    /* File uploader */

    [data-testid="stFileUploader"] {
        background: #111111 !important;
        border-radius: 12px !important;
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


    /* Logo */

    .ash-logo {
        display: flex;
        justify-content: center;
        margin-bottom: 5px;
    }


    /* Subtitle */

    .ash-subtitle {
        text-align: center;
        color: #888888 !important;
        margin-bottom: 30px;
        font-size: 15px;
    }


    /* Attachment badge */

    .attachment-badge {
        background: #1f1f1f;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 7px 12px;
        margin-bottom: 10px;
        color: #dddddd !important;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

@st.cache_resource
def get_client():

    token = st.secrets.get("HF_TOKEN")

    if not token:

        st.error(
            "HF_TOKEN is missing from Streamlit Secrets."
        )

        st.stop()

    return InferenceClient(
        token=token,
        provider="auto"
    )


client = get_client()


# ============================================================
# MODELS
# ============================================================

# Main text model
TEXT_MODEL = "Qwen/Qwen3-32B"

# Vision model for uploaded images
VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Image generation
IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""


if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""


if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None


if "image_name" not in st.session_state:
    st.session_state.image_name = ""


if "mode" not in st.session_state:
    st.session_state.mode = "Normal Chat"


# ============================================================
# HEADER
# ============================================================

try:

    st.markdown(
        "<div class='ash-logo'>",
        unsafe_allow_html=True
    )

    st.image(
        "logo.png",
        width=120
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

except Exception:

    st.markdown(
        """
        <h1 style="text-align:center;">
            ASH
        </h1>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div class="ash-subtitle">
        Your all-in-one AI assistant
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message.get("role")

    content = message.get("content", "")

    with st.chat_message(role):

        st.markdown(content)

        if message.get("image") is not None:

            st.image(
                message["image"],
                use_container_width=True
            )


# ============================================================
# PLUS MENU
# ============================================================

with st.popover(
    "➕",
    use_container_width=False
):

    st.markdown(
        "### ASH Tools"
    )


    # ========================================================
    # STUDY MODE
    # ========================================================

    modes = [
        "Normal Chat",
        "Explain Topic",
        "Make Notes",
        "Generate MCQs",
        "Exam Questions"
    ]


    selected_mode = st.selectbox(
        "Study Mode",
        modes,
        index=modes.index(
            st.session_state.mode
        )
    )


    st.session_state.mode = selected_mode


    st.divider()


    # ========================================================
    # PDF
    # ========================================================

    st.markdown(
        "### 📄 Upload PDF"
    )


    pdf_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )


    if pdf_file is not None:

        try:

            reader = PdfReader(
                pdf_file
            )

            extracted_text = []


            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:

                    extracted_text.append(
                        page_text
                    )


            st.session_state.pdf_text = (
                "\n".join(extracted_text)
            )

            st.session_state.pdf_name = (
                pdf_file.name
            )


            st.success(
                f"✅ {pdf_file.name} loaded"
            )


            st.caption(
                f"{len(reader.pages)} pages"
            )


        except Exception as error:

            st.error(
                "Could not read PDF."
            )

            st.code(
                str(error)
            )


    if st.session_state.pdf_name:

        st.caption(
            "📄 "
            + st.session_state.pdf_name
        )


        if st.button(
            "🗑️ Remove PDF",
            use_container_width=True
        ):

            st.session_state.pdf_text = ""

            st.session_state.pdf_name = ""

            st.rerun()


    st.divider()


    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    st.markdown(
        "### 🖼️ Upload Image"
    )


    image_file = st.file_uploader(
        "Choose an image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],
        label_visibility="collapsed"
    )


    if image_file is not None:

        st.session_state.image_bytes = (
            image_file.getvalue()
        )

        st.session_state.image_name = (
            image_file.name
        )


        st.image(
            st.session_state.image_bytes,
            use_container_width=True
        )


        st.success(
            "✅ Image attached"
        )


    if st.session_state.image_bytes:

        if st.button(
            "🗑️ Remove Image",
            use_container_width=True
        ):

            st.session_state.image_bytes = None

            st.session_state.image_name = ""

            st.rerun()


# ============================================================
# ATTACHMENT STATUS
# ============================================================

if st.session_state.pdf_name:

    st.markdown(
        f"""
        <div class="attachment-badge">
            📄 PDF: {st.session_state.pdf_name}
        </div>
        """,
        unsafe_allow_html=True
    )


if st.session_state.image_name:

    st.markdown(
        f"""
        <div class="attachment-badge">
            🖼️ Image: {st.session_state.image_name}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# STUDY INSTRUCTIONS
# ============================================================

def get_instructions():

    instructions = {

        "Normal Chat":
            """
            Answer the user's question clearly and accurately.
            Give useful information directly.
            """,

        "Explain Topic":
            """
            Explain the topic like a beginner is learning it
            for the first time.

            Use simple language, examples and steps.
            """,

        "Make Notes":
            """
            Create concise university study notes.

            Use:
            - Headings
            - Bullet points
            - Definitions
            - Examples
            - Important points
            """,

        "Generate MCQs":
            """
            Create 10 important MCQs.

            Every question must contain:
            A
            B
            C
            D

            Clearly identify the correct answer.
            """,

        "Exam Questions":
            """
            Create important university exam questions.

            Include:
            - Short questions
            - Long questions
            - Important concepts
            """
    }


    return instructions.get(
        st.session_state.mode,
        instructions["Normal Chat"]
    )


# ============================================================
# BUILD PDF CONTEXT
# ============================================================

def get_pdf_context():

    if not st.session_state.pdf_text:

        return ""


    pdf_text = st.session_state.pdf_text


    # Keep prompt reasonably small
    if len(pdf_text) > 8000:

        pdf_text = pdf_text[:8000]


    return f"""

The user uploaded study material.

Use the uploaded PDF as the main source
when answering questions related to it.

If the answer is not present in the
provided PDF, clearly say that you
could not find it in the uploaded PDF.

PDF CONTENT:

-------------------------
{pdf_text}
-------------------------
"""


# ============================================================
# ASK TEXT AI
# ============================================================

def ask_ai(prompt):

    system_prompt = f"""
You are ASH, an all-in-one AI assistant.

{get_instructions()}

Rules:

- Answer the user's actual question.
- Give direct and useful answers.
- Use simple language.
- Be accurate.
- Avoid unnecessary repetition.
- Explain difficult concepts step by step.
- Give examples when useful.
- For university questions, make answers easy to study.
- Do not claim information came from a PDF unless it is
  actually contained in the provided PDF.

{get_pdf_context()}
"""


    # --------------------------------------------------------
    # Previous conversation
    # --------------------------------------------------------

    conversation = [

        {
            "role": "system",
            "content": system_prompt
        }

    ]


    # Keep recent history to avoid huge prompts
    recent_messages = (
        st.session_state.messages[-10:]
    )


    for message in recent_messages:

        if message.get("role") in [
            "user",
            "assistant"
        ]:

            conversation.append(
                {
                    "role": message["role"],
                    "content": message["content"]
                }
            )


    # Current question
    conversation.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    response = client.chat.completions.create(

        model=TEXT_MODEL,

        messages=conversation,

        max_tokens=700,

        temperature=0.6
    )


    return (
        response.choices[0]
        .message
        .content
    )


# ============================================================
# STREAM TEXT AI
# ============================================================

def stream_ai(prompt):

    system_prompt = f"""
You are ASH, an all-in-one AI assistant.

{get_instructions()}

Rules:

- Answer the user's actual question.
- Be direct.
- Use simple language.
- Be accurate.
- Avoid unnecessary repetition.
- Explain difficult concepts step by step.
- Give examples when useful.
- Make university answers easy to memorize.

{get_pdf_context()}
"""


    conversation = [

        {
            "role": "system",
            "content": system_prompt
        }

    ]


    recent_messages = (
        st.session_state.messages[-10:]
    )


    for message in recent_messages:

        if message.get("role") in [
            "user",
            "assistant"
        ]:

            conversation.append(
                {
                    "role": message["role"],
                    "content": message["content"]
                }
            )


    conversation.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    stream = client.chat.completions.create(

        model=TEXT_MODEL,

        messages=conversation,

        max_tokens=700,

        temperature=0.6,

        stream=True
    )


    full_answer = ""


    for chunk in stream:

        try:

            token = (
                chunk.choices[0]
                .delta
                .content
            )

            if token:

                full_answer += token

                yield token

        except Exception:

            continue


# ============================================================
# IMAGE GENERATION
# ============================================================

def create_image(prompt):

    image = client.text_to_image(

        prompt=prompt,

        model=IMAGE_MODEL
    )

    return image


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(prompt):

    if not st.session_state.image_bytes:

        return ask_ai(prompt)


    image_data = (
        st.session_state.image_bytes
    )


    messages = [

        {
            "role": "system",
            "content": (
                "You are ASH, an intelligent AI assistant. "
                "Analyze the uploaded image carefully and "
                "answer the user's question accurately."
            )
        },

        {
            "role": "user",

            "content": [

                {
                    "type": "text",
                    "text": prompt
                },

                {
                    "type": "image_url",

                    "image_url": {
                        "url": (
                            "data:image/png;base64,"
                            + __import__("base64")
                            .b64encode(
                                image_data
                            )
                            .decode()
                        )
                    }
                }

            ]
        }

    ]


    response = client.chat.completions.create(

        model=VISION_MODEL,

        messages=messages,

        max_tokens=700
    )


    return (
        response.choices[0]
        .message
        .content
    )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Message ASH..."
)


# ============================================================
# PROCESS MESSAGE
# ============================================================

if prompt:

    # ========================================================
    # IMAGE REQUEST DETECTION
    # ========================================================

    lower_prompt = prompt.lower()


    image_request = (

        lower_prompt.startswith(
            "generate image"
        )

        or lower_prompt.startswith(
            "create image"
        )

        or lower_prompt.startswith(
            "make an image"
        )

        or lower_prompt.startswith(
            "draw "
        )

        or lower_prompt.startswith(
            "generate a picture"
        )

        or lower_prompt.startswith(
            "create a picture"
        )
    )


    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(

        {
            "role": "user",
            "content": prompt
        }

    )


    with st.chat_message("user"):

        st.markdown(prompt)


    # ========================================================
    # ASSISTANT
    # ========================================================

    with st.chat_message("assistant"):

        try:

            # =================================================
            # IMAGE GENERATION
            # =================================================

            if image_request:

                image_prompt = prompt


                prefixes = [

                    "generate image",

                    "create image",

                    "make an image",

                    "draw",

                    "generate a picture",

                    "create a picture"

                ]


                for prefix in prefixes:

                    if image_prompt.lower().startswith(
                        prefix
                    ):

                        image_prompt = (
                            image_prompt[
                                len(prefix):
                            ].strip()
                        )

                        break


                with st.spinner(
                    "🎨 Creating image..."
                ):

                    generated_image = create_image(
                        image_prompt
                    )


                st.image(
                    generated_image,
                    use_container_width=True
                )


                buffer = BytesIO()


                generated_image.save(

                    buffer,

                    format="PNG"
                )


                st.download_button(

                    "⬇️ Download Image",

                    data=buffer.getvalue(),

                    file_name="ash_image.png",

                    mime="image/png"
                )


                st.session_state.messages.append(

                    {
                        "role": "assistant",

                        "content":
                            "🎨 Here is your generated image.",

                        "image":
                            generated_image
                    }

                )


            # =================================================
            # IMAGE ANALYSIS
            # =================================================

            elif st.session_state.image_bytes:

                with st.spinner(
                    "🖼️ Analyzing image..."
                ):

                    answer = analyze_image(
                        prompt
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


                # Clear image after use
                st.session_state.image_bytes = None

                st.session_state.image_name = ""


            # =================================================
            # NORMAL AI
            # =================================================

            else:

                placeholder = st.empty()

                full_answer = ""


                with st.spinner(
                    "ASH is thinking..."
                ):

                    for token in stream_ai(
                        prompt
                    ):

                        full_answer += token

                        placeholder.markdown(
                            full_answer
                        )


                if not full_answer:

                    full_answer = (
                        "Sorry, I couldn't generate "
                        "an answer."
                    )

                    placeholder.markdown(
                        full_answer
                    )


                st.session_state.messages.append(

                    {
                        "role": "assistant",
                        "content": full_answer
                    }

                )


        # ====================================================
        # ERROR HANDLING
        # ====================================================

        except Exception as error:

            error_message = str(error)


            if (
                "429" in error_message
                or
                "Too Many Requests"
                in error_message
            ):

                st.error(
                    "⚠️ ASH has temporarily reached "
                    "the free inference limit. "
                    "Please wait and try again."
                )


            elif (
                "503" in error_message
                or
                "Service Unavailable"
                in error_message
            ):

                st.error(
                    "⚠️ The selected AI provider is "
                    "temporarily busy. Please try again "
                    "in a moment."
                )


            elif (
                "401" in error_message
                or
                "403" in error_message
            ):

                st.error(
                    "⚠️ Your Hugging Face token is "
                    "invalid or does not have the "
                    "required permission."
                )


            else:

                st.error(
                    "❌ ASH could not process your request."
                )

                st.code(
                    error_message
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#666666;
        font-size:12px;
        margin-top:40px;
    ">
        ASH • Your all-in-one AI assistant
    </div>
    """,
    unsafe_allow_html=True
)
