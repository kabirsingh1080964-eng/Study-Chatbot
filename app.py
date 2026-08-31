import streamlit as st
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from io import BytesIO
import base64


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

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    p, label {
        color: #ffffff !important;
    }

    [data-testid="stChatMessage"] {
        background: transparent !important;
        color: #ffffff !important;
    }

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

    button {
        background: #1f1f1f !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
        border-radius: 10px !important;
    }

    button:hover {
        background: #333333 !important;
    }

    [data-testid="stPopover"] {
        background: #111111 !important;
    }

    [data-testid="stFileUploader"] {
        background: #111111 !important;
        border-radius: 12px !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .ash-logo {
        display: flex;
        justify-content: center;
        margin-bottom: 5px;
    }

    .ash-subtitle {
        text-align: center;
        color: #888888 !important;
        margin-bottom: 30px;
        font-size: 15px;
    }

    .attachment-badge {
        background: #1f1f1f;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 7px 12px;
        margin-bottom: 10px;
        color: #dddddd !important;
        font-size: 13px;
    }

    /* Make the plus button compact */
    .plus-button-container {
        margin-top: 4px;
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

TEXT_MODEL = "Qwen/Qwen3-32B"

VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

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

if "image_type" not in st.session_state:
    st.session_state.image_type = "image/png"

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

        if content:
            st.markdown(content)

        if message.get("image") is not None:

            st.image(
                message["image"],
                use_container_width=True
            )


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
# BOTTOM CHAT BAR
#
# ➕ | Message ASH...
# ============================================================

plus_col, chat_col = st.columns(
    [0.75, 9.25],
    vertical_alignment="bottom"
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
        # STUDY MODE
        # ====================================================

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


        # ====================================================
        # PDF UPLOAD
        # ====================================================

        st.markdown(
            "### 📄 Upload PDF"
        )


        pdf_file = st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
            label_visibility="collapsed",
            key="ash_pdf_upload"
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
                    "❌ Could not read PDF."
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
                use_container_width=True,
                key="remove_pdf"
            ):

                st.session_state.pdf_text = ""

                st.session_state.pdf_name = ""

                st.rerun()


        st.divider()


        # ====================================================
        # IMAGE UPLOAD
        # ====================================================

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
            label_visibility="collapsed",
            key="ash_image_upload"
        )


        if image_file is not None:

            st.session_state.image_bytes = (
                image_file.getvalue()
            )

            st.session_state.image_name = (
                image_file.name
            )

            st.session_state.image_type = (
                image_file.type
                or "image/png"
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
                use_container_width=True,
                key="remove_image"
            ):

                st.session_state.image_bytes = None

                st.session_state.image_name = ""

                st.session_state.image_type = (
                    "image/png"
                )

                st.rerun()


# ============================================================
# CHAT INPUT
# ============================================================

with chat_col:

    prompt = st.chat_input(
        "Message ASH..."
    )


# ============================================================
# STUDY INSTRUCTIONS
# ============================================================

def get_instructions():

    instructions = {

        "Normal Chat":
            """
            Answer the user's question clearly,
            accurately and directly.
            """,

        "Explain Topic":
            """
            Explain the topic as if the student
            is learning it for the first time.

            Use:
            - Very simple language
            - Examples
            - Step-by-step explanations
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

            Every question must have:

            A
            B
            C
            D

            Clearly show the correct answer.
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
# PDF CONTEXT
# ============================================================

def get_pdf_context():

    if not st.session_state.pdf_text:

        return ""


    pdf_text = st.session_state.pdf_text


    # Limit PDF context for speed
    if len(pdf_text) > 8000:

        pdf_text = pdf_text[:8000]


    return f"""

The user uploaded a PDF.

Use the PDF as the main source when
answering questions related to it.

If the requested information is not
present in the PDF, clearly tell the
user that you could not find it there.

PDF CONTENT:

-------------------------
{pdf_text}
-------------------------
"""


# ============================================================
# BUILD CONVERSATION
# ============================================================

def build_messages(prompt):

    system_prompt = f"""
You are ASH, an all-in-one AI assistant.

{get_instructions()}

Rules:

- Answer the user's actual question.
- Give direct and useful information.
- Use simple language.
- Be accurate.
- Avoid unnecessary repetition.
- Explain difficult concepts step by step.
- Give examples when useful.
- Help university students understand topics.
- Make exam answers easy to memorize.
- Do not invent information from uploaded PDFs.

{get_pdf_context()}
"""


    conversation = [

        {
            "role": "system",
            "content": system_prompt
        }

    ]


    # Keep only recent conversation
    recent_messages = (
        st.session_state.messages[-12:]
    )


    for message in recent_messages:

        role = message.get("role")

        content = message.get(
            "content",
            ""
        )


        if role in [
            "user",
            "assistant"
        ] and content:

            conversation.append(
                {
                    "role": role,
                    "content": content
                }
            )


    # Current prompt
    conversation.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    return conversation


# ============================================================
# ASK AI
# ============================================================

def ask_ai(prompt):

    messages = build_messages(
        prompt
    )


    response = client.chat.completions.create(

        model=TEXT_MODEL,

        messages=messages,

        max_tokens=700,

        temperature=0.6
    )


    return (
        response.choices[0]
        .message
        .content
    )


# ============================================================
# STREAM AI
# ============================================================

def stream_ai(prompt):

    messages = build_messages(
        prompt
    )


    stream = client.chat.completions.create(

        model=TEXT_MODEL,

        messages=messages,

        max_tokens=700,

        temperature=0.6,

        stream=True
    )


    for chunk in stream:

        try:

            token = (
                chunk.choices[0]
                .delta
                .content
            )


            if token:

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


    encoded_image = base64.b64encode(
        image_data
    ).decode()


    mime_type = (
        st.session_state.image_type
        or "image/png"
    )


    image_url = (
        f"data:{mime_type};base64,"
        + encoded_image
    )


    messages = [

        {
            "role": "system",

            "content":
                """
                You are ASH, an intelligent
                AI assistant.

                Carefully analyze uploaded
                images.

                Answer the user's question
                accurately and clearly.
                """
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

                        "url": image_url
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
# IMAGE REQUEST DETECTION
# ============================================================

def is_image_request(prompt):

    text = prompt.lower().strip()


    keywords = [

        "generate image",

        "create image",

        "make an image",

        "make image",

        "generate a picture",

        "create a picture",

        "make a picture",

        "draw "

    ]


    return any(
        text.startswith(keyword)
        for keyword in keywords
    )


# ============================================================
# GET IMAGE PROMPT
# ============================================================

def clean_image_prompt(prompt):

    prefixes = [

        "generate image",

        "create image",

        "make an image",

        "make image",

        "generate a picture",

        "create a picture",

        "make a picture",

        "draw"

    ]


    result = prompt.strip()


    for prefix in prefixes:

        if result.lower().startswith(
            prefix
        ):

            result = result[
                len(prefix):
            ].strip()

            break


    return result


# ============================================================
# PROCESS MESSAGE
# ============================================================

if prompt:

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(

        {
            "role": "user",

            "content": prompt
        }

    )


    # ========================================================
    # SHOW USER
    # ========================================================

    with st.chat_message("user"):

        st.markdown(
            prompt
        )


    # ========================================================
    # ASSISTANT
    # ========================================================

    with st.chat_message("assistant"):

        try:

            # =================================================
            # IMAGE GENERATION
            # =================================================

            if is_image_request(
                prompt
            ):

                image_prompt = (
                    clean_image_prompt(
                        prompt
                    )
                )


                with st.spinner(
                    "🎨 Creating image..."
                ):

                    generated_image = (
                        create_image(
                            image_prompt
                        )
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


                image_bytes = (
                    buffer.getvalue()
                )


                st.download_button(

                    "⬇️ Download Image",

                    data=image_bytes,

                    file_name="ash_image.png",

                    mime="image/png",

                    key="download_generated_image"
                )


                # Save assistant response
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


                # Clear image
                st.session_state.image_bytes = None

                st.session_state.image_name = ""

                st.session_state.image_type = (
                    "image/png"
                )


            # =================================================
            # NORMAL AI CHAT
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
                        "Sorry, I couldn't "
                        "generate an answer."
                    )


                    placeholder.markdown(
                        full_answer
                    )


                # Save assistant response
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

            error_message = str(
                error
            )


            if (
                "429" in error_message
                or
                "Too Many Requests"
                in error_message
            ):

                st.error(
                    """
                    ⚠️ ASH has temporarily
                    reached the free AI
                    inference limit.

                    Please wait a little
                    and try again.
                    """
                )


            elif (
                "503" in error_message
                or
                "Service Unavailable"
                in error_message
            ):

                st.error(
                    """
                    ⚠️ The AI model is
                    temporarily busy.

                    Please try again in
                    a few moments.
                    """
                )


            elif (
                "401" in error_message
                or
                "403" in error_message
            ):

                st.error(
                    """
                    ⚠️ Your Hugging Face
                    token is invalid or
                    does not have the
                    required permission.
                    """
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
        margin-bottom:20px;
    ">
        ASH • Your all-in-one AI assistant
    </div>
    """,
    unsafe_allow_html=True
)
