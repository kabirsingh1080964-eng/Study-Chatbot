import streamlit as st
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from PIL import Image
from io import BytesIO
import base64


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ASH",
    page_icon="logo.png",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CSS - CHATGPT STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* BLACK BACKGROUND */
    .stApp {
        background: #000000;
        color: #ffffff;
    }

    /* Main content */
    .block-container {
        background: #000000;
    }

    /* Text */
    h1, h2, h3, p, span, label {
        color: #ffffff !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #000000;
        color: #ffffff;
        border-radius: 12px;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: #1a1a1a;
        border: 1px solid #444444;
        border-radius: 20px;
    }

    /* Chat input text */
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        background: #1a1a1a !important;
    }

    /* Placeholder */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #999999 !important;
    }

    /* Buttons */
    button {
        background: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
    }

    /* Popover */
    [data-testid="stPopover"] {
        background: #111111;
        color: #ffffff;
    }

    /* Select boxes */
    div[data-baseweb="select"] {
        background: #1a1a1a;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)

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

    /* Center content */
    .block-container {
        max-width: 900px;
        padding-top: 20px;
        padding-bottom: 120px;
    }

    /* ASH logo */
    .ash-logo {
        display: flex;
        justify-content: center;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .ash-name {
        text-align: center;
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .ash-subtitle {
        text-align: center;
        color: #777;
        font-size: 14px;
        margin-bottom: 35px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 8px;
    }

    /* Input */
    [data-testid="stChatInput"] {
        border-radius: 20px;
    }

    /* Buttons */
    button {
        border-radius: 10px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HF CLIENT
# ============================================================

@st.cache_resource
def get_hf_client():

    token = st.secrets.get("HF_TOKEN", "")

    if not token:
        st.error(
            "HF_TOKEN is missing. Add it to Streamlit Secrets."
        )

        st.stop()

    return InferenceClient(
        token=token
    )


hf_client = get_hf_client()


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

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "mode" not in st.session_state:
    st.session_state.mode = "Normal Chat"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="ash-logo">',
    unsafe_allow_html=True
)

try:

    st.image(
        "logo.png",
        width=110
    )

except Exception:

    st.markdown(
        "<h1 style='text-align:center;'>ASH</h1>",
        unsafe_allow_html=True
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    '<div class="ash-name">ASH</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="ash-subtitle">Your all-in-one AI assistant</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        # Show generated image if available
        if message.get("image"):

            st.image(
                message["image"],
                use_container_width=True
            )


# ============================================================
# AI TEXT FUNCTION
# ============================================================

def ask_ai(prompt):

    pdf_context = ""

    if st.session_state.pdf_text:

        pdf_text = st.session_state.pdf_text

        # Limit context for faster requests
        pdf_text = pdf_text[:10000]

        pdf_context = f"""

The user uploaded a PDF.

Use the PDF as the main source when answering.

PDF CONTENT:

-------------------------

{pdf_text}

-------------------------
"""


    mode_instruction = {

        "Normal Chat":
            """
            Answer the user's question naturally.
            Give useful and accurate information.
            """,

        "Explain Topic":
            """
            Explain the topic in extremely simple language.
            Assume the user is a beginner.
            Use examples and step-by-step explanations.
            """,

        "Make Notes":
            """
            Turn the answer into short study notes.
            Use headings, bullet points and definitions.
            """,

        "Generate MCQs":
            """
            Generate 10 important MCQs.
            Each question must have A, B, C and D.
            Clearly show the correct answer.
            """,

        "Exam Questions":
            """
            Create important university exam questions.
            Include short and long questions.
            """,

        "Summarize":
            """
            Summarize the provided information.
            Keep only the most important points.
            """
    }


    system_prompt = f"""
You are ASH, an intelligent all-in-one AI assistant.

{mode_instruction.get(
    st.session_state.mode,
    "Answer clearly and accurately."
)}

IMPORTANT RULES:

- Be accurate.
- Use simple language.
- Do not unnecessarily repeat the question.
- Give direct answers.
- Use examples when useful.
- For study questions, make information easy to memorize.
- If a PDF is provided, prioritize its information.

{pdf_context}
"""


    messages = [

        {
            "role": "system",
            "content": system_prompt
        },

        {
            "role": "user",
            "content": prompt
        }

    ]


    response = hf_client.chat.completions.create(

        model=TEXT_MODEL,

        messages=messages,

        max_tokens=700,

        temperature=0.4
    )


    return response.choices[0].message.content


# ============================================================
# IMAGE UNDERSTANDING
# ============================================================

def analyze_image(image_bytes, prompt):

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")


    image_message = [

        {
            "type": "text",
            "text": prompt
        },

        {
            "type": "image_url",
            "image_url": {
                "url": (
                    "data:image/jpeg;base64,"
                    + encoded
                )
            }
        }

    ]


    response = hf_client.chat.completions.create(

        model=VISION_MODEL,

        messages=[
            {
                "role": "user",
                "content": image_message
            }
        ],

        max_tokens=700
    )


    return response.choices[0].message.content


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(prompt):

    image = hf_client.text_to_image(

        prompt=prompt,

        model=IMAGE_MODEL
    )

    return image


# ============================================================
# PLUS MENU
# ============================================================

with st.popover(
    "➕",
    use_container_width=True
):

    st.markdown(
        "### ASH Tools"
    )


    # --------------------------------------------------------
    # STUDY MODE
    # --------------------------------------------------------

    mode = st.selectbox(

        "Study Mode",

        [
            "Normal Chat",
            "Explain Topic",
            "Make Notes",
            "Generate MCQs",
            "Exam Questions",
            "Summarize"
        ],

        index=[
            "Normal Chat",
            "Explain Topic",
            "Make Notes",
            "Generate MCQs",
            "Exam Questions",
            "Summarize"
        ].index(
            st.session_state.mode
        )
    )

    st.session_state.mode = mode


    st.divider()


    # --------------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------------

    st.markdown(
        "### 📄 Upload PDF"
    )

    pdf_file = st.file_uploader(

        "Choose PDF",

        type=["pdf"],

        label_visibility="collapsed"
    )


    if pdf_file:

        try:

            reader = PdfReader(
                pdf_file
            )

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:

                    text += page_text + "\n"


            st.session_state.pdf_text = text

            st.session_state.pdf_name = (
                pdf_file.name
            )


            st.success(
                f"📄 {pdf_file.name} loaded"
            )

            st.caption(
                f"{len(reader.pages)} pages"
            )


        except Exception as e:

            st.error(
                "Could not read PDF."
            )

            st.code(
                str(e)
            )


    if st.session_state.pdf_name:

        st.info(
            "Current PDF: "
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


    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    st.markdown(
        "### 🖼️ Upload Image"
    )


    uploaded_image = st.file_uploader(

        "Choose image",

        type=[
            "png",
            "jpg",
            "jpeg",
            "webp"
        ],

        label_visibility="collapsed"
    )


    if uploaded_image:

        image_bytes = uploaded_image.getvalue()

        st.session_state.uploaded_image = (
            image_bytes
        )

        st.image(
            image_bytes,
            caption="Uploaded image",
            use_container_width=True
        )

        st.success(
            "Image ready for AI analysis."
        )


# ============================================================
# IMAGE STATUS ABOVE INPUT
# ============================================================

if st.session_state.uploaded_image:

    st.caption(
        "🖼️ Image attached — ask ASH about it."
    )


if st.session_state.pdf_name:

    st.caption(
        "📄 PDF attached — ask ASH about it."
    )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Message ASH..."
)


# ============================================================
# PROCESS PROMPT
# ============================================================

if prompt:

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


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            # =================================================
            # IMAGE ANALYSIS
            # =================================================

            if st.session_state.uploaded_image:

                with st.spinner(
                    "🔎 Looking at the image..."
                ):

                    answer = analyze_image(

                        st.session_state.uploaded_image,

                        prompt
                    )


            # =================================================
            # IMAGE GENERATION
            # =================================================

            elif (
                prompt.lower().startswith(
                    "generate image"
                )
                or
                prompt.lower().startswith(
                    "create image"
                )
                or
                prompt.lower().startswith(
                    "make an image"
                )
            ):

                image_prompt = prompt

                image_prompt = (
                    image_prompt
                    .replace(
                        "generate image",
                        ""
                    )
                    .replace(
                        "create image",
                        ""
                    )
                    .replace(
                        "make an image",
                        ""
                    )
                    .strip()
                )


                with st.spinner(
                    "🎨 Creating your image..."
                ):

                    generated_image = (
                        generate_image(
                            image_prompt
                        )
                    )


                st.image(
                    generated_image,
                    caption="Generated by ASH",
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

                    file_name="ash_generated.png",

                    mime="image/png"
                )


                # Save text description
                answer = (
                    "I've generated the image "
                    "for you above."
                )


                st.session_state.messages.append(

                    {
                        "role": "assistant",

                        "content": answer,

                        "image": generated_image
                    }

                )


                # Clear uploaded image
                st.session_state.uploaded_image = None

                st.stop()


            # =================================================
            # NORMAL AI
            # =================================================

            else:

                with st.spinner(
                    "ASH is thinking..."
                ):

                    answer = ask_ai(
                        prompt
                    )


            # ------------------------------------------------
            # SHOW ANSWER
            # ------------------------------------------------

            st.markdown(
                answer
            )


            # ------------------------------------------------
            # SAVE ANSWER
            # ------------------------------------------------

            st.session_state.messages.append(

                {
                    "role": "assistant",
                    "content": answer
                }

            )


            # ------------------------------------------------
            # CLEAR IMAGE AFTER USE
            # ------------------------------------------------

            if st.session_state.uploaded_image:

                st.session_state.uploaded_image = None


        except Exception as e:

            error_text = str(e)

            if "429" in error_text:

                st.error(
                    "⚠️ AI service is temporarily out of "
                    "free inference credits. Please try "
                    "again later or switch to another "
                    "Hugging Face model/provider."
                )

            elif "503" in error_text:

                st.error(
                    "⚠️ The selected AI provider is "
                    "temporarily busy. Please try again."
                )

            else:

                st.error(
                    "❌ ASH could not process your request."
                )

                st.code(
                    error_text
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#999;
        font-size:12px;
        margin-top:30px;
    ">
        ASH • Your all-in-one AI assistant
    </div>
    """,
    unsafe_allow_html=True
)
