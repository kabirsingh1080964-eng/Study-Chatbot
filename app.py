import streamlit as st
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from io import BytesIO


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="ASH",
    page_icon="logo.png",
    layout="centered"
)


# ============================================================
# BLACK / CHATGPT STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    .main {
        background-color: #000000 !important;
    }

    .block-container {
        max-width: 900px;
        padding-top: 20px;
        padding-bottom: 120px;
        background-color: #000000 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    p, label {
        color: #ffffff !important;
    }

    [data-testid="stChatMessage"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }

    [data-testid="stChatInput"] {
        background-color: #1f1f1f !important;
        border: 1px solid #444444 !important;
        border-radius: 22px !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #1f1f1f !important;
        color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #999999 !important;
    }

    button {
        background-color: #1f1f1f !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
        border-radius: 10px !important;
    }

    button:hover {
        background-color: #333333 !important;
    }

    [data-testid="stPopover"] {
        background-color: #111111 !important;
    }

    [data-testid="stFileUploader"] {
        background-color: #111111 !important;
        border-radius: 10px !important;
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
        st.error("HF_TOKEN is missing from Streamlit Secrets.")
        st.stop()

    return InferenceClient(
        token=token
    )


client = get_client()


# ============================================================
# MODELS
# ============================================================

TEXT_MODEL = "Qwen/Qwen3-32B"

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

if "mode" not in st.session_state:
    st.session_state.mode = "Normal Chat"


# ============================================================
# HEADER
# ============================================================

try:

    st.image(
        "logo.png",
        width=120
    )

except Exception:

    st.markdown(
        "<h1 style='text-align:center;'>ASH</h1>",
        unsafe_allow_html=True
    )


st.markdown(
    """
    <h1 style="
        text-align:center;
        margin-top:-10px;
        margin-bottom:0px;
    ">
        ASH
    </h1>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <p style="
        text-align:center;
        color:#999999 !important;
        margin-top:0px;
        margin-bottom:35px;
    ">
        Your all-in-one AI assistant
    </p>
    """,
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


    # --------------------------------------------------------
    # STUDY MODE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------------

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
                f"✅ {pdf_file.name} loaded"
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


    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

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


        st.image(
            st.session_state.image_bytes,
            use_container_width=True
        )


        st.success(
            "Image attached."
        )


# ============================================================
# ATTACHMENT STATUS
# ============================================================

if st.session_state.pdf_name:

    st.caption(
        "📄 PDF attached: "
        + st.session_state.pdf_name
    )


if st.session_state.image_bytes:

    st.caption(
        "🖼️ Image attached"
    )


# ============================================================
# AI RESPONSE
# ============================================================

def ask_ai(prompt):

    pdf_context = ""

    if st.session_state.pdf_text:

        pdf_context = (
            "\n\nUploaded PDF:\n"
            + st.session_state.pdf_text[:8000]
        )


    instructions = {

        "Normal Chat":
            "Answer clearly and accurately.",

        "Explain Topic":
            (
                "Explain the topic in extremely simple "
                "language. Use examples and steps."
            ),

        "Make Notes":
            (
                "Create short study notes with headings "
                "and bullet points."
            ),

        "Generate MCQs":
            (
                "Create 10 MCQs with options A, B, C and D. "
                "Show the correct answer."
            ),

        "Exam Questions":
            (
                "Create important university exam questions "
                "including short and long questions."
            )
    }


    system_prompt = f"""
You are ASH, an all-in-one AI assistant.

{instructions[st.session_state.mode]}

Rules:

- Give direct answers.
- Use simple language.
- Be accurate.
- Avoid unnecessary repetition.
- Help students understand difficult concepts.
- Give examples when useful.

{pdf_context}
"""


    response = client.chat.completions.create(

        model=TEXT_MODEL,

        messages=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        max_tokens=700
    )


    return response.choices[0].message.content


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
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Message ASH..."
)


# ============================================================
# PROCESS MESSAGE
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
            # IMAGE GENERATION
            # =================================================

            image_request = (

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

                or

                prompt.lower().startswith(
                    "draw "
                )
            )


            if image_request:

                image_prompt = prompt

                prefixes = [

                    "generate image",

                    "create image",

                    "make an image",

                    "draw"
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
            # IMAGE ATTACHED
            # =================================================

            elif st.session_state.image_bytes:

                st.warning(
                    "Image analysis requires a vision model. "
                    "You can keep the image attached and "
                    "connect a vision provider later."
                )


                answer = ask_ai(
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


                st.session_state.image_bytes = None


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


                st.markdown(
                    answer
                )


                st.session_state.messages.append(

                    {
                        "role": "assistant",
                        "content": answer
                    }

                )


        except Exception as error:

            error_message = str(error)


            if "429" in error_message:

                st.error(
                    "⚠️ Free AI inference limit reached. "
                    "Please try again later."
                )


            elif "503" in error_message:

                st.error(
                    "⚠️ The AI model is temporarily busy. "
                    "Please try again in a moment."
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
        color:#777777;
        font-size:12px;
        margin-top:40px;
    ">
        ASH • Your all-in-one AI assistant
    </div>
    """,
    unsafe_allow_html=True
)
