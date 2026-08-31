import streamlit as st
from google import genai
from google.genai import types
from io import BytesIO
import base64
import time


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
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_client():

    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )


client = get_client()


# ============================================================
# MODELS
# ============================================================

# Fast model for normal conversation, studying,
# PDFs and image understanding.
FAST_MODEL = "gemini-2.5-flash-lite"

# More capable fallback.
SMART_MODEL = "gemini-3.7-flash"

# Native image generation model.
IMAGE_MODEL = "gemini-3.1-flash-image"


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

if "image_generation" not in st.session_state:
    st.session_state.image_generation = False


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Remove Streamlit top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 7rem;
        max-width: 850px;
    }

    /* Hide menu */
    #MainMenu {
        visibility: hidden;
    }

    /* Hide footer */
    footer {
        visibility: hidden;
    }

    /* Header */
    .ash-header {
        text-align: center;
        margin-top: 30px;
        margin-bottom: 35px;
    }

    .ash-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .ash-subtitle {
        font-size: 16px;
        opacity: 0.65;
    }

    /* Upload information */
    .upload-box {
        padding: 12px;
        border-radius: 12px;
        margin-bottom: 10px;
        background: rgba(128,128,128,0.08);
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
# HEADER
# ============================================================

st.markdown(
    """
    <div class="ash-header">

        <div class="ash-title">
            ASH
        </div>

        <div class="ash-subtitle">
            Your all-in-one AI assistant
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        if message.get("type") == "image":

            st.image(
                message["data"],
                use_container_width=True
            )

        else:

            st.markdown(
                message["content"]
            )


# ============================================================
# PLUS MENU
# ============================================================

# Put the uploader ABOVE the chat input so it doesn't
# interfere with Streamlit's chat_input positioning.

with st.popover(
    "＋",
    use_container_width=False
):

    st.markdown(
        "### Add to ASH"
    )

    st.caption(
        "Upload an image or PDF."
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
            "pdf"
        ],
        label_visibility="collapsed"
    )


    # --------------------------------------------------------
    # FILE HANDLING
    # --------------------------------------------------------

    if uploaded_file is not None:

        file_name = uploaded_file.name

        file_bytes = uploaded_file.getvalue()

        # Prevent duplicate upload
        if file_name not in st.session_state.uploaded_file_names:

            mime_type = uploaded_file.type

            st.session_state.uploaded_files.append(
                {
                    "name": file_name,
                    "bytes": file_bytes,
                    "mime_type": mime_type
                }
            )

            st.session_state.uploaded_file_names.append(
                file_name
            )

            st.success(
                f"✅ {file_name} attached"
            )


    # --------------------------------------------------------
    # SHOW CURRENT FILES
    # --------------------------------------------------------

    if st.session_state.uploaded_files:

        st.markdown(
            "#### Attached files"
        )

        for index, file_info in enumerate(
            st.session_state.uploaded_files
        ):

            col1, col2 = st.columns(
                [5, 1]
            )

            with col1:

                st.caption(
                    "📎 "
                    + file_info["name"]
                )

            with col2:

                if st.button(
                    "×",
                    key=f"remove_file_{index}"
                ):

                    removed_name = (
                        st.session_state
                        .uploaded_files[index]
                        ["name"]
                    )

                    st.session_state.uploaded_files.pop(
                        index
                    )

                    st.session_state.uploaded_file_names.remove(
                        removed_name
                    )

                    st.rerun()


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask anything..."
)


# ============================================================
# IMAGE GENERATION DETECTION
# ============================================================

def is_image_request(text):

    text = text.lower()

    image_words = [

        "generate image",
        "create image",
        "make image",
        "generate a picture",
        "create a picture",
        "make a picture",
        "draw",
        "illustration",
        "illustrate",
        "design an image",
        "generate photo",
        "create photo",
        "make photo",
        "ai image",
        "create artwork",
        "generate artwork"

    ]

    return any(
        word in text
        for word in image_words
    )


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(prompt, files):

    contents = []

    # --------------------------------------------------------
    # TEXT PROMPT
    # --------------------------------------------------------

    contents.append(
        prompt
    )


    # --------------------------------------------------------
    # OPTIONAL IMAGE REFERENCES
    # --------------------------------------------------------

    for file_info in files:

        mime_type = file_info["mime_type"]

        if mime_type.startswith("image/"):

            contents.append(
                types.Part.from_bytes(
                    data=file_info["bytes"],
                    mime_type=mime_type
                )
            )


    # --------------------------------------------------------
    # IMAGE GENERATION
    # --------------------------------------------------------

    response = client.models.generate_content(

        model=IMAGE_MODEL,

        contents=contents,

        config=types.GenerateContentConfig(

            response_modalities=[
                "IMAGE"
            ],

            response_format={
                "image": {
                    "aspect_ratio": "16:9",
                    "image_size": "2K"
                }
            }
        )
    )


    # --------------------------------------------------------
    # FIND GENERATED IMAGE
    # --------------------------------------------------------

    if response.candidates:

        for candidate in response.candidates:

            if not candidate.content:
                continue

            for part in candidate.content.parts:

                if (
                    hasattr(
                        part,
                        "inline_data"
                    )
                    and part.inline_data
                ):

                    return (
                        part.inline_data.data
                    )


    return None


# ============================================================
# NORMAL AI RESPONSE
# ============================================================

def generate_answer(prompt, files):

    # --------------------------------------------------------
    # SYSTEM INSTRUCTIONS
    # --------------------------------------------------------

    system_instruction = """
You are ASH, an advanced all-in-one AI assistant.

Your job is to help the user with ANY reasonable request.

You can help with:

- General knowledge
- Current information when available
- University studies
- Programming
- Mathematics
- Software engineering
- Writing
- Summaries
- Explanations
- Research
- PDFs
- Images
- Problem solving
- Brainstorming

IMPORTANT:

Answer directly.

Do not unnecessarily say:
"As an AI..."

Use simple language unless the user asks for technical detail.

For study questions:
- Explain step by step.
- Give examples.
- Make difficult concepts easy.
- Make answers useful for exams.

For uploaded PDFs:
- Carefully use the uploaded document.
- If the answer is not available in the document,
  clearly tell the user.

For uploaded images:
- Analyze the image carefully.
- Answer the user's question about the image.

Do not invent information from an uploaded file.

Keep normal answers reasonably concise.
"""


    # --------------------------------------------------------
    # BUILD CONTENTS
    # --------------------------------------------------------

    contents = []


    contents.append(
        system_instruction
    )


    # --------------------------------------------------------
    # ADD FILES
    # --------------------------------------------------------

    for file_info in files:

        contents.append(

            types.Part.from_bytes(

                data=file_info["bytes"],

                mime_type=file_info["mime_type"]
            )
        )


    # --------------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------------

    contents.append(
        "\n\nUSER QUESTION:\n"
        + prompt
    )


    # --------------------------------------------------------
    # FAST STREAMING RESPONSE
    # --------------------------------------------------------

    response_stream = client.models.generate_content_stream(

        model=FAST_MODEL,

        contents=contents,

        config=types.GenerateContentConfig(

            max_output_tokens=1000,

            temperature=0.3
        )
    )


    return response_stream


# ============================================================
# PROCESS USER PROMPT
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


    # --------------------------------------------------------
    # DISPLAY USER
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            prompt
        )


    # ========================================================
    # IMAGE GENERATION
    # ========================================================

    if is_image_request(prompt):

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Creating your image..."
            ):

                try:

                    image_bytes = generate_image(

                        prompt,

                        st.session_state.uploaded_files
                    )


                    if image_bytes:

                        st.image(

                            image_bytes,

                            caption="Generated by ASH",

                            use_container_width=True
                        )


                        st.download_button(

                            label="⬇️ Download Image",

                            data=image_bytes,

                            file_name="ash_generated_image.png",

                            mime="image/png"
                        )


                        st.session_state.messages.append(

                            {
                                "role": "assistant",
                                "type": "image",
                                "data": image_bytes
                            }
                        )


                    else:

                        st.error(
                            "ASH couldn't generate the image."
                        )


                except Exception as e:

                    st.error(
                        "Image generation failed."
                    )

                    st.code(
                        str(e)
                    )


    # ========================================================
    # NORMAL AI ANSWER
    # ========================================================

    else:

        with st.chat_message(
            "assistant"
        ):

            response_placeholder = st.empty()

            full_answer = ""

            try:

                stream = generate_answer(

                    prompt,

                    st.session_state.uploaded_files
                )


                # ------------------------------------------------
                # STREAM RESPONSE
                # ------------------------------------------------

                for chunk in stream:

                    if chunk.text:

                        full_answer += chunk.text

                        response_placeholder.markdown(
                            full_answer
                            + "▌"
                        )


                response_placeholder.markdown(
                    full_answer
                )


                # ------------------------------------------------
                # SAVE RESPONSE
                # ------------------------------------------------

                st.session_state.messages.append(

                    {
                        "role": "assistant",
                        "content": full_answer
                    }
                )


            except Exception as e:

                st.error(
                    "ASH couldn't process your request."
                )

                st.code(
                    str(e)
                )


    # ========================================================
    # CLEAR ATTACHMENTS AFTER MESSAGE
    # ========================================================

    st.session_state.uploaded_files = []

    st.session_state.uploaded_file_names = []

    st.rerun()
