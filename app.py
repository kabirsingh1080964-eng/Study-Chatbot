import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from huggingface_hub import InferenceClient
from io import BytesIO
import wave
import time


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="ASH Study Assistant",
    page_icon="logo.png",
    layout="centered"
)


# ============================================================
# CUSTOM DARK THEME STYLING
# ============================================================

st.markdown("""
    <style>
    /* Global Background and Text Color */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Sidebar styling if expanded */
    section[data-testid="stSidebar"] {
        background-color: #121212;
        color: #ffffff;
    }

    /* Input fields styling */
    .stTextInput input, .stTextArea textarea {
        background-color: #121212;
        color: #ffffff;
        border: 1px solid #262626;
    }

    /* Chat input box container */
    div[data-testid="stChatInput"] {
        background-color: #121212 !important;
        border: 1px solid #262626 !important;
        border-radius: 9999px !important;
    }

    div[data-testid="stChatInput"] textarea {
        color: #ffffff !important;
    }

    /* Buttons styling */
    .stButton button {
        background-color: #121212;
        color: #ffffff;
        border: 1px solid #262626;
        border-radius: 9999px;
    }
    
    .stButton button:hover {
        background-color: #262626;
        border-color: #404040;
    }

    /* Headers & Text */
    h1, h2, h3, h4, h5, h6, p, span {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)


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
# MODELS
# ============================================================

TEXT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.7-flash"
]

TTS_MODELS = [
    "gemini-2.5-flash-preview-tts",
    "gemini-3.1-flash-tts-preview"
]


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

if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = False

if "last_voice_id" not in st.session_state:
    st.session_state.last_voice_id = ""

if "voice_history" not in st.session_state:
    st.session_state.voice_history = []


# ============================================================
# HELPER: PCM TO WAV
# ============================================================

def pcm_to_wav(pcm_data):

    buffer = BytesIO()

    with wave.open(buffer, "wb") as wav_file:

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)

        wav_file.writeframes(pcm_data)

    return buffer.getvalue()


# ============================================================
# HELPER: GEMINI TEXT RESPONSE WITH RETRY
# ============================================================

def generate_text(contents, max_output_tokens=600):

    last_error = None

    for model_name in TEXT_MODELS:

        for attempt in range(2):

            try:

                response = client.models.generate_content(

                    model=model_name,

                    contents=contents,

                    config=types.GenerateContentConfig(
                        max_output_tokens=max_output_tokens,
                        temperature=0.4
                    )
                )

                if response.text:

                    return response.text.strip()

            except Exception as e:

                last_error = e

                error_text = str(e).lower()

                if (
                    "503" in error_text
                    or "unavailable" in error_text
                    or "high demand" in error_text
                    or "429" in error_text
                    or "resource exhausted" in error_text
                ):

                    time.sleep(1)

                    continue

                break

    raise Exception(
        f"All Gemini text models failed.\n{last_error}"
    )


# ============================================================
# HELPER: TRANSCRIBE VOICE
# ============================================================

def transcribe_voice(audio_data):

    mime_type = (
        audio_data.type
        if audio_data.type
        else "audio/wav"
    )

    audio_file = client.files.upload(

        file=audio_data,

        config={
            "mime_type": mime_type
        }
    )

    prompt = """
Listen carefully to the student's recording.

Convert the student's speech into text.

Return ONLY what the student said.

Do not answer the question.

Do not explain anything.

Do not add words.
"""

    return generate_text(
        [
            audio_file,
            prompt
        ],
        max_output_tokens=250
    )


# ============================================================
# HELPER: GENERATE AI ANSWER
# ============================================================

def get_ai_answer(prompt):

    mode = st.session_state.selected_mode

    instructions = {

        "💬 Normal Chat":
            """
Answer the student's question clearly and accurately.
""",

        "📚 Explain Topic":
            """
Explain the topic in extremely simple English.

Use:
- Simple language
- Examples
- Step-by-step explanation

Assume the student is a beginner.
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
Create important university exam-style questions.

Include:
- Short questions
- Long questions
- Important concepts
"""
    }

    pdf_context = ""

    if st.session_state.pdf_text:

        pdf_text = st.session_state.pdf_text

        if len(pdf_text) > 6000:

            pdf_text = pdf_text[:6000]

        pdf_context = f"""

The student uploaded university study material.

Use the material below as the main source.

If the answer cannot be found in the provided
material, say:

"I couldn't find this information in your uploaded PDF."

PDF CONTENT:

-------------------------
{pdf_text}
-------------------------
"""

    system_prompt = f"""
You are ASH Study Assistant.

{instructions.get(
    mode,
    "Answer clearly and accurately."
)}

Rules:

- Use simple English.
- Be accurate.
- Be helpful.
- Avoid unnecessary complicated words.
- Explain difficult concepts step by step.
- Give examples when useful.
- Keep answers reasonably concise.

{pdf_context}
"""

    if st.session_state.voice_mode:

        system_prompt += """

You are currently talking to the student by voice.

Speak conversationally.

Keep your answer relatively short.

Do not use unnecessary headings.

Do not give very long lists unless the student asks.

Sound natural, friendly and helpful.
"""

    return generate_text(

        [
            system_prompt,

            f"""
Student Question:

{prompt}
"""
        ],

        max_output_tokens=500
    )


# ============================================================
# HELPER: GENERATE VOICE
# ============================================================

def generate_voice(answer):

    last_error = None

    for model_name in TTS_MODELS:

        for attempt in range(2):

            try:

                response = client.models.generate_content(

                    model=model_name,

                    contents=[
                        (
                            "Speak the following answer "
                            "naturally and conversationally. "
                            "Use a friendly educational tone. "
                            "Do not add extra information.\n\n"
                            + answer
                        )
                    ],

                    config=types.GenerateContentConfig(

                        response_modalities=[
                            "AUDIO"
                        ],

                        speech_config=types.SpeechConfig(

                            voice_config=(
                                types.VoiceConfig(

                                    prebuilt_voice_config=(
                                        types.PrebuiltVoiceConfig(
                                            voice_name="Kore"
                                        )
                                    )
                                )
                            )
                        )
                    )
                )

                audio_bytes = None

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

                                audio_bytes = (
                                    part.inline_data.data
                                )

                                break

                        if audio_bytes:
                            break

                if audio_bytes:

                    return audio_bytes

            except Exception as e:

                last_error = e

                error_text = str(e).lower()

                if (
                    "503" in error_text
                    or "unavailable" in error_text
                    or "high demand" in error_text
                    or "429" in error_text
                    or "resource exhausted" in error_text
                ):

                    time.sleep(1)

                    continue

                break

    raise Exception(
        f"TTS models failed.\n{last_error}"
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
# BOTTOM BAR
# ============================================================

chat_col, voice_col, plus_col = st.columns(
    [7.5, 1, 1],
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
# VOICE ASSISTANT BUTTON
# ============================================================

with voice_col:

    if st.session_state.voice_mode:

        voice_button = st.button(
            "🔴",
            help="Close voice assistant",
            use_container_width=True
        )

    else:

        voice_button = st.button(
            "🎙️",
            help="Open two-way voice assistant",
            use_container_width=True
        )


# ============================================================
# VOICE BUTTON ACTION
# ============================================================

if voice_button:

    st.session_state.voice_mode = (
        not st.session_state.voice_mode
    )

    st.session_state.last_voice_id = ""

    st.rerun()


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
                f"📄 {st.session_state.pdf_name} loaded"
            )

        if st.session_state.pdf_name:

            if st.button(
                "🗑️ Remove PDF",
                use_container_width=True
            ):

                st.session_state.pdf_text = ""

                st.session_state.pdf_name = ""

                st.rerun()


# ============================================================
# TWO-WAY VOICE ASSISTANT
# ============================================================

if st.session_state.voice_mode:

    st.divider()

    st.subheader(
        "🎙️ ASH Voice Assistant"
    )

    st.caption(
        "Speak → ASH understands → ASH answers → ASH speaks."
    )

    voice_recording = st.audio_input(

        "🎤 Press record and speak",

        key="ash_voice_input"
    )

    if voice_recording is not None:

        recording_id = (
            str(voice_recording.size)
            + "_"
            + str(voice_recording.type)
        )

        if (
            recording_id
            != st.session_state.last_voice_id
        ):

            st.session_state.last_voice_id = (
                recording_id
            )

            with st.spinner(
                "🎧 Understanding your voice..."
            ):

                try:

                    voice_prompt = (
                        transcribe_voice(
                            voice_recording
                        )
                    )

                except Exception as e:

                    st.error(
                        "❌ Could not understand your voice."
                    )

                    st.code(
                        str(e)
                    )

                    voice_prompt = None

            if voice_prompt:

                st.session_state.messages.append(

                    {
                        "role": "user",
                        "content": voice_prompt
                    }
                )

                with st.chat_message(
                    "user"
                ):

                    st.markdown(
                        voice_prompt
                    )

                with st.chat_message(
                    "assistant"
                ):

                    with st.spinner(
                        "🤖 Thinking..."
                    ):

                        try:

                            answer = (
                                get_ai_answer(
                                    voice_prompt
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
                                "❌ AI response failed."
                            )

                            st.code(
                                str(e)
                            )

                            answer = None

                if answer:

                    with st.spinner(
                        "🔊 ASH is speaking..."
                    ):

                        try:

                            audio_bytes = (
                                generate_voice(
                                    answer
                                )
                            )

                            if audio_bytes:

                                wav_audio = (
                                    pcm_to_wav(
                                        audio_bytes
                                    )
                                )

                                st.audio(

                                    wav_audio,

                                    format="audio/wav",

                                    autoplay=True
                                )

                            else:

                                st.warning(
                                    "ASH generated a text answer "
                                    "but no audio was returned."
                                )

                        except Exception as e:

                            st.warning(
                                "⚠️ Text answer generated, "
                                "but voice reply failed."
                            )

                            st.code(
                                str(e)
                            )

    st.divider()

    st.info(
        "🎙️ Voice mode is ON. "
        "Record your next question whenever you are ready."
    )


# ============================================================
# NORMAL TEXT CHAT
# ============================================================

if typed_prompt:

    prompt = typed_prompt

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

    mode = (
        st.session_state.selected_mode
    )

    if mode == "🎨 Generate Image":

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

    else:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🤖 Thinking..."
            ):

                try:

                    answer = (
                        get_ai_answer(
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
