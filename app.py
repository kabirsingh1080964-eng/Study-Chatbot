import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from huggingface_hub import InferenceClient
from io import BytesIO
import wave


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

if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = False

if "last_voice_audio" not in st.session_state:
    st.session_state.last_voice_audio = None


# ============================================================
# HELPER: PCM -> WAV
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
# HELPER: GEMINI TEXT RESPONSE
# ============================================================

def get_ai_answer(prompt):

    mode = st.session_state.selected_mode

    instructions = {

        "💬 Normal Chat":
            """
            Answer the student's question clearly
            and accurately.
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

            Each question must have:

            A
            B
            C
            D

            Clearly show the correct answer.
            """,

        "🎯 Exam Questions":
            """
            Create important university exam questions.

            Include:
            - Short questions
            - Long questions
            - Important concepts
            """
    }

    pdf_context = ""

    if st.session_state.pdf_text:

        pdf_text = st.session_state.pdf_text

        # Keep request reasonably small
        if len(pdf_text) > 6000:
            pdf_text = pdf_text[:6000]

        pdf_context = f"""

The student uploaded a PDF.

Use the PDF as the main source.

If the answer cannot be found in the PDF,
say:

"I couldn't find this information in your uploaded PDF."

PDF:

-------------------------
{pdf_text}
-------------------------
"""

    system_prompt = f"""
You are ASH Study Assistant.

{instructions.get(
    mode,
    "Answer the question clearly and accurately."
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

    response = client.models.generate_content(
        model="gemini-3.7-flash",
        contents=[
            system_prompt,
            f"Student Question:\n{prompt}"
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=700
        )
    )

    if response.text:
        return response.text.strip()

    return "Sorry, I couldn't generate an answer."


# ============================================================
# HELPER: GEMINI VOICE RESPONSE
# ============================================================

def generate_voice(answer):

    response = client.models.generate_content(

        model="gemini-3.1-flash-tts-preview",

        contents=[
            (
                "Speak the following answer naturally, "
                "clearly and conversationally. "
                "Do not add extra information.\n\n"
                + answer
            )
        ],

        config=types.GenerateContentConfig(

            response_modalities=[
                "AUDIO"
            ],

            speech_config=types.SpeechConfig(

                voice_config=types.VoiceConfig(

                    prebuilt_voice_config=(
                        types.PrebuiltVoiceConfig(
                            voice_name="Kore"
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
                    hasattr(part, "inline_data")
                    and part.inline_data
                ):

                    audio_bytes = (
                        part.inline_data.data
                    )

                    break

            if audio_bytes:
                break

    return audio_bytes


# ============================================================
# HEADER
# ============================================================

st.image(
    "logo.png",
    width=130
)

st.title("ASH Study Assistant")

st.caption(
    "Your AI-powered university study assistant."
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# BOTTOM BAR
# ============================================================

chat_col, voice_col, plus_col = st.columns(
    [7.5, 1, 1],
    vertical_alignment="bottom"
)


# ============================================================
# CHAT INPUT
# ============================================================

with chat_col:

    typed_prompt = st.chat_input(
        "Ask anything about your studies..."
    )


# ============================================================
# VOICE BUTTON
# ============================================================

with voice_col:

    voice_clicked = st.button(
        "🎙️",
        help="Talk to ASH Study Assistant",
        use_container_width=True
    )


# ============================================================
# PLUS MENU
# ============================================================

with plus_col:

    with st.popover(
        "➕",
        use_container_width=True
    ):

        st.subheader("Study Tools")

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

        st.session_state.selected_mode = selected_mode

        st.divider()

        st.write("📄 Upload Study Material")

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

                st.code(str(e))

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
# VOICE MODE
# ============================================================

if voice_clicked:

    st.session_state.voice_mode = True


# ============================================================
# TWO-WAY VOICE INTERFACE
# ============================================================

if st.session_state.voice_mode:

    st.divider()

    st.subheader(
        "🎙️ ASH Voice Assistant"
    )

    st.caption(
        "Speak to ASH and it will answer you with voice."
    )

    st.info(
        "🎤 Record your question below."
    )

    voice_recording = st.audio_input(
        "🎤 Tap and speak",
        key="voice_conversation_input"
    )

    # --------------------------------------------
    # PROCESS VOICE
    # --------------------------------------------

    if voice_recording is not None:

        # Prevent processing exact same recording
        recording_id = str(
            voice_recording.size
        ) + str(
            voice_recording.type
        )

        if (
            recording_id
            != st.session_state.last_voice_audio
        ):

            st.session_state.last_voice_audio = (
                recording_id
            )

            with st.spinner(
                "🎧 Listening..."
            ):

                try:

                    audio_file = (
                        client.files.upload(
                            file=voice_recording,
                            config={
                                "mime_type":
                                voice_recording.type
                                or "audio/wav"
                            }
                        )
                    )

                    transcription = (
                        client.models.generate_content(

                            model="gemini-3.7-flash",

                            contents=[
                                audio_file,

                                """
                                Listen carefully.

                                Convert the student's speech
                                into text.

                                Return ONLY what the student said.
                                Do not answer.
                                """
                            ],

                            config=(
                                types.GenerateContentConfig(
                                    max_output_tokens=200
                                )
                            )
                        )
                    )

                    voice_prompt = (
                        transcription.text.strip()
                    )

                    # ----------------------------------------
                    # SHOW USER MESSAGE
                    # ----------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": voice_prompt
                        }
                    )

                    with st.chat_message("user"):

                        st.markdown(
                            voice_prompt
                        )

                    # ----------------------------------------
                    # GET AI ANSWER
                    # ----------------------------------------

                    with st.chat_message("assistant"):

                        with st.spinner(
                            "🤖 Thinking..."
                        ):

                            answer = get_ai_answer(
                                voice_prompt
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

                    # ----------------------------------------
                    # GENERATE VOICE
                    # ----------------------------------------

                    with st.spinner(
                        "🔊 Speaking..."
                    ):

                        audio_bytes = generate_voice(
                            answer
                        )

                    if audio_bytes:

                        wav_audio = pcm_to_wav(
                            audio_bytes
                        )

                        st.audio(
                            wav_audio,
                            format="audio/wav",
                            autoplay=True
                        )

                    else:

                        st.warning(
                            "AI answered, but voice "
                            "audio was not returned."
                        )

                except Exception as e:

                    st.error(
                        "❌ Voice conversation failed."
                    )

                    st.code(
                        str(e)
                    )

    st.divider()

    if st.button(
        "❌ Close Voice Assistant",
        use_container_width=True
    ):

        st.session_state.voice_mode = False
        st.session_state.last_voice_audio = None

        st.rerun()


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

    with st.chat_message("user"):

        st.markdown(prompt)

    mode = st.session_state.selected_mode

    # ========================================================
    # IMAGE GENERATION
    # ========================================================

    if mode == "🎨 Generate Image":

        with st.chat_message("assistant"):

            with st.spinner(
                "🎨 Creating image..."
            ):

                try:

                    image = image_client.text_to_image(
                        prompt=prompt,
                        model=(
                            "black-forest-labs/"
                            "FLUX.1-schnell"
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
                        data=image_buffer.getvalue(),
                        file_name=(
                            "ash_study_image.png"
                        ),
                        mime="image/png"
                    )

                except Exception as e:

                    st.error(
                        "❌ Image generation failed."
                    )

                    st.code(str(e))

    # ========================================================
    # NORMAL TEXT RESPONSE
    # ========================================================

    else:

        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 Thinking..."
            ):

                try:

                    answer = get_ai_answer(
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

                except Exception as e:

                    st.error(
                        "❌ Something went wrong."
                    )

                    st.code(str(e))
