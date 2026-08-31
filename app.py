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


# ============================================================
# HELPER - PCM TO WAV
# ============================================================

def pcm_to_wav(pcm_data):

    wav_buffer = BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)

        wav_file.writeframes(pcm_data)

    wav_buffer.seek(0)

    return wav_buffer.getvalue()


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
# BOTTOM CHAT BAR
#
# Layout:
#
#        [ Chat Input                  ] [🎤] [➕]
#
# ============================================================

chat_col, voice_col, plus_col = st.columns(
    [8, 0.8, 0.8],
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

    audio_value = st.audio_input(
        "🎤",
        sample_rate=16000,
        label_visibility="collapsed",
        key="voice_input"
    )


# ============================================================
# PLUS BUTTON
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
# FUNCTION - GET PDF CONTEXT
# ============================================================

def get_pdf_context():

    if not st.session_state.pdf_text:

        return ""


    pdf_text = (
        st.session_state.pdf_text
    )


    # Keep prompt smaller and faster

    if len(pdf_text) > 6000:

        pdf_text = pdf_text[:6000]


    return f"""

The student uploaded university study material.

Use this material as the main source.

If the answer cannot be found in the
provided material, clearly say:

"I couldn't find this information
in your uploaded PDF."

PDF CONTENT:

-------------------------
{pdf_text}
-------------------------

"""


# ============================================================
# FUNCTION - GENERATE AI TEXT
# ============================================================

def generate_answer(prompt):

    mode = st.session_state.selected_mode


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

            Clearly identify the correct answer.
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

    return "I couldn't generate an answer."


# ============================================================
# FUNCTION - GENERATE VOICE
# ============================================================

def generate_voice(answer):

    tts_response = client.models.generate_content(

        model="gemini-3.1-flash-tts-preview",

        contents=(
            "Speak this answer naturally, "
            "clearly and conversationally. "
            "Do not add information.\n\n"
            + answer
        ),

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


    if tts_response.candidates:

        for candidate in (
            tts_response.candidates
        ):

            if not candidate.content:

                continue


            for part in (
                candidate.content.parts
            ):

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

        return pcm_to_wav(
            audio_bytes
        )


    return None


# ============================================================
# FUNCTION - TRANSCRIBE VOICE
# ============================================================

def transcribe_voice(audio_value):

    audio_file = client.files.upload(

        file=audio_value,

        config={
            "mime_type": "audio/wav"
        }
    )


    response = client.models.generate_content(

        model="gemini-3.7-flash",

        contents=[

            audio_file,

            """
            Listen to the student's voice.

            Convert their speech into text.

            Return ONLY what the student said.

            Do not answer the question.
            """
        ],

        config=types.GenerateContentConfig(

            max_output_tokens=200
        )
    )


    if response.text:

        return response.text.strip()


    return None


# ============================================================
# PROCESS VOICE
# ============================================================

voice_prompt = None


if audio_value is not None:

    with st.spinner(
        "🎧 Listening..."
    ):

        try:

            voice_prompt = (
                transcribe_voice(
                    audio_value
                )
            )


        except Exception as e:

            st.error(
                "❌ Could not understand your voice."
            )

            st.code(
                str(e)
            )


# ============================================================
# CHOOSE PROMPT
# ============================================================

prompt = typed_prompt


if voice_prompt:

    prompt = voice_prompt


# ============================================================
# PROCESS QUESTION
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


    with st.chat_message("user"):

        st.markdown(
            prompt
        )


    # ========================================================
    # IMAGE MODE
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
    # TEXT / VOICE AI
    # ========================================================

    else:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🤖 Thinking..."
            ):

                try:

                    answer = generate_answer(
                        prompt
                    )


                    # ----------------------------------------
                    # SHOW TEXT
                    # ----------------------------------------

                    st.markdown(
                        answer
                    )


                    # ----------------------------------------
                    # SAVE TEXT
                    # ----------------------------------------

                    st.session_state.messages.append(

                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


                    # ====================================================
                    # TWO-WAY VOICE
                    #
                    # Whenever the user speaks:
                    #
                    # 🎤 User speaks
                    #       ↓
                    # Speech → Text
                    #       ↓
                    # Gemini thinks
                    #       ↓
                    # Text answer
                    #       ↓
                    # Text → Voice
                    #       ↓
                    # 🔊 AI speaks
                    #
                    # Then user can press 🎤 again.
                    # ====================================================

                    if voice_prompt:

                        with st.spinner(
                            "🔊 AI is speaking..."
                        ):

                            try:

                                voice_reply = (
                                    generate_voice(
                                        answer
                                    )
                                )


                                if voice_reply:

                                    st.audio(
                                        voice_reply,

                                        format=(
                                            "audio/wav"
                                        ),

                                        autoplay=True
                                    )


                                else:

                                    st.warning(
                                        "⚠️ AI generated "
                                        "the text answer but "
                                        "no audio was returned."
                                    )


                            except Exception as voice_error:

                                st.warning(
                                    "⚠️ Text answer generated, "
                                    "but voice reply failed."
                                )

                                st.code(
                                    str(voice_error)
                                )


                except Exception as e:

                    st.error(
                        "❌ Something went wrong."
                    )

                    st.code(
                        str(e)
                    )
