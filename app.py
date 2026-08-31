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
# API CONNECTIONS
# ============================================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

image_client = InferenceClient(
    api_key=st.secrets["HF_TOKEN"]
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

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "💬 Normal Chat"

if "voice_agent_answer" not in st.session_state:
    st.session_state.voice_agent_answer = ""

if "voice_agent_audio" not in st.session_state:
    st.session_state.voice_agent_audio = None


# ============================================================
# TITLE
# ============================================================

st.image("logo.png", width=150)

st.title("ASH Study Assistant")

st.write(
    "Your AI-powered university study assistant."
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# BOTTOM TOOL AREA
# ============================================================

voice_col, chat_col, plus_col = st.columns(
    [1, 7, 1],
    vertical_alignment="bottom"
)


# ============================================================
# VOICE BUTTON - LEFT
# ============================================================

with voice_col:

    audio_value = st.audio_input(
        "🎤",
        label_visibility="collapsed"
    )


# ============================================================
# CHAT INPUT - CENTER
# ============================================================

with chat_col:

    typed_prompt = st.chat_input(
        "Ask anything about your studies..."
    )


# ============================================================
# PLUS BUTTON - RIGHT
# ============================================================

with plus_col:

    with st.popover(
        "+",
        use_container_width=True
    ):

        st.subheader("Study Tools")

        # ====================================================
        # STUDY MODES
        # ====================================================

        st.write("📚 Choose Study Mode")

        study_modes = [
            "💬 Normal Chat",
            "📚 Explain Topic",
            "📝 Make Notes",
            "❓ Generate MCQs",
            "🎯 Exam Questions",
            "🎨 Generate Image",
            "🎧 Voice Agent"
        ]

        selected_mode = st.radio(
            "Study Mode",
            study_modes,
            index=study_modes.index(
                st.session_state.selected_mode
            ),
            label_visibility="collapsed"
        )

        st.session_state.selected_mode = selected_mode

        st.divider()

        # ====================================================
        # PDF UPLOAD
        # ====================================================

        st.write("📄 Upload Study Material")

        uploaded_file = st.file_uploader(
            "Choose your PDF",
            type=["pdf"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:

            try:

                pdf_reader = PdfReader(
                    uploaded_file
                )

                pdf_text = ""

                for page in pdf_reader.pages:

                    text = page.extract_text()

                    if text:
                        pdf_text += text + "\n"

                st.session_state.pdf_text = pdf_text

                st.session_state.pdf_name = (
                    uploaded_file.name
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

                st.code(str(e))

        elif st.session_state.pdf_name:

            st.success(
                f"📄 {st.session_state.pdf_name} is loaded"
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
                file=audio_value,
                config={
                    "mime_type": "audio/wav"
                }
            )

            voice_response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    audio_file,
                    """
                    Listen to the student's spoken question.

                    Convert the spoken question into text.

                    Return ONLY the student's question.

                    Do not answer the question.
                    """
                ]
            )

            voice_prompt = (
                voice_response.text.strip()
            )

            st.info(
                "🎤 You said: "
                + voice_prompt
            )

        except Exception as e:

            st.error(
                "❌ Could not understand the recording."
            )

            st.code(str(e))


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
    # CURRENT MODE
    # ========================================================

    mode = st.session_state.selected_mode


    # ========================================================
    # STUDY MODE INSTRUCTIONS
    # ========================================================

    instructions = {

        "💬 Normal Chat":
            """
            Answer the student's question clearly
            and accurately.
            """,

        "📚 Explain Topic":
            """
            Explain the topic in extremely simple words.

            Use:
            - Simple language
            - Examples
            - Step-by-step explanations

            Assume the student is a beginner.
            """,

        "📝 Make Notes":
            """
            Convert the topic into short,
            easy-to-revise study notes.

            Use:
            - Headings
            - Bullet points
            - Important definitions
            - Examples
            """,

        "❓ Generate MCQs":
            """
            Create 10 important multiple-choice questions.

            Each question must contain:

            A
            B
            C
            D

            Clearly identify the correct answer.
            """,

        "🎯 Exam Questions":
            """
            Create important university exam-style
            questions.

            Include:
            - Short questions
            - Long questions
            - Important concepts
            """,

        "🎧 Voice Agent":
            """
            Answer naturally like a friendly AI voice assistant.

            Keep answers clear and conversational.

            Use simple English.

            Do not make the answer unnecessarily long.
            """
    }


    # ========================================================
    # PDF INSTRUCTIONS
    # ========================================================

    if st.session_state.pdf_text:

        pdf_instructions = f"""
The student has uploaded university study material.

IMPORTANT RULES:

1. Use the uploaded PDF as the MAIN source.
2. Answer the question using information from the PDF.
3. Do not invent information and claim it came from
   the PDF.
4. If the answer cannot be found in the PDF, clearly say:

"I couldn't find this information in your uploaded PDF."

UPLOADED PDF:

-------------------------
{st.session_state.pdf_text}
-------------------------
"""

    else:

        pdf_instructions = """
No PDF has been uploaded.

Use your normal knowledge to answer the question.
"""


    # ========================================================
    # IMAGE GENERATION
    # ========================================================

    if mode == "🎨 Generate Image":

        st.subheader(
            "🎨 AI Image Generator"
        )

        with st.spinner(
            "🎨 Creating your image..."
        ):

            try:

                image = image_client.text_to_image(
                    prompt=prompt,
                    model="black-forest-labs/FLUX.1-schnell"
                )

                st.image(
                    image,
                    caption=(
                        "Generated by "
                        "ASH Study Assistant"
                    ),
                    use_container_width=True
                )

                image_bytes = BytesIO()

                image.save(
                    image_bytes,
                    format="PNG"
                )

                image_bytes = (
                    image_bytes.getvalue()
                )

                st.download_button(
                    label="⬇️ Download Image",
                    data=image_bytes,
                    file_name="ash_study_image.png",
                    mime="image/png",
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    "❌ Image generation failed."
                )

                st.code(str(e))


    # ========================================================
    # NORMAL CHAT + VOICE AGENT RESPONSE
    # ========================================================

    else:

        system_prompt = f"""
You are ASH Study Assistant.

Your goal is to help university students
understand their subjects.

CURRENT STUDY MODE:

{instructions[mode]}

RULES:

- Use simple English.
- Explain difficult concepts step by step.
- Give examples whenever useful.
- Avoid unnecessary complicated terminology.
- Make exam answers easy to memorize.
- Be accurate.
- Be helpful and educational.

{pdf_instructions}
"""


        # ====================================================
        # TEXT RESPONSE
        # ====================================================

        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 Thinking..."
            ):

                try:

                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=(
                            system_prompt
                            + "\n\nStudent question:\n"
                            + prompt
                        )
                    )

                    answer = response.text

                    st.markdown(answer)

                    # ----------------------------------------
                    # SAVE RESPONSE
                    # ----------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


                    # =================================================
                    # VOICE AGENT TEXT-TO-SPEECH
                    # =================================================

                    if mode == "🎧 Voice Agent":

                        with st.spinner(
                            "🔊 Generating voice reply..."
                        ):

                            tts_response = (
                                client.models.generate_content(
                                    model=(
                                        "gemini-3.1-flash-tts-preview"
                                    ),
                                    contents=(
                                        "Speak this answer naturally and "
                                        "clearly:\n\n"
                                        + answer
                                    ),
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
                            )

                            # Get raw audio bytes
                            audio_bytes = None

                            for candidate in (
                                tts_response.candidates
                            ):

                                if candidate.content:

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


                            # =========================================
                            # CONVERT PCM TO WAV
                            # =========================================

                            if audio_bytes:

                                wav_buffer = BytesIO()

                                with wave.open(
                                    wav_buffer,
                                    "wb"
                                ) as wav_file:

                                    wav_file.setnchannels(1)
                                    wav_file.setsampwidth(2)
                                    wav_file.setframerate(24000)

                                    wav_file.writeframes(
                                        audio_bytes
                                    )

                                wav_buffer.seek(0)

                                st.audio(
                                    wav_buffer,
                                    format="audio/wav"
                                )

                                st.download_button(
                                    label="⬇️ Download Voice Reply",
                                    data=wav_buffer.getvalue(),
                                    file_name=(
                                        "ash_voice_reply.wav"
                                    ),
                                    mime="audio/wav"
                                )

                            else:

                                st.warning(
                                    "The AI generated a text answer, "
                                    "but no audio was returned."
                                )


                except Exception as e:

                    st.error(
                        "❌ Something went wrong."
                    )

                    st.code(
                        str(e)
                    )
