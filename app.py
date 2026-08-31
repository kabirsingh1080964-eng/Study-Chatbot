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


# ============================================================
# HELPER FUNCTION
# ============================================================

def pcm_to_wav(pcm_data):
    """
    Convert Gemini PCM audio into WAV audio.
    """

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

st.title("ASH Study Assistant")

st.caption(
    "Your AI-powered university study assistant."
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# BOTTOM CHAT AREA
# ============================================================

voice_col, chat_col, plus_col = st.columns(
    [0.7, 8, 0.7],
    vertical_alignment="bottom"
)


# ============================================================
# MICROPHONE
# ============================================================

with voice_col:

    audio_value = st.audio_input(
        "🎤",
        label_visibility="collapsed"
    )


# ============================================================
# TEXT INPUT
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
        "➕",
        use_container_width=True
    ):

        st.subheader("Study Tools")

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
            )
        )

        st.session_state.selected_mode = selected_mode

        st.divider()

        # ====================================================
        # PDF UPLOAD
        # ====================================================

        st.write("📄 Upload Study Material")

        uploaded_file = st.file_uploader(
            "Choose PDF",
            type=["pdf"]
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
                        extracted_text += text + "\n"

                st.session_state.pdf_text = extracted_text

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
# VOICE → TEXT
# ============================================================

voice_prompt = None

if audio_value is not None:

    with st.spinner(
        "🎧 Listening..."
    ):

        try:

            # Get actual MIME type from browser
            audio_mime_type = (
                audio_value.type
                if audio_value.type
                else "audio/wav"
            )

            # Upload recorded audio to Gemini
            audio_file = client.files.upload(
                file=audio_value,
                config={
                    "mime_type": audio_mime_type
                }
            )

            # Ask Gemini to transcribe it
            voice_response = client.models.generate_content(
                model="gemini-3.7-flash",
                contents=[
                    audio_file,
                    (
                        "Listen to the student's voice. "
                        "Convert it into the exact question "
                        "they asked. "
                        "Return ONLY the question. "
                        "Do not answer it."
                    )
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=200
                )
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
                "❌ Could not understand your voice."
            )

            st.code(
                str(e)
            )


# ============================================================
# SELECT USER QUESTION
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
    # CURRENT MODE
    # ========================================================

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

                    image_buffer = BytesIO()

                    image.save(
                        image_buffer,
                        format="PNG"
                    )

                    image_bytes = (
                        image_buffer.getvalue()
                    )

                    st.download_button(
                        "⬇️ Download Image",
                        data=image_bytes,
                        file_name="ash_study_image.png",
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
    # NORMAL AI CHAT
    # ========================================================

    else:

        # ====================================================
        # STUDY INSTRUCTIONS
        # ====================================================

        instructions = {

            "💬 Normal Chat":
                (
                    "Answer the question clearly "
                    "and accurately."
                ),

            "📚 Explain Topic":
                (
                    "Explain the topic in extremely "
                    "simple English. Use examples "
                    "and step-by-step explanations."
                ),

            "📝 Make Notes":
                (
                    "Create short revision notes using "
                    "headings, bullet points, definitions "
                    "and examples."
                ),

            "❓ Generate MCQs":
                (
                    "Create 10 important MCQs. "
                    "Each question must have four options "
                    "A, B, C and D. "
                    "Clearly show the correct answer."
                ),

            "🎯 Exam Questions":
                (
                    "Create important university "
                    "exam-style questions. "
                    "Include short and long questions."
                )
        }


        # ====================================================
        # PDF CONTEXT
        # ====================================================

        pdf_context = ""

        if st.session_state.pdf_text:

            pdf_text = (
                st.session_state.pdf_text
            )

            # Limit PDF size for faster response
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


        # ====================================================
        # SYSTEM PROMPT
        # ====================================================

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

{pdf_context}
"""


        # ====================================================
        # GEMINI RESPONSE
        # ====================================================

        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 Thinking..."
            ):

                try:

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

                    answer = (
                        response.text
                        if response.text
                        else "I couldn't generate an answer."
                    )

                    # --------------------------------------------
                    # SHOW TEXT ANSWER
                    # --------------------------------------------

                    st.markdown(
                        answer
                    )

                    # --------------------------------------------
                    # SAVE ANSWER
                    # --------------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


                    # =================================================
                    # VOICE REPLY
                    #
                    # IMPORTANT:
                    # Only generate speech when the user used
                    # the microphone.
                    # Normal typed questions remain text-only.
                    # =================================================

                    if voice_prompt:

                        with st.spinner(
                            "🔊 Speaking..."
                        ):

                            try:

                                tts_response = (
                                    client.models.generate_content(
                                        model=(
                                            "gemini-3.1-flash-tts-preview"
                                        ),
                                        contents=(
                                            "Speak this answer naturally "
                                            "and conversationally. "
                                            "Do not add anything that "
                                            "is not in the answer.\n\n"
                                            + answer
                                        ),
                                        config=(
                                            types.GenerateContentConfig(
                                                response_modalities=[
                                                    "AUDIO"
                                                ],
                                                speech_config=(
                                                    types.SpeechConfig(
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
                                    )
                                )


                                # -------------------------------------
                                # FIND AUDIO
                                # -------------------------------------

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


                                # -------------------------------------
                                # PLAY AUDIO
                                # -------------------------------------

                                if audio_bytes:

                                    wav_audio = pcm_to_wav(
                                        audio_bytes
                                    )

                                    st.audio(
                                        wav_audio,
                                        format="audio/wav"
                                    )

                                else:

                                    st.warning(
                                        "⚠️ AI answered in text, "
                                        "but no voice audio was returned."
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
