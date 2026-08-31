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

if "voice_agent_active" not in st.session_state:
st.session_state.voice_agent_active = False

# ============================================================

# CUSTOM CSS

# ============================================================

st.markdown(
""" <style>

```
/* Main page width */
.block-container {
    max-width: 900px;
    padding-bottom: 120px;
}

/* Small circular-style tool buttons */
div[data-testid="stButton"] button {
    border-radius: 50px;
}

/* Reduce audio input height */
div[data-testid="stAudioInput"] {
    min-width: 55px;
}

/* Voice agent button */
.voice-agent-label {
    text-align: center;
    font-size: 12px;
    margin-top: -5px;
    opacity: 0.75;
}

</style>
""",
unsafe_allow_html=True
```

)

# ============================================================

# HEADER

# ============================================================

st.image(
"logo.png",
width=120
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

```
with st.chat_message(
    message["role"]
):

    st.markdown(
        message["content"]
    )
```

# ============================================================

# BOTTOM CHAT AREA

#

# 🎤 | Ask anything... | 🎧 | ➕

# ============================================================

voice_col, chat_col, agent_col, plus_col = st.columns(
[0.8, 7, 0.8, 0.8],
vertical_alignment="bottom"
)

# ============================================================

# 🎤 VOICE INPUT

# ============================================================

with voice_col:

```
audio_value = st.audio_input(
    "🎤",
    label_visibility="collapsed"
)
```

# ============================================================

# 💬 CHAT INPUT

# ============================================================

with chat_col:

```
typed_prompt = st.chat_input(
    "Ask anything about your studies..."
)
```

# ============================================================

# 🎧 VOICE AGENT BUTTON

# ============================================================

with agent_col:

```
voice_agent_button = st.button(
    "🎧",
    help="Voice Agent",
    use_container_width=True
)

if voice_agent_button:

    st.session_state.voice_agent_active = True

    st.session_state.selected_mode = (
        "🎧 Voice Agent"
    )

    st.rerun()
```

# ============================================================

# ➕ PLUS MENU

# ============================================================

with plus_col:

```
with st.popover(
    "➕",
    use_container_width=True
):

    st.subheader(
        "Study Tools"
    )

    # ----------------------------------------------------
    # STUDY MODES
    # ----------------------------------------------------

    st.write(
        "📚 Study Mode"
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
        "Choose mode",
        study_modes,
        index=study_modes.index(
            st.session_state.selected_mode
            if st.session_state.selected_mode in study_modes
            else "💬 Normal Chat"
        ),
        label_visibility="collapsed"
    )

    st.session_state.selected_mode = selected_mode

    # ----------------------------------------------------
    # VOICE AGENT STATUS
    # ----------------------------------------------------

    if st.session_state.voice_agent_active:

        st.info(
            "🎧 Voice Agent is active"
        )

        if st.button(
            "❌ Exit Voice Agent",
            use_container_width=True
        ):

            st.session_state.voice_agent_active = False

            st.session_state.selected_mode = (
                "💬 Normal Chat"
            )

            st.rerun()

    st.divider()

    # ----------------------------------------------------
    # PDF UPLOAD
    # ----------------------------------------------------

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

            pdf_reader = PdfReader(
                uploaded_file
            )

            pdf_text = ""

            for page in pdf_reader.pages:

                text = page.extract_text()

                if text:

                    pdf_text += (
                        text + "\n"
                    )

            st.session_state.pdf_text = pdf_text

            st.session_state.pdf_name = (
                uploaded_file.name
            )

            st.success(
                f"✅ {uploaded_file.name}"
            )

            st.caption(
                f"{len(pdf_reader.pages)} pages"
            )

        except Exception as e:

            st.error(
                "Could not read PDF."
            )

            st.code(
                str(e)
            )

    elif st.session_state.pdf_name:

        st.success(
            f"📄 {st.session_state.pdf_name}"
        )

    # ----------------------------------------------------
    # REMOVE PDF
    # ----------------------------------------------------

    if st.session_state.pdf_name:

        if st.button(
            "🗑️ Remove PDF",
            use_container_width=True
        ):

            st.session_state.pdf_text = ""

            st.session_state.pdf_name = ""

            st.rerun()
```

# ============================================================

# VOICE TO TEXT

# ============================================================

voice_prompt = None

if audio_value is not None:

```
with st.spinner(
    "🎧 Understanding..."
):

    try:

        audio_file = client.files.upload(
            file=audio_value,
            config={
                "mime_type": "audio/wav"
            }
        )

        voice_response = (
            client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    audio_file,
                    (
                        "Convert this spoken "
                        "question to text. "
                        "Return ONLY the question."
                    )
                ]
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
            "❌ Voice could not be understood."
        )

        st.code(
            str(e)
        )
```

# ============================================================

# SELECT PROMPT

# ============================================================

prompt = typed_prompt

if voice_prompt:

```
prompt = voice_prompt
```

# ============================================================

# PROCESS QUESTION

# ============================================================

if prompt:

```
# ========================================================
# SAVE USER MESSAGE
# ========================================================

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


# ========================================================
# CURRENT MODE
# ========================================================

mode = st.session_state.selected_mode


# ========================================================
# IMAGE GENERATION
# ========================================================

if mode == "🎨 Generate Image":

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
# TEXT + VOICE AGENT
# ========================================================

else:

    instructions = {

        "💬 Normal Chat":
            "Answer clearly and accurately.",

        "📚 Explain Topic":
            (
                "Explain in very simple "
                "English with examples "
                "and steps."
            ),

        "📝 Make Notes":
            (
                "Create short revision "
                "notes with headings "
                "and bullet points."
            ),

        "❓ Generate MCQs":
            (
                "Create 10 MCQs with "
                "four options and show "
                "the correct answer."
            ),

        "🎯 Exam Questions":
            (
                "Create important university "
                "exam questions including "
                "short and long questions."
            ),

        "🎧 Voice Agent":
            (
                "Act like a friendly AI "
                "voice assistant. "
                "Answer naturally, "
                "conversationally and briefly."
            )
    }


    # ====================================================
    # PDF CONTEXT
    # ====================================================

    pdf_context = ""

    if st.session_state.pdf_text:

        words = [
            word.lower()
            for word in prompt.split()
            if len(word) > 3
        ]

        paragraphs = (
            st.session_state.pdf_text
            .split("\n")
        )

        relevant_parts = []

        for paragraph in paragraphs:

            paragraph_lower = (
                paragraph.lower()
            )

            if any(
                word in paragraph_lower
                for word in words
            ):

                relevant_parts.append(
                    paragraph
                )

            if len(
                "\n".join(
                    relevant_parts
                )
            ) >= 5000:

                break

        if relevant_parts:

            pdf_context = (
                "\n\nPDF CONTENT:\n"
                + "\n".join(
                    relevant_parts
                )
            )

        else:

            pdf_context = (
                "\n\nPDF CONTENT:\n"
                + st.session_state.pdf_text[
                    :3000
                ]
            )


    # ====================================================
    # SYSTEM PROMPT
    # ====================================================

    system_prompt = f"""
```

You are ASH Study Assistant.

Task:
{instructions[mode]}

Rules:

* Use simple English.
* Be accurate.
* Be helpful.
* Explain difficult things step by step.
* Avoid unnecessary long answers.

If PDF content is provided,
use it as the main source.

{pdf_context}
"""

```
    # ====================================================
    # GEMINI RESPONSE
    # ====================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🤖 Thinking..."
        ):

            try:

                response = (
                    client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=(
                            system_prompt
                            + "\n\nQuestion:\n"
                            + prompt
                        )
                    )
                )

                answer = response.text

                st.markdown(
                    answer
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


                # =================================================
                # 🎧 VOICE AGENT RESPONSE
                # =================================================

                if mode == "🎧 Voice Agent":

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
                                        "Say this naturally "
                                        "and conversationally:\n\n"
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

                                wav_buffer = (
                                    BytesIO()
                                )

                                with wave.open(
                                    wav_buffer,
                                    "wb"
                                ) as wav_file:

                                    wav_file.setnchannels(
                                        1
                                    )

                                    wav_file.setsampwidth(
                                        2
                                    )

                                    wav_file.setframerate(
                                        24000
                                    )

                                    wav_file.writeframes(
                                        audio_bytes
                                    )

                                wav_buffer.seek(
                                    0
                                )

                                st.audio(
                                    wav_buffer,
                                    format="audio/wav"
                                )

                            else:

                                st.warning(
                                    "No voice audio was returned."
                                )

                        except Exception as e:

                            st.warning(
                                "Text response worked, "
                                "but voice generation failed."
                            )

                            st.code(
                                str(e)
                            )


            except Exception as e:

                st.error(
                    "❌ Something went wrong."
                )

                st.code(
                    str(e)
                )
