import streamlit as st


# ============================================================
# GEMINI LIVE VOICE COMPONENT
# ============================================================

VOICE_HTML = """

<div id="ash-voice-container">

    <button id="ash-voice-button">
        🎤
    </button>

    <div id="ash-voice-status">
        Tap 🎤 to talk
    </div>

</div>

"""


VOICE_CSS = """

#ash-voice-container {

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    gap: 3px;

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}


#ash-voice-button {

    width: 42px;

    height: 42px;

    border-radius: 50%;

    border: 1px solid
        rgba(128,128,128,0.35);

    background:
        var(--st-secondary-background-color);

    color:
        var(--st-text-color);

    font-size: 20px;

    cursor: pointer;

    display: flex;

    align-items: center;

    justify-content: center;

    transition:
        transform 0.15s ease,
        background 0.15s ease;

}


#ash-voice-button:hover {

    transform: scale(1.05);

}


#ash-voice-button.listening {

    background: #ef4444;

    color: white;

    animation:
        ash-pulse 1.2s infinite;

}


@keyframes ash-pulse {

    0% {
        box-shadow:
            0 0 0 0
            rgba(239,68,68,0.5);
    }

    70% {
        box-shadow:
            0 0 0 10px
            rgba(239,68,68,0);
    }

    100% {
        box-shadow:
            0 0 0 0
            rgba(239,68,68,0);
    }

}


#ash-voice-status {

    font-size: 9px;

    opacity: 0.65;

    white-space: nowrap;

}

"""


VOICE_JS = r"""

export default function(component) {

    const {
        parentElement,
        data,
        setStateValue
    } = component;


    const button =
        parentElement.querySelector(
            "#ash-voice-button"
        );


    const status =
        parentElement.querySelector(
            "#ash-voice-status"
        );


    // ========================================================
    // VARIABLES
    // ========================================================

    let websocket = null;

    let audioContext = null;

    let microphoneStream = null;

    let processor = null;

    let source = null;

    let isListening = false;

    let audioQueue = [];

    let isPlaying = false;

    let nextPlayTime = 0;

    let currentSourceNodes = [];


    // ========================================================
    // GEMINI LIVE MODEL
    // ========================================================

    const MODEL =
        "gemini-3.1-flash-live-preview";


    // ========================================================
    // BASE64 HELPERS
    // ========================================================

    function base64ToUint8Array(base64) {

        const binary =
            atob(base64);

        const bytes =
            new Uint8Array(
                binary.length
            );

        for (
            let i = 0;
            i < binary.length;
            i++
        ) {

            bytes[i] =
                binary.charCodeAt(i);

        }

        return bytes;
    }


    function floatToPCM16(float32Array) {

        const pcm =
            new Int16Array(
                float32Array.length
            );


        for (
            let i = 0;
            i < float32Array.length;
            i++
        ) {

            let sample =
                Math.max(
                    -1,
                    Math.min(
                        1,
                        float32Array[i]
                    )
                );


            pcm[i] =
                sample < 0
                    ? sample * 32768
                    : sample * 32767;

        }


        return new Uint8Array(
            pcm.buffer
        );
    }


    function arrayBufferToBase64(buffer) {

        let binary = "";

        const bytes =
            new Uint8Array(buffer);


        const chunkSize = 0x8000;


        for (
            let i = 0;
            i < bytes.length;
            i += chunkSize
        ) {

            binary += String.fromCharCode(
                ...bytes.subarray(
                    i,
                    Math.min(
                        i + chunkSize,
                        bytes.length
                    )
                )
            );

        }


        return btoa(binary);
    }


    // ========================================================
    // RESAMPLE MICROPHONE AUDIO
    // ========================================================

    function downsampleTo16k(
        input,
        inputSampleRate
    ) {

        const outputSampleRate = 16000;


        if (
            inputSampleRate ===
            outputSampleRate
        ) {

            return input;

        }


        const ratio =
            inputSampleRate /
            outputSampleRate;


        const outputLength =
            Math.round(
                input.length /
                ratio
            );


        const output =
            new Float32Array(
                outputLength
            );


        let offsetResult = 0;

        let offsetBuffer = 0;


        while (
            offsetResult <
            outputLength
        ) {

            const nextOffsetBuffer =
                Math.round(
                    (offsetResult + 1) *
                    ratio
                );


            let accum = 0;

            let count = 0;


            for (
                let i =
                    offsetBuffer;

                i <
                    nextOffsetBuffer &&
                i <
                    input.length;

                i++
            ) {

                accum += input[i];

                count++;

            }


            output[offsetResult] =
                count > 0
                    ? accum / count
                    : 0;


            offsetResult++;

            offsetBuffer =
                nextOffsetBuffer;

        }


        return output;
    }


    // ========================================================
    // STOP PLAYBACK
    // ========================================================

    function stopPlayback() {

        for (
            const node
            of currentSourceNodes
        ) {

            try {

                node.stop();

            } catch (e) {}

        }


        currentSourceNodes = [];

        audioQueue = [];

        isPlaying = false;

        nextPlayTime =
            audioContext
                ? audioContext.currentTime
                : 0;

    }


    // ========================================================
    // PLAY GEMINI AUDIO
    // ========================================================

    function playAudioChunk(
        base64Audio
    ) {

        if (!audioContext) {

            return;

        }


        const bytes =
            base64ToUint8Array(
                base64Audio
            );


        const pcm16 =
            new Int16Array(
                bytes.buffer,
                bytes.byteOffset,
                Math.floor(
                    bytes.byteLength / 2
                )
            );


        const float32 =
            new Float32Array(
                pcm16.length
            );


        for (
            let i = 0;
            i < pcm16.length;
            i++
        ) {

            float32[i] =
                pcm16[i] / 32768;

        }


        const audioBuffer =
            audioContext.createBuffer(
                1,
                float32.length,
                24000
            );


        audioBuffer
            .getChannelData(0)
            .set(float32);


        const sourceNode =
            audioContext.createBufferSource();


        sourceNode.buffer =
            audioBuffer;


        sourceNode.connect(
            audioContext.destination
        );


        const now =
            audioContext.currentTime;


        if (
            nextPlayTime <
            now
        ) {

            nextPlayTime = now;

        }


        sourceNode.start(
            nextPlayTime
        );


        currentSourceNodes.push(
            sourceNode
        );


        sourceNode.onended = () => {

            const index =
                currentSourceNodes.indexOf(
                    sourceNode
                );


            if (index >= 0) {

                currentSourceNodes.splice(
                    index,
                    1
                );

            }

        };


        nextPlayTime +=
            audioBuffer.duration;

    }


    // ========================================================
    // HANDLE GEMINI MESSAGE
    // ========================================================

    function handleGeminiMessage(
        event
    ) {

        try {

            const message =
                JSON.parse(
                    event.data
                );


            // ----------------------------------------------
            // SETUP COMPLETE
            // ----------------------------------------------

            if (
                message.setupComplete
            ) {

                status.textContent =
                    "🟢 Connected — speak naturally";

                return;

            }


            // ----------------------------------------------
            // SERVER CONTENT
            // ----------------------------------------------

            const serverContent =
                message.serverContent;


            if (!serverContent) {

                return;

            }


            // ----------------------------------------------
            // INTERRUPTION
            // ----------------------------------------------

            if (
                serverContent.interrupted
            ) {

                stopPlayback();

                status.textContent =
                    "🎤 Listening...";

                return;

            }


            // ----------------------------------------------
            // MODEL AUDIO
            // ----------------------------------------------

            const modelTurn =
                serverContent.modelTurn;


            if (
                modelTurn &&
                modelTurn.parts
            ) {

                for (
                    const part
                    of modelTurn.parts
                ) {

                    if (
                        part.inlineData &&
                        part.inlineData.data
                    ) {

                        playAudioChunk(
                            part.inlineData.data
                        );

                    }

                }

            }


            // ----------------------------------------------
            // INPUT TRANSCRIPTION
            // ----------------------------------------------

            if (
                serverContent.inputTranscription &&
                serverContent.inputTranscription.text
            ) {

                setStateValue(
                    "user_transcript",
                    serverContent
                        .inputTranscription
                        .text
                );

            }


            // ----------------------------------------------
            // OUTPUT TRANSCRIPTION
            // ----------------------------------------------

            if (
                serverContent.outputTranscription &&
                serverContent.outputTranscription.text
            ) {

                setStateValue(
                    "assistant_transcript",
                    serverContent
                        .outputTranscription
                        .text
                );

            }


            // ----------------------------------------------
            // TURN COMPLETE
            // ----------------------------------------------

            if (
                serverContent.turnComplete
            ) {

                status.textContent =
                    "🎤 Your turn";

            }

        }

        catch (error) {

            console.error(
                "Gemini message error:",
                error
            );

        }

    }


    // ========================================================
    // SEND SETUP MESSAGE
    // ========================================================

    function sendSetup() {

        const setupMessage = {

            setup: {

                model:
                    "models/" +
                    MODEL,

                generationConfig: {

                    responseModalities: [
                        "AUDIO"
                    ],

                    speechConfig: {

                        voiceConfig: {

                            prebuiltVoiceConfig: {

                                voiceName:
                                    "Kore"

                            }

                        }

                    }

                },

                systemInstruction: {

                    parts: [

                        {

                            text:
                                data.system_instruction ||
                                "You are ASH Study Assistant."

                        }

                    ]

                },

                realtimeInputConfig: {

                    automaticActivityDetection: {

                        disabled: false

                    },

                    activityHandling:
                        "START_OF_ACTIVITY_INTERRUPTS"

                },

                inputAudioTranscription: {},

                outputAudioTranscription: {}

            }

        };


        websocket.send(
            JSON.stringify(
                setupMessage
            )
        );

    }


    // ========================================================
    // CONNECT GEMINI LIVE
    // ========================================================

    function connectWebSocket() {

        if (
            websocket &&
            websocket.readyState ===
                WebSocket.OPEN
        ) {

            return;

        }


        if (!data.token) {

            status.textContent =
                "❌ Voice unavailable";

            return;

        }


        status.textContent =
            "Connecting...";


        const url =
            "wss://generativelanguage.googleapis.com/"
            + "ws/google.ai.generativelanguage.v1alpha."
            + "GenerativeService."
            + "BidiGenerateContentConstrained"
            + "?access_token="
            + encodeURIComponent(
                data.token
            );


        websocket =
            new WebSocket(
                url
            );


        websocket.onopen = () => {

            sendSetup();

        };


        websocket.onmessage =
            handleGeminiMessage;


        websocket.onerror = (
            error
        ) => {

            console.error(
                "Gemini Live error:",
                error
            );


            status.textContent =
                "❌ Connection error";

        };


        websocket.onclose = () => {

            websocket = null;

            if (isListening) {

                stopMicrophone();

            }


            status.textContent =
                "Tap 🎤 to talk";

        };

    }


    // ========================================================
    // SEND AUDIO
    // ========================================================

    function sendAudio(
        pcmBytes
    ) {

        if (
            !websocket ||
            websocket.readyState !==
                WebSocket.OPEN
        ) {

            return;

        }


        const audioMessage = {

            realtimeInput: {

                audio: {

                    data:
                        arrayBufferToBase64(
                            pcmBytes
                        ),

                    mimeType:
                        "audio/pcm;rate=16000"

                }

            }

        };


        websocket.send(
            JSON.stringify(
                audioMessage
            )
        );

    }


    // ========================================================
    // START MICROPHONE
    // ========================================================

    async function startMicrophone() {

        try {

            if (
                !websocket ||
                websocket.readyState !==
                    WebSocket.OPEN
            ) {

                connectWebSocket();

                status.textContent =
                    "Connecting...";

                return;

            }


            audioContext =
                new (
                    window.AudioContext ||
                    window.webkitAudioContext
                )();


            await audioContext.resume();


            microphoneStream =
                await navigator
                    .mediaDevices
                    .getUserMedia(
                        {
                            audio: {

                                channelCount: 1,

                                echoCancellation: true,

                                noiseSuppression: true,

                                autoGainControl: true

                            }

                        }
                    );


            source =
                audioContext.createMediaStreamSource(
                    microphoneStream
                );


            processor =
                audioContext.createScriptProcessor(
                    4096,
                    1,
                    1
                );


            processor.onaudioprocess =
                function(event) {

                    if (!isListening) {

                        return;

                    }


                    const input =
                        event.inputBuffer
                            .getChannelData(0);


                    const downsampled =
                        downsampleTo16k(
                            input,
                            audioContext
                                .sampleRate
                        );


                    const pcm =
                        floatToPCM16(
                            downsampled
                        );


                    sendAudio(
                        pcm.buffer
                    );

                };


            source.connect(
                processor
            );


            processor.connect(
                audioContext.destination
            );


            isListening = true;


            button.classList.add(
                "listening"
            );


            button.textContent =
                "⏹️";


            status.textContent =
                "🎤 Listening...";

        }

        catch (error) {

            console.error(
                "Microphone error:",
                error
            );


            status.textContent =
                "❌ Microphone permission denied";

        }

    }


    // ========================================================
    // STOP MICROPHONE
    // ========================================================

    function stopMicrophone() {

        isListening = false;


        button.classList.remove(
            "listening"
        );


        button.textContent =
            "🎤";


        if (processor) {

            try {

                processor.disconnect();

            } catch (e) {}

            processor = null;

        }


        if (source) {

            try {

                source.disconnect();

            } catch (e) {}

            source = null;

        }


        if (microphoneStream) {

            microphoneStream
                .getTracks()
                .forEach(
                    track => track.stop()
                );

            microphoneStream = null;

        }


        status.textContent =
            "Tap 🎤 to talk";

    }


    // ========================================================
    // BUTTON
    // ========================================================

    button.onclick = async () => {

        if (isListening) {

            stopMicrophone();

        }

        else {

            await startMicrophone();

        }

    };


    // ========================================================
    // CLEANUP
    // ========================================================

    return () => {

        stopMicrophone();

        stopPlayback();


        if (websocket) {

            try {

                websocket.close();

            } catch (e) {}

            websocket = null;

        }


        if (audioContext) {

            try {

                audioContext.close();

            } catch (e) {}

            audioContext = null;

        }

    };

}
"""


# ============================================================
# REGISTER COMPONENT
# ============================================================

_live_voice_component = st.components.v2.component(

    name="ash_live_voice_agent",

    html=VOICE_HTML,

    css=VOICE_CSS,

    js=VOICE_JS
)


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def voice_agent(
    token,
    system_instruction="",
    key=None
):

    return _live_voice_component(

        data={

            "token": token,

            "system_instruction":
                system_instruction

        },

        key=key,

        default={

            "user_transcript": "",

            "assistant_transcript": ""

        }
    )
