# -*- coding: utf-8 -*-
"""Streamlit app for AI Voice Recognition (English, French, Chinese, Hindi)"""

import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
import tempfile
import soundfile as sf
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
import gc
import time  # Import the time module
import requests # Import the requests module


st.title("🎙️ AI Voice Recognition (English, French, Chinese, Hindi)")
st.markdown(
    "This app supports voice recognition for English, French, Chinese, and Hindi."
)
st.markdown(
    "Due to resource limitations, the punctuation applied is a basic, rule-based segmentation and capitalization."
)

# Initialize session state for language
if "language" not in st.session_state:
    st.session_state["language"] = "English"


# Language selection
languages = ["English", "French", "Chinese", "Hindi"]
new_language = st.selectbox(
    "Select the language of the audio:",
    languages,
    key="language_selectbox",  # Unique key for the selectbox
    index=languages.index(st.session_state["language"])
    if st.session_state["language"] in languages
    else 0,
)

# Check if the language has changed
if new_language != st.session_state["language"]:
    print(f"Language changed from {st.session_state['language']} to {new_language}")
    st.session_state.clear()
    st.session_state["language"] = new_language
    st.rerun()

# Add a reset button
if st.button("Reset App"):
    print("Reset button clicked")
    st.session_state.clear()
    st.session_state["language"] = "English"  # Reset to default language
    st.rerun()


uploaded_file = st.file_uploader(
    "Upload a WAV or MP3 file", type=["wav", "mp3"], key="file_uploader"
)  # Accept MP3


if uploaded_file is not None:
    st.session_state["uploaded_file"] = uploaded_file
    st.audio(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        print(f"Creating temporary file: {tmp.name}")
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        print(f"Processing uploaded file: {uploaded_file.name}, type: {uploaded_file.type}")
        if uploaded_file.type == "audio/mpeg":  # Check if it's an MP3
            print("Detected MP3 file")
            sound = AudioSegment.from_mp3(tmp_path)
            sound = sound.set_frame_rate(16000)  # Ensure consistent sampling rate
            sound = sound.set_channels(1)  # Ensure mono audio
            raw_audio_data = (
                np.array(sound.get_array_of_samples()).astype(np.float32())
                / (2**15 - 1)
            )  # Normalize
            sampling_rate = sound.frame_rate
            speech_array = raw_audio_data
            print(f"MP3: sampling_rate={sampling_rate}, shape={speech_array.shape}")
        else:  # It's a WAV file
            print("Detected WAV file")
            audio, sampling_rate = sf.read(tmp_path)
            speech_array = audio.astype("float32")
            if len(speech_array.shape) > 1:
                speech_array = speech_array.mean(axis=1)
            print(f"WAV: sampling_rate={sampling_rate}, shape={speech_array.shape}")

        # Apply noise reduction
        print("Applying noise reduction...")
        reduced_noise = nr.reduce_noise(y=speech_array, sr=sampling_rate)
        speech_array = reduced_noise
        print("Noise reduction complete.")

    except Exception as e:
        st.error(f"Error loading/processing audio file: {e}")
        print(f"Error loading/processing audio file: {e}")
        st.stop()

    # Resample to 16000 Hz if necessary
    if sampling_rate != 16000:
        print(f"Resampling from {sampling_rate} to 16000 Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sampling_rate, new_freq=16000
        )
        speech = resampler(torch.tensor(speech_array).unsqueeze(0)).squeeze().numpy()
        print(f"Resampled speech shape: {speech.shape}")
    else:
        speech = speech_array
        print("No resampling needed.")

    # Load Wav2Vec2 models
    def load_asr_model(language):
        max_retries = 5  # Maximum number of retries
        retry_delay = 1  # Initial delay in seconds

        with st.spinner(f"Running asr model for {language}"):
            print(f"Loading model for language: {language}")
            model_name = "facebook/wav2vec2-large-960h-lv60-self"  # Default English model

            if language == "French":
                model_name = "facebook/wav2vec2-large-xlsr-53_56k"
            elif language == "Chinese":
                model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
            elif language == "Hindi":
                model_name = "shiwangi27/wave2vec2-large-xlsr-hindi"

            for attempt in range(max_retries):
                try:
                    processor = Wav2Vec2Processor.from_pretrained(model_name)
                    model = Wav2Vec2ForCTC.from_pretrained(model_name)
                    return processor, model  # Return the model and processor if successful
                except Exception as e:
                    if "HTTP Error 429" in str(e):
                        print(
                            f"Hugging Face rate limit hit. Retrying in {retry_delay} seconds (Attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(retry_delay)  # Wait before retrying
                        retry_delay *= 2  # Exponential backoff (double the delay)
                    else:
                        st.error(f"Error loading model: {e}")
                        print(f"Error loading model: {e}")
                        return None, None  # Return None, None on other errors

            st.error(
                "Failed to load the model after multiple retries due to rate limiting.  Please try again later."
            )
            return None, None  # Return None, None if all retries fail

    # Load model and processor
    if st.session_state.get("processor") is None or st.session_state.get("model") is None or st.session_state["language"] != st.session_state.get("model_language", None):
        print(f"Loading model and processor for language: {st.session_state['language']}")
        try:
            processor, model = load_asr_model(st.session_state["language"])
            if processor is not None and model is not None:
                st.session_state["processor"] = processor
                st.session_state["model"] = model
                st.session_state["model_language"] = st.session_state[
                    "language"
                ]  # store the language
                print(
                    f"Model and processor loaded into session state for language: {st.session_state['language']}"
                )
            else:
                st.error(
                    "Failed to load the model or processor. Please try again with a different audio file or language."
                )
                print("Failed to load model or processor.")
                st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            print(f"An unexpected error occurred: {e}")
            st.stop()
    else:
        processor = st.session_state["processor"]
        model = st.session_state["model"]
        print(
            f"Model and processor loaded from session state for language: {st.session_state['language']}"
        )

    # Process the speech input
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    print(f"Input tensor shape: {inputs['input_values'].shape}")

    # Transcription
    with st.spinner(f"Transcribing in {st.session_state['language']}... please wait ⏳"):
        try:
            print("Starting transcription...")
            with torch.no_grad():
                logits = model(**inputs).logits
            print(f"Logits shape: {logits.shape}")
            predicted_ids = torch.argmax(logits, dim=-1)
            print(f"Predicted IDs shape: {predicted_ids.shape}")
            transcription = processor.decode(predicted_ids[0])
            print(f"Transcription: {transcription}")
            st.session_state["transcription"] = transcription
            print("Transcription complete.")
        except Exception as e:
            st.error(f"Error during transcription: {e}")
            print(f"Error during transcription: {e}")
            st.stop()

    st.markdown("### ✏️ Raw Transcription:")
    st.success(st.session_state["transcription"])
    st.markdown(f"**🔢 Word Count:** {len(st.session_state['transcription'].split())}")

    st.markdown("### 📝 Transcription with Basic Punctuation (Word Limit):")
    # Basic Punctuation (using the max_words approach from earlier)
    def segment_and_punctuate(text, max_words=15):
        words = text.split()
        segments = []
        current_segment = []
        for word in words:
            current_segment.append(word)
            if len(current_segment) >= max_words:
                segments.append(" ".join(current_segment) + ".")
                current_segment = []
        if current_segment:
            segments.append(" ".join(current_segment) + ".")
        return " ".join(segments)

    def capitalize_first_letter(punctuated_text):
        segments = punctuated_text.split(".")
        capitalized_segments = []
        for segment in segments:
            stripped_segment = segment.strip()
            if stripped_segment:
                first_word = stripped_segment.split()[0]
                rest_of_segment = " ".join(stripped_segment.split()[1:])
                capitalized_segments.append(
                    first_word[0].upper()
                    + first_word[1:].lower()
                    + (" " + rest_of_segment.lower() if rest_of_segment else "")
                )
            else:
                capitalized_segments.append("")
        return ". ".join(capitalized_segments).strip() + "." if capitalized_segments else ""

    with st.spinner("Adding basic punctuation... ✍️"):
        if st.session_state.get("transcription"):
            punctuated_text = segment_and_punctuate(st.session_state["transcription"])
            capitalized_text = capitalize_first_letter(punctuated_text)
            st.info(capitalized_text)
            st.session_state["punctuated_text"] = capitalized_text

else:
    st.markdown("Please upload a WAV or MP3 file to begin.")

