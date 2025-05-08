# -*- coding: utf-8 -*-
"""Streamlit app for AI Voice Recognition (Multilingual with Basic Punctuation)"""

import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
import tempfile
import soundfile as sf
import numpy as np
import noisereduce as nr

st.title("🎙️ AI Voice Recognition (Multilingual)")
st.markdown("This app supports multilingual voice recognition. Due to resource limitations, the punctuation applied is a basic, rule-based segmentation and capitalization.")

# Language selection
language = st.selectbox(
    "Select the language of the audio:",
    ["English", "French", "German", "Spanish", "Chinese", "Russian", "Japanese", "Hindi", "Other"]
)

# Load Wav2Vec2 models
@st.cache_resource
def load_asr_model(language):
    model_name = "facebook/wav2vec2-large-960h-lv60-self" # Default English model
    processor_name = "facebook/wav2vec2-large-960h-lv60-self"

    if language == "French":
        model_name = "facebook/wav2vec2-large-xlsr-53_56k"
        processor_name = "facebook/wav2vec2-large-xlsr-53_56k"
    elif language == "German":
        model_name = "facebook/wav2vec2-large-xlsr-53_56k"
        processor_name = "facebook/wav2vec2-large-xlsr-53_56k"
    elif language == "Spanish":
        model_name = "facebook/wav2vec2-large-xlsr-53_56k"
        processor_name = "facebook/wav2vec2-large-xlsr-53_56k"
    elif language == "Chinese":
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
        processor_name = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    elif language == "Russian":
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
        processor_name = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    elif language == "Japanese":
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
        processor_name = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
    elif language == "Hindi":
        model_name = "shiwangi27/wave2vec2-large-xlsr-hindi"
        processor_name = "shiwangi27/wave2vec2-large-xlsr-hindi"
    elif language == "Other":
        model_name = "facebook/wav2vec2-large-xlsr-53_56k"
        processor_name = "facebook/wav2vec2-large-xlsr-53_56k"

    processor = Wav2Vec2Processor.from_pretrained(processor_name, use_auth_token=False)
    model = Wav2Vec2ForCTC.from_pretrained(model_name, use_auth_token=False)
    return processor, model

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        audio, sampling_rate = sf.read(tmp_path)
        speech_array = audio.astype("float32")
        if len(speech_array.shape) > 1:
            speech_array = speech_array.mean(axis=1)

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(y=speech_array, sr=sampling_rate)
        speech_array = reduced_noise

    except Exception as e:
        st.error(f"Error loading/processing audio file: {e}")
        st.stop()

    # Resample to 16000 Hz if necessary
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech = resampler(torch.tensor(speech_array).unsqueeze(0)).squeeze().numpy()
    else:
        speech = speech_array

    processor, model = load_asr_model(language)
    # Process the speech input
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

    # Transcription
    with st.spinner(f"Transcribing in {language}... please wait ⏳"):
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

    st.markdown("### ✏️ Raw Transcription:")
    st.success(transcription)
    st.markdown(f"**🔢 Word Count:** {len(transcription.split())}")

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
                capitalized_segments.append(first_word[0].upper() + first_word[1:].lower() + (" " + rest_of_segment.lower() if rest_of_segment else ""))
            else:
                capitalized_segments.append("")
        return ". ".join(capitalized_segments).strip() + "." if capitalized_segments else ""

    with st.spinner("Adding basic punctuation... ✍️"):
        punctuated_text = segment_and_punctuate(transcription)
        capitalized_text = capitalize_first_letter(punctuated_text)
        st.info(capitalized_text)
