# -*- coding: utf-8 -*-
"""Streamlit app for AI Voice Recognition (with Punctuation)"""

import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
import torch
import torchaudio
import tempfile
import soundfile as sf
import numpy as np
import noisereduce as nr
from transformers import AutoTokenizer, AutoModelForTokenClassification

st.title("🎙️ AI Voice Recognition (with Punctuation)")
st.markdown(
    """
    AI-powered voice recognition with basic noise reduction and **intelligent punctuation**.
    Please upload WAV files smaller than 500KB for best performance.
    """
)

# Load Wav2Vec2 ASR model
@st.cache_resource
def load_asr_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", use_auth_token=False)
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", use_auth_token=False)
    return processor, model

processor, model = load_asr_model()

# Load punctuation restoration model
@st.cache_resource
def load_punctuation_model():
    model_name = "oliverguhr/fullstop-punctuation-multilingual-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return tokenizer, model

punctuation_restorer = load_punctuation_model()

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

        reduced_noise = nr.reduce_noise(y=speech_array, sr=sampling_rate)
        speech_array = reduced_noise

    except Exception as e:
        st.error(f"Error loading/processing audio file: {e}")
        st.stop()

    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech = resampler(torch.tensor(speech_array).unsqueeze(0)).squeeze().numpy()
    else:
        speech = speech_array

    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

    with st.spinner("Transcribing... please wait ⏳"):
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

    st.markdown("### ✏️ Raw Transcription:")
    st.success(transcription)
    st.markdown(f"**🔢 Word Count:** {len(transcription.split())}")

    with st.spinner("Adding punctuation... ✍️"):
        tokenizer, punctuation_model = load_punctuation_model()
        inputs_for_punctuation = tokenizer(transcription.split(), is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = punctuation_model(**inputs_for_punctuation)
        predictions = torch.argmax(outputs.logits, dim=2)

        punctuation_map = {
            0: "",
            1: ".",
            2: ",",
            3: "?",
            4: "-",
            5: ":"
        }

        predicted_punctuations = [punctuation_map.get(p.item(), "") for p in predictions[0][1:-1]]

        punctuated_words = []
        word_index = 0
        for i, token in enumerate(transcription.split()):
            punctuated_words.append(token)
            if word_index < len(predicted_punctuations) and predicted_punctuations[word_index]:
                punctuated_words[-1] += predicted_punctuations[word_index]
            word_index += 1

        punctuated_text = " ".join(punctuated_words).strip()

        st.markdown("### 📝 Transcription with Punctuation:")
        st.info(punctuated_text)
