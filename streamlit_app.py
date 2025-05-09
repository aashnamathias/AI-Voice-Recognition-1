import time
import random
import requests
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import streamlit as st
from typing import Optional, Tuple

def load_asr_model(language) -> Tuple[Optional[Wav2Vec2Processor], Optional[Wav2Vec2ForCTC]]:
    """Loads the ASR model with retry, exponential backoff, and jitter."""
    max_retries = 5
    base_delay = 1  # Initial delay in seconds
    model_name = "facebook/wav2vec2-large-960h-lv60-self"

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
            return processor, model
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                delay = base_delay * (2**attempt)  # Exponential backoff
                jitter = random.uniform(0, 1)  # Add jitter
                sleep_duration = delay + jitter
                print(
                    f"Hugging Face rate limit hit. Retrying in {sleep_duration:.2f} seconds (Attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_duration)
            else:
                st.error(f"Error loading model: {e}")
                print(f"Error loading model: {e}")
                return None, None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            print(f"Error loading model: {e}")
            return None, None

    st.error(
        "Failed to load the model after multiple retries. Please check your network connection and try again later."
    )
    return None, None
