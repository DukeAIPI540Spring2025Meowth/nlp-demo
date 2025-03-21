# This utility script manages text-to-speech (TTS) conversion using the official ElevenLabs API.
# It defines helper functions that:
#   1) Initialize an ElevenLabs API client, retrieving the API key securely either from Streamlit's session state
#      or environment variables, and providing clear user warnings if the key is missing.
#   2) Convert input text into audio using a specified voice ID and the multilingual ElevenLabs model.
#      It carefully handles the streaming audio output, manually assembling audio chunks into a single byte-stream
#      suitable for immediate playback within the Streamlit application. Robust error handling ensures that issues
#      with the TTS conversion are clearly communicated to the user.

import os
import streamlit as st
from elevenlabs import ElevenLabs
from typing import Optional 

def get_elevenlabs_client() -> Optional[ElevenLabs]:
    """
    Create an ElevenLabs client using the API key from st.session_state.
    Returns None if no key is provided.
    """
    api_key = st.session_state.get("eleven_labs_api_key", "")
    if not api_key:
        st.warning("No ElevenLabs API key found. Text-to-speech will be disabled.")
        return None
    return ElevenLabs(api_key=api_key)

def speak_text(text, voice_id):
    """
    Use the ElevenLabs official library for text-to-speech.
    1) Creates a client with your ElevenLabs API key.
    2) Converts `text` to speech with the given `voice_id`.
    3) Manually accumulates generator chunks into a single bytes object.
    4) Returns that bytes object for usage (st.audio(...) in the main loop).
    """
    text = text.strip()
    if not text:
        return b""  # No audio for empty text

    client = get_elevenlabs_client()
    if not client:
        return b""  # Gracefully handle missing client (no API key provided)

    try:
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        audio_bytes = b"".join(audio_generator)
        return audio_bytes
    except Exception as e:
        st.error(f"Error during ElevenLabs TTS conversion: {e}")
        return b""
