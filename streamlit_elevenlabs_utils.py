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

def get_elevenlabs_client() -> ElevenLabs:
    """
    Create an ElevenLabs client using the key in st.session_state 
    """
    api_key = st.session_state.get("eleven_labs_api_key", "")
    if not api_key:
        st.warning("No Eleven Labs API key found. Please provide it in the sidebar.")
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
        return b""  # Return empty bytes if no text

    client = get_elevenlabs_client()
    try:
        # This call returns a generator of audio chunks
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )

        # Manually concatenate chunks into a single bytes object
        audio_bytes = b""
        for chunk in audio_generator:
            audio_bytes += chunk

        return audio_bytes  
    except Exception as e:
        st.error(f"Error calling Eleven Labs TTS: {e}")
        return b""
