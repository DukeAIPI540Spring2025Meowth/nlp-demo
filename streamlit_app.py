# This Streamlit script creates a mental health-oriented virtual assistant named "MindHeal Assistant".
# It provides empathetic, supportive responses using a locally fine-tuned Llama LLM, with an optional
# refinement and evaluation step performed by an external LLM ("Judge") powered by OpenAI. Users
# can select different visual themes ("moods") that dynamically change the app's color scheme, input
# API keys for OpenAI via a sidebar, and interact through text-based chat inputs.
# The assistant returns original and optionally refined responses and a numeric evaluation score.
# Important ethical disclaimers and guidance on the responsible use of this experimental tool
# are provided clearly throughout the user interface.


import streamlit as st

# Import custom LLM modules ---
from streamlit_llm_utils import (
    init_main_llm,
    init_judge_llm,
    generate_main_response,
    judge_refine_and_score
)

# Define mood-based themes, each with colors carefully selected to reflect specific emotional tones.
MOOD_COLORS = {
    "Calm": {
        "bg": "#edf9f9",       # background
        "text": "#000000",     # main text
        "primary": "#01959f",  # accent color for headers
    },
    "Cheerful": {
        "bg": "#fffbe7",
        "text": "#000000",
        "primary": "#ff914d",
    },
    "Reflective": {
        "bg": "#f1f2f6",
        "text": "#000000",
        "primary": "#596275",
    },
    "Somber": {
        "bg": "#e0e0e0",
        "text": "#000000",
        "primary": "#616161",
    },
    "Energetic": {
        "bg": "#fffbd1",
        "text": "#000000",
        "primary": "#ffc107",
    },
}

def set_mood_style(mood: str):
    colors = MOOD_COLORS.get(mood, MOOD_COLORS["Calm"])
    st.markdown(
        f"""
        <style>
        body {{
            background-color: {colors["bg"]};
            color: {colors["text"]}; /* black text */
            margin: 0;
            padding: 0;
            font-family: "Trebuchet MS", sans-serif;
        }}
        /* Force text color to black globally */
        body, .block-container, .block-container * {{
            color: #000000 !important;
        }}
        .block-container {{
            background-color: {colors["bg"]};
            border-radius: 8px;
            padding: 2rem !important;
            margin-top: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: {colors["primary"]} !important;
        }}
        .sidebar .sidebar-content {{
            background-color: #f0f0f0 !important;
        }}
        .css-1n76uvr {{
            background-color: {colors["bg"]} !important; 
            border-radius: 6px;
            margin-bottom: 0.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    # Let user pick a mood from the sidebar, then apply the style
    st.sidebar.header("Mood Theme")
    mood_choice = st.sidebar.selectbox("Select a mood theme:", list(MOOD_COLORS.keys()))
    set_mood_style(mood_choice)

    st.title("MindHeal Assistant: Response Refinement & Evaluation")

    st.markdown(
        "**Disclaimer:** This application is strictly for **informational and educational purposes only**. "
        "It is **not intended** as a replacement for professional medical or mental health advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider or mental health professional with any concerns or questions "
        "about your mental health or medical conditions.\n\n"
        "If you are experiencing a crisis or require immediate support, please contact your local emergency services "
        "or a mental health helpline in your region."
    )

    st.write(
        "Welcome to **MindHeal Assistant**, an empathetic virtual assistant powered by a carefully fine-tuned language model. "
        "It provides thoughtful, supportive, and mental-health-oriented responses to your inquiries. "
        "Additionally, you can enable an optional 'Judge' LLM, powered by OpenAI, to refine these responses further and provide a quality evaluation.\n\n"
        "**Key Features:**\n"
        "1. **Original & Refined Responses:** Compare the initial response from the primary LLM with the professionally refined alternative.\n"
        "2. **Quality Score:** Receive a numeric rating (1–10) assessing how effectively the initial response addresses your input.\n\n"
        "Please note: This virtual assistant is **not** a substitute for professional therapeutic services."
    )


    # Sidebar configuration

    st.sidebar.header("Configuration")

    # OpenAI key
    st.session_state.setdefault("openai_api_key", "")
    openai_api_key_input = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key
    )

    # Check if API key has changed (or entered for first time)
    if openai_api_key_input != st.session_state.openai_api_key:
        st.session_state.openai_api_key = openai_api_key_input.strip()
        if st.session_state.openai_api_key:
            st.session_state.judge_llm = init_judge_llm(st.session_state.openai_api_key)
        else:
            st.session_state.judge_llm = None

    # Initialize judge LLM if not initialized yet but key is provided
    if "judge_llm" not in st.session_state:
        if st.session_state.openai_api_key:
            st.session_state.judge_llm = init_judge_llm(st.session_state.openai_api_key)
        else:
            st.session_state.judge_llm = None

    # Check if judge is usable
    judge_enabled = st.sidebar.checkbox("Use Judge?", value=True)

    if judge_enabled and not st.session_state.judge_llm:
        st.sidebar.warning("Please enter your OpenAI API key to enable the judge.")

    # Conversation state 
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are a caring and empathetic mental health assistant. You provide general "
                    "information, emotional support, and gentle suggestions. You are not a "
                    "replacement for professional healthcare advice."
                )
            }
        ]
    if "main_llm" not in st.session_state:
        st.session_state.main_llm = init_main_llm()
    if "judge_llm" not in st.session_state:
        st.session_state.judge_llm = init_judge_llm(st.session_state.openai_api_key)


    # Chat input
    conversation_placeholder = st.container()
    user_input = st.chat_input("How can I support you today? (Press Enter to submit)")

    # If user typed something, process it
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate main LLM response
        main_response = generate_main_response(
            st.session_state.main_llm,
            st.session_state.messages
        )

        # Initialize defaults
        refined_response, score = main_response, None

        # Only run judge if explicitly enabled and client exists
        if judge_enabled and st.session_state.judge_llm:
            refined_response, score = judge_refine_and_score(
                st.session_state.judge_llm,
                user_input,
                main_response
            )
        elif judge_enabled and not st.session_state.judge_llm:
            st.warning("Judge enabled but OpenAI API key missing or invalid.")

        # Store assistant message conditionally
        assistant_message = {
            "role": "assistant",
            "content": main_response
        }

        # Only add refined response and score if judge is enabled
        if judge_enabled and score is not None:
            assistant_message["refined"] = refined_response
            assistant_message["score"] = score

        st.session_state.messages.append(assistant_message)


    # Display the conversation
    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "system":
            continue

        with conversation_placeholder:
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(msg["content"])
            else:  # assistant
                with st.chat_message("assistant", avatar="🩺"):
                    st.write(msg["content"])  # Always show original response

                    # Only show refined and score if judge is enabled
                    if judge_enabled and "refined" in msg and "score" in msg:
                        st.markdown("**Refined Response:**")
                        st.write(msg["refined"])
                        st.write(f"**Judge's Score:** {msg['score']}")


    # Footer / Additional Info 
    st.markdown("<hr style='margin-top:2rem;'>", unsafe_allow_html=True)
    st.caption(
        "This application is **experimental**. Use it responsibly. If you find any content "
        "inappropriate or inaccurate, please disregard it and consult a professional. "
        "Always prioritize your well-being and safety."
    )

if __name__ == "__main__":
    main()
