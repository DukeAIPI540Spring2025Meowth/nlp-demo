from openai import OpenAI
import json
import streamlit as st
from prompts import criteria_based_evaluation_prompt
import sys
import os
# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.naive.model import main as naive_main
#from scripts.etl.extract import load_test_data


def initialize_session_states():
    if 'judge_model' not in st.session_state:
        st.session_state.judge_model = "gpt-4o"

    if 'crf_response' not in st.session_state:
        st.session_state.crf_response = ""

    if 'rbe_response' not in st.session_state:
        st.session_state.rbe_response = ""


# Model selection
available_models = ["gpt-4", "gpt-3.5-turbo", naive_main]

if 'api_key' not in st.session_state:
        st.session_state.api_key = None

# Initialize OpenAI client
if st.session_state.api_key:
    client = OpenAI(api_key=st.session_state.api_key)

def get_response(user_prompt, model, json_format=True):
    # Initialize OpenAI client
    if st.session_state.api_key:
        client = OpenAI(api_key=st.session_state.api_key)

        if json_format:
            completion = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': user_prompt}],
                response_format={"type":"json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': user_prompt}],
            )
            return completion.choices[0].message.content
        
    else:
         st.error("Please provide API Key.")

def evaluation_by_criteria_ref_free():
    
    st.subheader("Criteria based Reference Free Evaluation")
    # Input prompt
    prompt = st.text_area("Enter your prompt:")
    criteria_list = ("Technical Accuracy, Strucural Adherence, Empathetic Tone, Intervention depth, Clinical safety").split(',')
    
    # Model selection for comparison
    st.subheader("Select Model to Evaluate")  
    model = st.selectbox("Select Model", available_models, key="model")

    if prompt and st.button("Generate"):
        # Get responses
        with st.spinner("Generating response ..."):
            st.session_state.crf_response = get_response(prompt, model, json_format=False)

    if st.session_state.crf_response:
        st.write(f"**{model} Response:**")
        st.write(st.session_state.crf_response)
    
    if st.session_state.crf_response and st.button("Evaluate"):
        # Detailed evaluation section
        st.subheader("Detailed Evaluation for each criteria")
        for cri in criteria_list:
            with st.spinner(f"Evaluating Responses for {cri}..."):
                eval_result = get_response(criteria_based_evaluation_prompt.format(criteria=cri, response=st.session_state.crf_response), st.session_state.judge_model)
            
            if eval_result:
                # Display detailed evaluation
                with st.expander(f"{cri.capitalize()} Analysis"):
                    eval_result
                    # st.write(f"**Score:** {eval_result['score']}/5")
                    # st.write(f"**Detailed Explanation:** {eval_result['explanation']}")
                    
                    # st.write("\n**Strengths:**")
                    # for strength in eval_result['strengths']:
                    #     st.write(f"- {strength}")
                        
                    # st.write("\n**Areas for Improvement:**")
                    # for improvement in eval_result['improvements']:
                    #     st.write(f"- {improvement}")
                        
                    # st.write(f"\n**Key Observations:** {eval_result['observations']}")

# def reference_based_evaluation():
#     """Evaluate responses against a reference/ground truth answer"""
#     st.subheader("Reference-Based Evaluation")
    
#     # Input fields
#     prompt = st.text_area("Enter your prompt:", value="")
#     reference_answer = st.text_area("Enter reference answer:", value="")
#     model = st.selectbox("Select Model", available_models, key="ref_model")
    
#     if prompt and reference_answer and st.button("Generate"):
#         with st.spinner("Generating response..."):
#             st.session_state.rbe_response = get_response(prompt, model, json_format=False)
            
#     if st.session_state.rbe_response:
#         st.write(f"**{model} Response:**")
#         st.write(st.session_state.rbe_response)
        
#     if st.session_state.rbe_response and st.button("Evaluate"):
#         with st.spinner("Generating evaluation..."):
#             eval_result = get_response(reference_based_eval_prompt.format(reference_answer=reference_answer, model_response=st.session_state.rbe_response), st.session_state.judge_model)
        
#         if eval_result:
#             st.write(f"**Score:** {eval_result['score']}/10")
#             st.write(f"**Detailed Explanation:** {eval_result['explanation']}")

def main():

    initialize_session_states()

    st.title("NLP AI Therapist Evaluation: LLM-As-a-Judge")

    evaluation_methods = {
        "Reference-Free Criteria Evaluation": evaluation_by_criteria_ref_free,
        #"Reference-based Evaluation": reference_based_evaluation,
    }

    st.session_state.api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    
    method = st.sidebar.selectbox(
        "Select Evaluation Method",
        list(evaluation_methods.keys())
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Method Description")
    
    descriptions = {
        "Reference-Free Criteria Evaluation": "Evaluate as per a defined criteria without ground truth",
        #"Reference-based Evaluation": "Evaluate responses against a reference/ground truth answer",
    }
    
    if not st.session_state.api_key:
        st.error("Please provide an API key.")
    
    else:
        st.sidebar.write(descriptions[method])
        
        evaluation_methods[method]()
if __name__ == "__main__":
    main()