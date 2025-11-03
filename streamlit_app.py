import streamlit as st
import argparse
import os
import torch
from PIL import Image
import app_utils as utils

# Page config
st.set_page_config(page_title="AttentionViz", layout="wide")

# Sidebar for configuration
st.sidebar.title="Configuration"

# Model arguments (simulating argparse)
class Args:
    def __init__(self):
        self.model_name_or_path = st.sidebar.text_input("Model Name/Path", value="Intel/llava-gemma-2b")
        self.load_4bit = st.sidebar.checkbox("Load 4-bit", value=False)
        self.load_8bit = st.sidebar.checkbox("Load 8-bit", value=False)
        # Add other args as needed by utils_model.get_processor_model
        # The original app.py had these.
        self.device_map = "auto"

args = Args()

# Initialize model
if "model_initialized" not in st.session_state:
    with st.spinner("Loading model..."):
        try:
            utils.initialize_model(args)
            st.session_state.model_initialized = True
            st.success("Model loaded!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

# Session State for Chat
if "state" not in st.session_state:
    st.session_state.state = utils.State()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main UI
st.title("AttentionViz: My Interpretability Project")
st.markdown("### Exploring the Inner Workings of Vision-Language Models")
st.markdown("Built by **Chaitanya Sirivuri**")

# Image Upload
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)

# Chat Interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image"):
            st.image(message["image"], caption="User Image", width=300)

# User Input
if prompt := st.chat_input("Ask something about the image..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        if image:
            st.image(image, width=300)
    
    st.session_state.chat_history.append({"role": "user", "content": prompt, "image": image})
    
    # Process with model
    # We need to update the 'state' object that utils expects
    # The utils.add_text and utils.lvlm_bot seem to mutate the state object
    
    # Update internal state
    st.session_state.state = utils.add_text(st.session_state.state, prompt, image, "pad") # 'pad' is a guess for image_process_mode, need to check default
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Parameters
            temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
            top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.7)
            max_new_tokens = st.sidebar.slider("Max New Tokens", 1, 1024, 512)
            
            try:
                st.session_state.state = utils.lvlm_bot(st.session_state.state, temperature, top_p, max_new_tokens)
                response_text = st.session_state.state.messages[-1]["content"]
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                
                # Show Attention/Relevancy if available
                # The utils save these to temp files and store paths in state
                if st.session_state.state.attention_key:
                    st.info(f"Attention data saved to: {st.session_state.state.attention_key}")
                    # Here we could load and visualize if we ported the plotting logic
                    
            except Exception as e:
                st.error(f"Error generating response: {e}")

# Reset Button
if st.sidebar.button("Clear History"):
    st.session_state.state = utils.clear_history()
    st.session_state.chat_history = []
    st.rerun()
