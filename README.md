# AttentionViz: Streamlit Edition

An interpretability tool for Large Vision-Language Models (LVLMs), adapted for Streamlit.

This project allows you to visualize attention maps and relevancy scores for models like LLaVA-Gemma, helping to understand how these models process visual and textual information.

## Features

- **Chat Interface**: Interact with the model using text and images.
- **Attention Visualization**: (Coming soon to UI) Visualize attention heads and layers.
- **Relevancy Maps**: (Coming soon to UI) See which parts of the image are most relevant to the generated text.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App**:
    ```bash
    streamlit run streamlit_app.py
    ```

## Usage

- Upload an image using the sidebar.
- Enter a prompt in the chat input.
- Adjust generation parameters (Temperature, Top P, etc.) in the sidebar.
- Explore the model's responses.

## Models Supported

- `Intel/llava-gemma-2b` (Default)
- Other LLaVA variants (configurable via sidebar)

## Acknowledgements

Based on the original [AttentionViz](https://github.com/IntelLabs/multimodal_cognitive_ai/tree/main/lvlm_interpret) work.
