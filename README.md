# AttentionViz: My Interpretability Project

Welcome to **AttentionViz**, a tool I built to explore and visualize how Large Vision-Language Models (LVLMs) "see" and "think". 

This project is designed to peel back the layers of complex AI models like LLaVA-Gemma, allowing us to understand the connection between the visual input (images) and the textual output (answers).

## What Can This Project Do?

*   **Interactive Chat**: You can upload any image and ask questions about it. The model will analyze the image and provide an answer.
*   **Visual Reasoning**: It uses state-of-the-art vision-language models to understand context, objects, and relationships within images.
*   **Interpretability (Under Development)**: The core goal of this project is to visualize the *attention mechanisms*—literally showing you which parts of an image the model focuses on when generating specific words in its response.

## How I Built It

I transformed the original research code into a streamlined, user-friendly **Streamlit** application. This involved:
*   Refactoring the backend to decouple it from legacy UI frameworks.
*   Designing a clean, interactive frontend using Streamlit.
*   Optimizing the codebase for easier deployment and usage.

## Getting Started

### Option 1: Run with Docker (Recommended)

I've containerized the application to make it easy to run without worrying about dependencies.

1.  **Build the Image**:
    ```bash
    docker-compose build
    ```

2.  **Run the App**:
    ```bash
    docker-compose up
    ```

3.  **Access**: Open your browser and go to `http://localhost:8501`.

### Option 2: Run Locally

If you prefer to run it directly on your machine:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App**:
    ```bash
    streamlit run streamlit_app.py
    ```

## Models

The project currently defaults to `Intel/llava-gemma-2b`, a powerful yet efficient model. You can configure other model paths in the sidebar.

---
*Built by Chaitanya Sirivuri*
