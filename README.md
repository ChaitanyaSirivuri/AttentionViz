# AttentionViz: Interpretability for VLMs

Welcome to **AttentionViz**, a tool designed to explore and visualize how Large Vision-Language Models (LVLMs) "see" and "think".

Inspired by **GRAD-CAM**, which provides visual explanations for CNNs, this project applies similar interpretability concepts to Vision-Language Models using **attention mechanisms**. It is designed to peel back the layers of complex AI models like LLaVA-Gemma, allowing users to understand the connection between the visual input (images) and the textual output (answers).

## What Can This Project Do?

*   **Interactive Chat**: Upload any image and ask questions about it. The model will analyze the image and provide an answer.
*   **Visual Reasoning**: Uses state-of-the-art vision-language models to understand context, objects, and relationships within images.
*   **Interpretability (Under Development)**: The core goal of this project is to visualize the *attention mechanisms*—literally showing which parts of an image the model focuses on when generating specific words in its response.

## Implementation

The original research code has been transformed into a streamlined, user-friendly **Streamlit** application. This involved:
*   Refactoring the backend to decouple it from legacy UI frameworks.
*   Designing a clean, interactive frontend using Streamlit.
*   Optimizing the codebase for easier deployment and usage.

## Getting Started

### Option 1: Run with Docker (Recommended)

The application has been containerized to make it easy to run without worrying about dependencies.

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

To run it directly on a local machine:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App**:
    ```bash
    streamlit run streamlit_app.py
    ```

## Models

The project currently defaults to `Intel/llava-gemma-2b`, a powerful yet efficient model. Other model paths can be configured in the sidebar.

---
*Built by Chaitanya Sirivuri*
