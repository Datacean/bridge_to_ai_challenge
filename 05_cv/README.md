# Bridge To AI Challenge

Code of the challenge.

Quick start (conda + pip)
-------------------------

1. Create and activate a conda environment:

```bash
conda create -n btaic python=3.12 -y
conda activate btaic
```

2. Install Python dependencies with pip:

```bash
pip install -r requirements.txt
```

## Image Generation (`content_generator.py`)

This directory includes `content_generator.py`, a command-line script for generating images from text prompts using OpenAI's text-to-image models (like DALL-E).

### Features

- **Text-to-Image**: Creates images based on a descriptive prompt.
- **Batch Generation**: Generate multiple images (`--count`) from a single prompt.
- **Customization**: Specify image size (`--size`) and the OpenAI model (`--model`).
- **Simple CLI**: Easy-to-use command-line interface for generating content.

### How to Use

1.  **Set Your API Key**:
    Create a file named `.env` in the `05_cv` directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```

2.  **Run the Script**:
    Execute the script from your terminal, providing a text prompt.

    **Example:**
    ```bash
    python content_generator.py --prompt "A sleek, black and gold running shoe with neon blue accents, studio lighting" --count 2
    ```
    Images will be saved in the `05_cv/outputs` directory by default.
