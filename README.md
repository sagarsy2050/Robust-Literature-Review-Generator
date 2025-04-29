📚 Robust Literature Review Generator

A safe, lightweight, and CPU-compatible Streamlit application that automates the process of literature review using Google search results, Arxiv papers, and GPT-2.
🚀 Features

    🔍 Google Search & Arxiv Integration
    Fetches and parses top search results and scholarly papers relevant to your topic.

    🧠 GPT-2 Based Text Generation
    Synthesizes a formal literature review with safety checks and context truncation.

    🛡️ Safety-Oriented Design
    Input validation, request timeouts, HTML parsing safeguards, and max content limits built-in.

    📄 Review Report
    Generates a structured review with references and academic tone (optionally including Hayek).

    🖥️ Runs on CPU
    Designed to be usable without requiring a GPU.

🛠️ Installation

# Clone the repository
git clone https://github.com/sagarsy2050/lit-review-app.git
cd lit-review-app

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

📦 Requirements
    Python 3.8+
    Streamlit
    transformers
    torch (CPU version)
    huggingface_hub
    arxiv
    beautifulsoup4
    requests

You can install all dependencies using:
pip install -r requirements.txt

💡 Usage

To run the app locally:
streamlit run app.py

Then open the link in your browser (http://localhost:8501).
🧩 How It Works
    User Input: A topic is entered via the UI.
    Agents Execute:
        Google Search Agent scrapes and summarizes results.
        Arxiv Agent retrieves academic paper metadata.

    Report Agent: GPT-2 synthesizes a formal review based on inputs.
    Output: Results and report are displayed in a safe, user-friendly layout.

🔐 Safety Measures

    Truncation of prompts and outputs to avoid overloading the model.
    Input validation to protect against empty or malformed queries.
    Rate limiting (via time.sleep) for web scraping.
    Graceful error handling across all stages.

🧪 Example Topics

    "Quantum computing in cryptography"
    "Climate change impact on agriculture"
    "Applications of game theory in economics"


🧠 Note on GPT-2

GPT-2 is used for demonstration. For production or higher quality outputs, consider fine-tuned or more advanced models like GPT-3.5 or Claude.
