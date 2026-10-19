# Sentence Analyzer

A Streamlit app that analyzes one or more sentences using LangChain and a Groq-hosted language model. Enter sentences on separate lines and the app returns three analyses for each one.

## Features

For each non-empty input line, the app provides:

- **Sentiment**: positive, negative, or neutral.
- **Main topic**: a concise description of the sentence's main topic.
- **Follow-up question**: a question that could help explore the text further.

The app uses LangChain runnables to prepare inputs, run the analysis chains in parallel, and format the results.

## Requirements

- Python 3.9 or a compatible Python version
- A Groq API key

## Setup

1. Clone or download this repository and open its folder.
2. Create and activate a virtual environment (recommended).
3. Install the dependencies:

   ```bash
   pip install streamlit python-dotenv langchain-core langchain-groq
   ```

4. Create a `.env` file in the project folder:

   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter one or more sentences in the text area, using one sentence per line.
2. Click **Analyze**.
3. Review the sentiment, main topic, and follow-up question shown for each sentence.

## Model

The app is configured to use `openai/gpt-oss-20b` through Groq.

## Configuration and security

Keep your API key private. Do not commit the `.env` file to a public repository. The app expects the environment variable `GROQ_API_KEY`.
