import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Streamlit page
st.set_page_config(page_title="Sentence Analyzer")
st.title("Sentence Analyzer with Runnables")
st.write("Analyze one sentence or multiple sentences using LangChain + Groq.")

# Load API key
api_key = os.getenv("GROQ_API_KEY")

# User input
user_input = st.text_area(
    "Enter one sentence or multiple sentences (one per line):",
    height=200,
    placeholder="Example:\nI love learning AI.\nThe weather is terrible today."
)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter at least one sentence.")
    elif not api_key:
        st.error("GROQ_API_KEY not found. Check your .env file.")
    else:
        # Split input into statements
        statements = [line.strip() for line in user_input.splitlines() if line.strip()]

        # Initialize model
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=api_key
        )

        # Output parser
        parser = StrOutputParser()

        # Prompt templates
        sentiment_template = ChatPromptTemplate.from_template(
            """In a single word, either 'positive', 'negative', or 'neutral',
provide the overall sentiment of the following piece of text: {text}"""
        )

        main_topic_template = ChatPromptTemplate.from_template(
            """Identify and state, as concisely as possible, the main topic
of the following piece of text. Only provide the main topic and no other helpful comments.
Text: {text}"""
        )

        followup_template = ChatPromptTemplate.from_template(
            """What is an appropriate and interesting follow-up question that would help
me learn more about the provided text? Only supply the question.
Text: {text}"""
        )

        # Custom runnable: convert input string -> {"text": ...}
        prep_for_template = RunnableLambda(lambda text: {"text": text})

        # 3 chains
        sentiment_chain = sentiment_template | llm | parser
        main_topic_chain = main_topic_template | llm | parser
        followup_chain = followup_template | llm | parser

        # Custom runnable: keep original statement
        statement_runnable = RunnableLambda(lambda x: x["text"])

        # Parallel chain
        parallel_chain = RunnableParallel({
            "sentiment": sentiment_chain,
            "main_topic": main_topic_chain,
            "followup": followup_chain,
            "statement": statement_runnable
        })

        # Output formatter
        output_formatter = RunnableLambda(lambda responses: {
            "statement": responses["statement"],
            "sentiment": responses["sentiment"],
            "main_topic": responses["main_topic"],
            "followup": responses["followup"]
        })

        # Full chain
        chain = prep_for_template | parallel_chain | output_formatter

        # Batch process statements
        formatted_outputs = chain.batch(statements)

        # Display results
        st.subheader("Results")

        for result in formatted_outputs:
            st.markdown(f"**Statement:** {result['statement']}")
            st.write(f"**Overall sentiment:** {result['sentiment']}")
            st.write(f"**Main topic:** {result['main_topic']}")
            st.write(f"**Follow-up question:** {result['followup']}")
            st.markdown("---")

       