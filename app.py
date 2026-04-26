import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv(dotenv_path=".env")

st.set_page_config(page_title="Sentence Analyzer")
st.title(" Sentence Analyzer App")
st.write("Analyze one sentence or multiple sentences using LangChain + Groq.")

api_key = os.getenv("GROQ_API_KEY")

user_input = st.text_area(
    "Enter one sentence or multiple sentences (one per line):",
    height=200,
    placeholder="Example:\nI love learning AI.\nThe weather is depressing today."
)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter at least one sentence.")
    elif not api_key:
        st.error("GROQ_API_KEY not found. Check your .env file.")
    else:
        statements = [line.strip() for line in user_input.splitlines() if line.strip()]

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key
        )

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

        sentiment_prompts = [sentiment_template.invoke({"text": statement}) for statement in statements]
        main_topic_prompts = [main_topic_template.invoke({"text": statement}) for statement in statements]
        followup_prompts = [followup_template.invoke({"text": statement}) for statement in statements]

        sentiments = llm.batch(sentiment_prompts)
        main_topics = llm.batch(main_topic_prompts)
        followups = llm.batch(followup_prompts)

        st.subheader("Results")

        for statement, sentiment, main_topic, followup in zip(statements, sentiments, main_topics, followups):
            st.markdown(f"**Statement:** {statement}")
            st.write(f"**Overall sentiment:** {sentiment.content}")
            st.write(f"**Main topic:** {main_topic.content}")
            st.write(f"**Follow-up question:** {followup.content}")
            st.markdown("---")