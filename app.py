from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

base_url = 'http://llama:8000/v1'
model = 'meta/llama-3-8b-instruct'
llm = ChatNVIDIA(base_url=base_url, model=model, temperature=0)

import re
import contractions  # pip install contractions

def normalize_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Expand contractions
    text = contractions.fix(text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

reviews = [
    "I LOVE this product! It's absolutely amazing. ",
    "Not bad, but could be better. I've seen worse.",
    "Terrible experience... I'm never buying again!!",
    "Pretty good, isn't it? Will buy again!",
    "Excellent value for the money!!! Highly recommend."
]

RunnableLambda(normalize_text).batch(reviews)

sentiment_template = ChatPromptTemplate.from_template(
    """In a single word, either 'positive' or 'negative', \
    provide the overall sentiment of the following piece of text: {text}"""
)

sentiment_template.invoke({"text": "i love this product! it is absolutely amazing."})

prep_for_sentiment_template = RunnableLambda(lambda text: {"text": text})
prep_for_sentiment_template.batch(reviews)

parser = StrOutputParser()
sentiment_chain = RunnableLambda(normalize_text) | prep_for_sentiment_template | sentiment_template | llm | parser

print(sentiment_chain.batch(reviews))
