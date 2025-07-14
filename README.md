Sentiment Analysis using LLaMA3 + LangChain


This project performs sentiment analysis on user reviews using a locally hosted LLaMA3-8B-Instruct model via LangChain's NVIDIA integration.

Features

-Uses ChatNVIDIA for LLaMA3 inference.
-Text normalization: lowercasing, contractions, whitespace.
-Few-shot prompt template for accurate sentiment detection.
-Supports batch processing of multiple reviews.

Requirements

pip install langchain contractions

