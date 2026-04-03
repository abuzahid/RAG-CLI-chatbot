from langchain_core.prompts import PromptTemplate
from src.config import get_config


def get_system_message() -> str:
    """Get system message from config"""
    config = get_config()
    return config.conversation.system_message.strip()


def get_qa_prompt() -> PromptTemplate:
    """Get the QA prompt template for RAG chain"""

    template = """{system_message}

Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, say so honestly.
Don't make up information.

Context:
{context}

Question: {question}

Helpful Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        partial_variables={"system_message": get_system_message()}
    )
