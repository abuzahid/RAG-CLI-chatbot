from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from src.vectorstore.store import get_vectorstore, VectorStore
from src.chat.session import ChatSession
from src.config import get_config


class RetrievalChain:
    """RAG-based conversational retrieval chain using LCEL"""

    def __init__(
        self,
        session: Optional[ChatSession] = None,
        vectorstore: Optional[VectorStore] = None,
    ):
        self.session = session or ChatSession.from_config()
        self.vectorstore = vectorstore or get_vectorstore()
        self.config = get_config()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            api_key=self.config.openai_api_key,
        )

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.retrieval.top_k}
        )

    def _format_docs(self, docs):
        """Format retrieved documents into a string"""
        return "\n\n".join(doc.page_content for doc in docs)

    def _get_chat_history(self) -> List[Any]:
        """Convert session history to LangChain messages"""
        messages = []
        for msg in self.session.get_messages():
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def query(self, question: str) -> str:
        """Process a user query and return the response

        Args:
            question: User's question

        Returns:
            AI's response
        """
        # Retrieve relevant documents
        docs = self.retriever.invoke(question)
        context = self._format_docs(docs)
        history = self._get_chat_history()

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.conversation.system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", """Context:
{context}

Question: {question}"""),
        ])

        # Format and invoke
        formatted_prompt = prompt.format_messages(
            context=context,
            question=question,
            history=history
        )

        # Get response
        response = self.llm.invoke(formatted_prompt)
        result = response.content

        # Update session with new messages
        self.session.add_user_message(question)
        self.session.add_ai_message(result)

        return result

    def chat_history(self) -> List[dict]:
        """Get current chat history"""
        return self.session.get_messages()

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.session.clear()


def create_retrieval_chain(
    session: Optional[ChatSession] = None,
    vectorstore: Optional[VectorStore] = None,
) -> RetrievalChain:
    """Convenience function to create a retrieval chain"""
    return RetrievalChain(session=session, vectorstore=vectorstore)
