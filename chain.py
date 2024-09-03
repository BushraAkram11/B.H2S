from typing import List, Tuple
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.load import dumps
from pinecone import Pinecone
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_pinecone import PineconeVectorStore

# HuggingFace embeddings and Pinecone setup
embeddings = HuggingFaceEmbeddings()

# Sidebar: Get user's Google API key input
user_google_api_key = st.session_state.get("user_google_api_key", "")

# Default Google API key (replace with your actual default key)
default_google_api_key = "YOUR_DEFAULT_GOOGLE_API_KEY"

# Use user's API key if available, otherwise use the default key
google_api_key = user_google_api_key if user_google_api_key else default_google_api_key

pinecone_api_key = "ae70f7c9-557a-4a1d-b944-5ecc208513ad"
pc = Pinecone(api_key=pinecone_api_key)

index_name = "hopetoskill"
index = pc.Index(index_name)

retriever = PineconeVectorStore(
    embedding=embeddings, index=index
).as_retriever()

# Condense a chat history and follow-up question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# RAG answer synthesis prompt
template = """You are an assistant for question-answering tasks for the Hope To Skill AI training initiative. Use the following retrieved context to answer the user's question:
Question: {question} 
Context: {context} 
Answer:"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

def _combine_documents(docs):
    return "\n\n".join(set(dumps(doc) for doc in docs))

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# User input class
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str

_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, api_key=google_api_key)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(itemgetter("question")),
)

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever | _combine_documents,
    }
)
chain = (
    (_inputs | ANSWER_PROMPT | ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=google_api_key, temperature=0.4) | StrOutputParser())
    .with_types(input_type=ChatHistory)
    .with_fallbacks(
        [
            RunnableLambda(
                lambda prompt: "There is an error while generating your response. Please try again."
            )
        ]
    )
)
