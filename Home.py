from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from more_itertools import chunked
import requests
import PyPDF2
from io import BytesIO

# --- Logo ---
st.set_page_config(page_title="Hope_To_Skill AI Chatbot", page_icon=":robot_face:")
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        font-size: 24px;
        color: gray;
        margin-bottom: 20px;
    }
    .stTextInput input {
        border: 2px solid black !important;
        border-radius: 5px;
        padding: 10px;
        box-sizing: border-box;
    }
    .stTextInput {
        margin-bottom: 20px;
    }
    </style>
    <div class="title">Hope To Skill AI-Chatbot</div>
    <div class="subtitle">Welcome to Hope To Skill AI Chatbot, How can I help you today?</div>
    """,
    unsafe_allow_html=True
)

# Sidebar with logo and Google API Key input
with st.sidebar:
    st.image("https://yt3.googleusercontent.com/G5iAGza6uApx12jz1CBkuuysjvrbonY1QBM128IbDS6bIH_9FvzniqB_b5XdtwPerQRN9uk1=s900-c-k-c0x00ffffff-no-rj", width=290)
    st.sidebar.subheader("Google API Key")
    user_google_api_key = st.sidebar.text_input(
        "🔑 Enter your Google Gemini API key to Ask Questions",
        type="password",
        key="password_input",
        help="Enter your Google API key here",
        placeholder="Your Google API Key"
    )
    st.session_state.google_api_key = user_google_api_key or ""  # Default key

# Function to extract text from PDF
def extract_text_from_pdf(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    with BytesIO(response.content) as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Extract text from the provided PDF URL
pdf_url = "https://drive.google.com/uc?export=download&id=12GSfTxJMpqtGKi5GWFZfIhBWRJ38nnVm"
pdf_text = extract_text_from_pdf(pdf_url)

# Ensure the user has provided an API key
if not st.session_state.google_api_key:
    st.warning("Please enter your Google API key in the sidebar to use the chatbot.")
else:
    prompt = ChatPromptTemplate(
        messages=[
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I assist you today?")

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                _chat_history = st.session_state.langchain_messages[1:40]
                _chat_history_transform = list(
                    chunked([msg.content for msg in _chat_history], n=2)
                )

                from chain import chain

                # Pass the Google API key from session state
                response = chain.stream(
                    {
                        "question": prompt,
                        "chat_history": _chat_history_transform,
                        "google_api_key": st.session_state.google_api_key,
                        "pdf_text": pdf_text  # Pass PDF text as part of the context
                    }
                )

                for res in response:
                    full_response += res or ""
                    message_placeholder.markdown(full_response + "|")
                    message_placeholder.markdown(full_response)

                msgs.add_user_message(prompt)
                msgs.add_ai_message(full_response)

            except Exception as e:
                st.error(f"An error occurred: {e}")
