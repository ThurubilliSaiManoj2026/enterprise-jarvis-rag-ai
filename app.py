import streamlit as st
from rag import jarvis_answer

st.set_page_config(page_title="Enterprise Jarvis", page_icon="🤖")
st.title("Enterprise Jarvis 🤖")

query = st.text_input("Ask Jarvis")

if query:
    with st.spinner("Thinking..."):
        response = jarvis_answer(query)
    st.write(response)
