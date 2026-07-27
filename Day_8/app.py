import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv


import google.generativeai as genai

genai.configure(api_key="AQ.Ab8RN6Km7qUxv6m1NHNym8XWEomawbh_EImsQB2resnzr-bXlA")

for model in genai.list_models():
    print(model.name)

# Load API Key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

#model = genai.GenerativeModel("gemini-1.5-flash")
model = genai.GenerativeModel("gemini-2.5-flash")

st.title("My First LLM Chatbot")

user_input = st.text_input("Ask me anything")

if st.button("Send"):

    if user_input:

        response = model.generate_content(user_input)

        st.write("### Response")

        st.write(response.text)