📄 Multi-PDF Chatbot
A web app that lets you chat with multiple PDFs at once.
Built using LangChain, Google Gemini-1.5-pro(primary), and Gemini-1.5-flash (fallback).

🚀 Live Demo
🔗 https://multi-pdfchatbot2024.streamlit.app/

✨ Features
1. Upload and chat with multiple PDF files simultaneously.
2. Uses Google Gemini (gemini-1.5-pro by default).
3. Automatically switches to Gemini-1.5-flash if the quota is exhausted.
4. Answers questions by summarizing and searching across documents.

🛠️ Tech Stack
Python + Streamlit
LangChain for document retrieval & question answering
Google Generative AI API (Gemini)

⚠️ Notes
1. Quota limits: If Gemini-1.5-pro quota exhausted, the model is changed to Gemini-1.5-flash.
2. PDF size: Very large PDFs may take longer to process.
