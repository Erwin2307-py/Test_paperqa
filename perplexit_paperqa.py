import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def main():
    st.title("📄 PaperQA2 - Dokumentenbasierte Frage-Antwort")
    st.markdown("Laden Sie ein Dokument hoch und stellen Sie Fragen dazu.")

    # OpenAI-API-Key Eingabe
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    # Dokumenten-Upload
    uploaded_file = st.file_uploader("PDF/TXT-Datei hochladen", type=["pdf", "txt"])
    
    if uploaded_file and openai_api_key.startswith("sk-"):
        # Dokument verarbeiten
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(uploaded_file.read().decode())
        
        # Embeddings und Vektordatenbank
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        docsearch = Chroma.from_texts(texts, embeddings)
        
        # QA-Kette initialisieren
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(api_key=openai_api_key),
            chain_type="stuff",
            retriever=docsearch.as_retriever()
        )
        
        # Frage-Eingabe
        query = st.text_input("Stellen Sie Ihre Frage zum Dokument:")
        if query:
            response = qa.run(query)
            st.subheader("Antwort:")
            st.write(response)

if __name__ == "__main__":
    main()
