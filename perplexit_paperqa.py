# app.py
import streamlit as st
from paperqa import Docs
import os
import tempfile
import PyPDF2

def main():
    st.set_page_config(
        page_title="PaperQA-2 Research Assistant",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 PaperQA-2 Academic Research Assistant")
    st.markdown("""
    ### Powered by PaperQA-2
    Upload your research papers (PDFs) and ask questions about their content!
    """)

    # Initialize session state
    if 'docs' not in st.session_state:
        st.session_state.docs = Docs()

    # Sidebar for document upload
    with st.sidebar:
        st.header("📤 Document Management")
        uploaded_files = st.file_uploader(
            "Upload research papers (PDF)",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        # Add document to PaperQA
                        st.session_state.docs.add(tmp_path, metadata=uploaded_file.name)
                        
                        # Clean up temporary file
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            st.success(f"Successfully processed {len(uploaded_files)} documents!")

    # Main content area
    st.header("💬 Research Query Interface")
    
    # Question input
    question = st.text_input(
        "Enter your research question:",
        placeholder="What is the main finding of the study?",
        help="Ask anything about the uploaded research papers"
    )

    # Answer display
    if st.button("Ask PaperQA-2"):
        if not uploaded_files:
            st.warning("Please upload documents first!")
            return
            
        if not question.strip():
            st.warning("Please enter a question!")
            return

        with st.spinner("Analyzing documents..."):
            try:
                answer = st.session_state.docs.query(question)
                
                # Display main answer
                st.subheader("Answer")
                st.markdown(f"**{answer.answer}**")
                
                # Show context and sources
                with st.expander("Detailed Answer & Sources"):
                    st.markdown(answer.context)
                    st.markdown("### Sources:")
                    for source in answer.sources:
                        st.markdown(f"- {source}")
                        
            except Exception as e:
                st.error(f"Error during query: {str(e)}")

if __name__ == "__main__":
    main()
