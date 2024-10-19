import streamlit as st
import re
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec  # Use Pinecone class and ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyMuPDFLoader
from youtube_transcript_api import YouTubeTranscriptApi
import tempfile

# For reranking 
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Load environment variables
load_dotenv()

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
                .top-margin{
                    margin-top: 4rem;
                    margin-bottom:2rem;
                }
                .block-button{
                    padding: 10px; 
                    width: 100%;
                    background-color: #c4fcce;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

class DataProcessing:

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)

        # Initialize BM25 Encoder and Pinecone in session state if not already done
        if 'bm25' not in st.session_state:
            st.session_state.bm25 = BM25Encoder().default()
        
        # Initialize Pinecone client using the new class-based API
        if 'pinecone_client' not in st.session_state:
            st.session_state.pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        self.index_name = "rag-finance"
        self.embedding_model = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

        # Store the client of OpenAI in Streamlit session state
        if 'openai_client' not in st.session_state:
            st.session_state.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.client = st.session_state.openai_client

        # Check if the index exists before using it
        if 'pinecone_index' not in st.session_state:
            # Check if index exists; if not, create it
            if self.index_name not in st.session_state.pinecone_client.list_indexes().names():
                st.session_state.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=1536,  # Assuming vector dimension is 1536
                    metric='dotproduct',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            st.session_state.pinecone_index = st.session_state.pinecone_client.Index(self.index_name)

        self.index = st.session_state.pinecone_index
        self.bm25 = st.session_state.bm25

        # Initialize the reranker and retriever
        self.model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        self.compressor = CrossEncoderReranker(model=self.model, top_n=4)
        
        # The retriever will be initialized later when the corpus is available
        self.retriever = None

    def initialize_compression_retriever(self):
        """Initialize compression retriever using the base retriever"""
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.retriever
        )

    def get_youtube_id(self, url):
        regex = (
            r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
        match = re.match(regex, url)
        return match.group(6) if match else None

    def process_youtube(self, youtube_id, corpus):
        try:
            result = YouTubeTranscriptApi.get_transcript(youtube_id)
            yt_captions = ""
            for item in iter(result):
                yt_captions += item['text'] + " "
        except (YouTubeTranscriptApi.TranscriptsDisabled, YouTubeTranscriptApi.NoTranscriptFound):
            st.error("Transcript is disabled or not found for this video.")

        chunks = self.splitter.create_documents([yt_captions])
        for chunk in chunks:
            corpus.append(chunk.page_content)

    def process_pdf(self, pdf_file, corpus):
        if pdf_file is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Use PyMuPDFLoader to process the file
                loader = PyMuPDFLoader(tmp_file_path)
                data = loader.load()

                os.unlink(tmp_file_path)

                content = ""
                for doc in data:
                    content += doc.page_content
                chunks = self.splitter.create_documents([content])
                for chunk in chunks:
                    corpus.append(chunk.page_content)
            except Exception as e:
                st.error(f"An error occurred while processing the PDF: {str(e)}")

    def process_whatsapp(self, text, corpus):
        """
        Process a conversation text by removing timestamps and '<Media omitted>' lines, then split the text into chunks and add them to the corpus.

        Args:
            text (str): The conversation text to process.
            corpus (list): The list of documents to which the processed chunks will be added.
        """
        # Regex to match the time only, keeping the date intact
        pattern = r"(?<=^\d{2}/\d{2}/\d{2}), \d{1,2}:\d{2}(?:\u202f)?(?:am|pm) -"

        # Process each line
        processed_lines = []
        for line in text.splitlines():
            # Remove only the time from the beginning, keeping the date
            cleaned_line = re.sub(pattern, '', line).strip()

            # Remove any line containing '<Media omitted>'
            if '<Media omitted>' not in cleaned_line and cleaned_line:
                processed_lines.append(cleaned_line)

        complete_text = "\n".join(processed_lines)
        chunks = self.splitter.create_documents([complete_text])
        for chunk in chunks:
            corpus.append(chunk.page_content)


    def improve_query(self, user_query):
        """
        This function takes a user's raw query in Hinglish (Hindi + English) and improves it to make it clearer, more detailed, and optimal for document retrieval systems.
        It uses the OpenAI API to generate a response based on the context and query, refining the query by adding clarity, removing ambiguity, and adding any missing context.
        The intent of the question remains the same while enhancing its clarity.

        Args:
            user_query (str): The user's raw query in Hinglish (Hindi + English).

        Returns:
            str: The optimized query for document retrieval systems.
        """

        # Define the prompt template with Hinglish support
        prompt_template = f"""
        ### Task Description:
        You are an assistant that specializes in improving user questions to make them clearer, more detailed, and optimal for document retrieval systems. 
        You will be provided with a user's raw query in Hinglish (Hindi + English). Your task is to rephrase the query by making it more specific, removing ambiguity, and adding any missing context that can help the system provide the best possible answer. 
        Make sure the intent of the question remains the same while enhancing its clarity.

        ### Instructions:
        1. Take the user's raw query, which may be in Hinglish (Hindi + English).
        2. If the query is vague, refine it by adding clarity and ensuring all parts of the question are meaningful.
        3. Make the query as specific as possible without changing the original intent.
        4. If necessary, add any additional context that could assist the retrieval system in providing the best response.

        ### Input:
        User's Raw Query: "{user_query}"

        ### Output:
        Query:
        """

        # Call the OpenAI API to generate the response based on context and query
        response = self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Replace this with the actual model if needed
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_template}
                    ],
                    temperature=0.7,
                    top_p=1,
                    n=1
                )
        
        # Extract the optimized query from the response
        optimized_query = response.choices[0].message.content.strip()
        return optimized_query

    def generate_response(self, context_data, query):
        """
        Generate a prompt for the GPT-4-mini model to provide a clear, accurate, and well-structured response
        based on the query and the given context data, with a heading "AI Response:".

        Args:
            context_data (str): The context data that serves as input for the model.
            query (str): The specific query provided by the user.

        Returns:
            str: The generated response from GPT-4-mini.
        """

        # Create a prompt that instructs the AI to respond with a heading "AI Response:" before the actual output
        prompt_template = f"""
        You are an intelligent assistant (GPT-4-mini) tasked with providing a detailed, accurate, and concise response
        based on the following context and query:

        Context:
        \"{context_data}\"

        Query:
        \"{query}\"

        Guidelines:
        1. Begin your response with the heading: **AI Response:**
        2. Carefully analyze both the context and the query to generate a comprehensive and relevant response.
        3. Ensure that the response is well-structured, uses clear language, and is factually accurate.
        4. If the context or query is unclear or ambiguous, politely ask the user for clarification or to rephrase the question.

        Example response formats:
        - "AI Response: Based on the context provided and your query, here is the relevant information..."
        - "AI Response: The context and query provided are unclear. Could you please rephrase the question or provide more specific information?"

        Please provide your response below.
        """

        try:
            # Call the OpenAI API to generate the response based on context and query
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Replace this with the actual model if needed
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_template}
                ],
                temperature=0.7,
                top_p=1,
                n=1
            )

            # Extract the generated response text
            generated_response = response.choices[0].message.content.strip()
            return generated_response

        except Exception as e:
            return f"An error occurred while generating the response: {str(e)}"


def chat_with_docs():
    dp_obj = DataProcessing()
    corpus = []

    # File upload options for PDF, WhatsApp, and YouTube transcript
    st.title("Chat with your Data")
    st.write("**********")
    pdf_tab, whatsapp_tab, youtube_tab = st.columns(spec=(1,1,1), gap="large")
    with pdf_tab:
        pdf_upload = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False, key="pdf_upload")
        if pdf_upload:
            with st.spinner("Processing PDF..."):
                dp_obj.process_pdf(pdf_upload, corpus)
            st.success("PDF document uploaded and setup")
    with whatsapp_tab:
        chat_upload = st.file_uploader("Upload the Whatsapp chat text file", type="txt", accept_multiple_files=False, key="chat_upload")
        if chat_upload is not None:
            with st.spinner("Processing WhatsApp chat..."):
                raw_text = chat_upload.read().decode("utf-8")
                dp_obj.process_whatsapp(raw_text, corpus)
            st.success("WhatsApp chat uploaded and setup")
    with youtube_tab:
        youtube_link = st.text_input("Enter a YouTube Link")
        youtube_id = dp_obj.get_youtube_id(youtube_link)
        if youtube_id:
            with st.spinner("Processing YouTube transcript..."):
                dp_obj.process_youtube(youtube_id, corpus)
            st.success("YouTube transcript uploaded and setup")

    if corpus:

        dp_obj.bm25.fit(corpus)
        dp_obj.bm25.dump(r"artifacts/bm25_values.json")
        dp_obj.bm25 = BM25Encoder().load(r"artifacts/bm25_values.json")
        dp_obj.retriever = PineconeHybridSearchRetriever(
            embeddings=dp_obj.embedding_model, 
            sparse_encoder=dp_obj.bm25, 
            index=dp_obj.index,
            top_k=5
        )
        
        # Add the corpus to the retriever and initialize the compression retriever
        dp_obj.retriever.add_texts(corpus)
        dp_obj.initialize_compression_retriever()

        # Use st.chat_message for displaying messages
        st.write("**********")
        with st.chat_message("system"):
            st.write("You can upload documents and ask questions based on the context.")

        # Use st.chat_input to capture the user's query in a chat-like format
        query_input = st.chat_input("Ask your query")
        if query_input:
            with st.spinner("Optimizing your query for the best response..."):
                improved_query = dp_obj.improve_query(query_input)
            with st.chat_message("user"):
                st.write(f"User Query: {query_input}")

            with st.chat_message("system"):
                st.write(f"Improved Query: {improved_query}")


            try:
                sparse_vector = dp_obj.bm25.encode_queries([query_input])
                if not sparse_vector[0]:
                    st.warning("The query resulted in an empty sparse vector. Try a different query.")
                else:
                    # Use the compression retriever to get reranked documents
                    compressed_docs = dp_obj.compression_retriever.invoke(query_input)
                    context_data = ""
                    for doc in compressed_docs:
                        context_data += doc.page_content
                    
                    with st.chat_message("system"):
                        with st.spinner("Generating response..."):
                            response = dp_obj.generate_response(query=improved_query, context_data=context_data)
                        st.write(response)
            except Exception as e:
                st.error(f"An error occurred while processing the query: {str(e)}")
    else:
        st.warning("No documents processed. Please upload or enter content to process.")


chat_with_docs()