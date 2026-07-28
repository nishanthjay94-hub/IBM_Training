import os
import tempfile

import streamlit as st
import numpy as np
import faiss

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


# 1. LOAD ENVIRONMENT VARIABLES

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# 2. PAGE CONFIGURATION

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# 3. TITLE

st.title(" PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its content."
)


# 4. PDF UPLOAD

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


# 5. EXTRACT TEXT FROM PDF

def extract_text_from_pdf(pdf_file):

    # Create PDF reader
    pdf_reader = PdfReader(pdf_file)

    pages_text = []

    # Read every page
    for page_number, page in enumerate(
        pdf_reader.pages
    ):

        text = page.extract_text()

        if text:

            pages_text.append({

                "page": page_number + 1,

                "text": text

            })

    return pages_text


# 6. CHUNK TEXT

def create_chunks(
    pages_text,
    chunk_size=300,
    chunk_overlap=50
):

    chunks = []

    for page in pages_text:

        text = page["text"]

        page_number = page["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({

                    "text": chunk_text,

                    "page": page_number

                })

            # Move forward while keeping overlap

            start = end - chunk_overlap

    return chunks


# 7. CREATE EMBEDDING

def create_embedding(text):

    response = client.embeddings.create(

        model="text-embedding-3-small",

        input=text

    )

    return response.data[0].embedding


# 8. CREATE VECTOR INDEX

def create_faiss_index(chunks):

    embeddings = []

    for chunk in chunks:

        embedding = create_embedding(
            chunk["text"]
        )

        embeddings.append(
            embedding
        )


    # Convert to NumPy array

    embeddings = np.array(
        embeddings
    ).astype("float32")


    # Get embedding dimension

    dimension = embeddings.shape[1]


    # Create FAISS index

    index = faiss.IndexFlatL2(
        dimension
    )


    # Add embeddings

    index.add(
        embeddings
    )


    return index


# 9. RETRIEVE RELEVANT CHUNKS

def retrieve_chunks(
    query,
    index,
    chunks,
    top_k=3
):

    # Create query embedding

    query_embedding = create_embedding(
        query
    )


    # Convert to NumPy

    query_embedding = np.array(
        [query_embedding]
    ).astype("float32")


    # Search FAISS

    distances, indices = index.search(

        query_embedding,

        top_k

    )


    retrieved_chunks = []


    for i in range(top_k):

        chunk_index = indices[0][i]

        retrieved_chunks.append({

            "text":
            chunks[chunk_index]["text"],

            "page":
            chunks[chunk_index]["page"],

            "distance":
            distances[0][i]

        })


    return retrieved_chunks


# 10. GENERATE ANSWER

def generate_answer(
    query,
    retrieved_chunks
):

    # Combine retrieved chunks

    context = ""

    for chunk in retrieved_chunks:

        context += f"""

Source Page: {chunk["page"]}

{chunk["text"]}

"""


    # Create RAG prompt

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context.

Do not use outside knowledge.

If the answer cannot be found in the context,
say:

"I could not find the answer in the uploaded document."

Always provide the page number of the relevant information
when possible.

Context:
{context}

User Question:
{query}

Answer:
"""


    # Call OpenAI

    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[

            {
                "role": "system",

                "content":
                "You answer questions using "
                "retrieved document context."
            },

            {
                "role": "user",

                "content": prompt
            }

        ],

        temperature=0

    )


    return response.choices[0].message.content


# 11. PROCESS PDF

if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


    # Extract text

    with st.spinner(
        "Reading PDF..."
    ):

        pages_text = extract_text_from_pdf(
            uploaded_file
        )


    st.info(
        f"Extracted text from "
        f"{len(pages_text)} pages."
    )


    # Create chunks

    with st.spinner(
        "Creating document chunks..."
    ):

        chunks = create_chunks(
            pages_text
        )


    st.info(
        f"Created {len(chunks)} chunks."
    )


    # Create FAISS index

    with st.spinner(
        "Creating embeddings and vector index..."
    ):

        index = create_faiss_index(
            chunks
        )


    st.success(
        "Document successfully indexed!"
    )


    # 12. USER QUESTION

    query = st.text_input(

        "Ask a question about the document:"

    )


    if query:

        # Retrieve chunks

        with st.spinner(
            "Searching document..."
        ):

            retrieved_chunks = retrieve_chunks(

                query,

                index,

                chunks,

                top_k=3

            )


        # Generate answer

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(

                query,

                retrieved_chunks

            )

        # 13. DISPLAY ANSWER

        st.subheader(
            "Answer"
        )

        st.write(
            answer
        )

        # 14. DISPLAY SOURCES

        st.subheader(
            "Sources"
        )


        for i, chunk in enumerate(
            retrieved_chunks
        ):

            with st.expander(

                f"Source {i + 1} "
                f"- Page {chunk['page']}"

            ):

                st.write(
                    chunk["text"]
                )

                st.write(

                    f"Distance: "
                    f"{chunk['distance']:.4f}"

                )