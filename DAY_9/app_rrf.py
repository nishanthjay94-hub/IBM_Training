import os

import streamlit as st
import numpy as np
import faiss

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. LOAD ENVIRONMENT VARIABLES

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY is not loaded. "
        "Please check your .env file."
    )
    st.stop()

client = OpenAI(
    api_key=api_key
)


# 2. PAGE CONFIGURATION

st.set_page_config(
    page_title="RRF RAG Assistant",
    page_icon="🔎",
    layout="wide"
)


# 3. TITLE

st.title(
    "🔎 Hybrid RAG with RRF Reranking"
)

st.write(
    "This application combines semantic search "
    "and keyword search using Reciprocal Rank Fusion."
)


# 4. PDF UPLOAD

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)

# 5. EXTRACT PDF TEXT

def extract_text_from_pdf(pdf_file):

    pdf_reader = PdfReader(pdf_file)

    pages_text = []

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


# 6. CREATE CHUNKS

def create_chunks(
    pages_text,
    chunk_size=500,
    chunk_overlap=100
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

                    "id": len(chunks),

                    "text": chunk_text,

                    "page": page_number

                })

            start = end - chunk_overlap

    return chunks


# 7. CREATE OPENAI EMBEDDING

def create_embedding(text):

    response = client.embeddings.create(

        model="text-embedding-3-small",

        input=text

    )

    return response.data[0].embedding


# 8. CREATE SEMANTIC FAISS INDEX

def create_faiss_index(chunks):

    embeddings = []

    for chunk in chunks:

        embedding = create_embedding(
            chunk["text"]
        )

        embeddings.append(
            embedding
        )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# 9. CREATE TF-IDF INDEX

def create_tfidf_index(chunks):

    documents = [

        chunk["text"]

        for chunk in chunks

    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    document_vectors = vectorizer.fit_transform(
        documents
    )

    return vectorizer, document_vectors


# 10. SEMANTIC RETRIEVAL

def semantic_retrieval(
    query,
    index,
    chunks,
    top_k=10
):

    query_embedding = create_embedding(
        query
    )

    query_embedding = np.array(
        [query_embedding]
    ).astype("float32")

    distances, indices = index.search(

        query_embedding,

        top_k

    )

    results = []

    for i in range(top_k):

        chunk_index = indices[0][i]

        results.append({

            "chunk_id":
            chunks[chunk_index]["id"],

            "text":
            chunks[chunk_index]["text"],

            "page":
            chunks[chunk_index]["page"],

            "distance":
            float(distances[0][i])

        })

    return results


# 11. KEYWORD RETRIEVAL USING TF-IDF

def keyword_retrieval(
    query,
    vectorizer,
    document_vectors,
    chunks,
    top_k=10
):

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(

        query_vector,

        document_vectors

    )[0]

    top_indices = similarities.argsort()[

        -top_k:

    ][::-1]

    results = []

    for index in top_indices:

        results.append({

            "chunk_id":
            chunks[index]["id"],

            "text":
            chunks[index]["text"],

            "page":
            chunks[index]["page"],

            "score":
            float(similarities[index])

        })

    return results


# 12. RECIPROCAL RANK FUSION

def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    k=60
):

    rrf_scores = {}

    chunk_data = {}


    # ----------------------------------------------
    # Process Semantic Ranking
    # ----------------------------------------------

    for rank, result in enumerate(
        semantic_results
    ):

        chunk_id = result["chunk_id"]

        rrf_scores[chunk_id] = (

            rrf_scores.get(
                chunk_id,
                0
            )

            +

            1 / (k + rank + 1)

        )

        chunk_data[chunk_id] = result


    # ----------------------------------------------
    # Process Keyword Ranking
    # ----------------------------------------------

    for rank, result in enumerate(
        keyword_results
    ):

        chunk_id = result["chunk_id"]

        rrf_scores[chunk_id] = (

            rrf_scores.get(
                chunk_id,
                0
            )

            +

            1 / (k + rank + 1)

        )

        chunk_data[chunk_id] = result


    # ----------------------------------------------
    # Sort by RRF Score
    # ----------------------------------------------

    sorted_chunks = sorted(

        rrf_scores.items(),

        key=lambda x: x[1],

        reverse=True

    )


    # ----------------------------------------------
    # Create Final Results
    # ----------------------------------------------

    final_results = []

    for chunk_id, score in sorted_chunks:

        result = chunk_data[chunk_id].copy()

        result["rrf_score"] = score

        final_results.append(
            result
        )


    return final_results


# ==================================================
# 13. GENERATE ANSWER
# ==================================================

def generate_answer(
    query,
    retrieved_chunks
):

    context = ""

    for chunk in retrieved_chunks:

        context += f"""

Source Page: {chunk["page"]}

{chunk["text"]}

"""


    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context.

Do not use outside knowledge.

If the answer cannot be found in the context, say:

"I could not find the answer in the uploaded document."

Always mention the source page when possible.

Context:
{context}

User Question:
{query}

Answer:
"""


    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[

            {
                "role": "system",

                "content":
                "Answer questions using "
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


# ==================================================
# 14. PROCESS PDF
# ==================================================

if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


    # ----------------------------------------------
    # Extract Text
    # ----------------------------------------------

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


    # ----------------------------------------------
    # Create Chunks
    # ----------------------------------------------

    with st.spinner(
        "Creating chunks..."
    ):

        chunks = create_chunks(
            pages_text
        )


    st.info(
        f"Created {len(chunks)} chunks."
    )


    # ----------------------------------------------
    # Create FAISS Index
    # ----------------------------------------------

    with st.spinner(
        "Creating semantic index..."
    ):

        faiss_index = create_faiss_index(
            chunks
        )


    # ----------------------------------------------
    # Create TF-IDF Index
    # ----------------------------------------------

    with st.spinner(
        "Creating keyword index..."
    ):

        vectorizer, document_vectors = \
            create_tfidf_index(
                chunks
            )


    st.success(
        "Semantic and keyword indexes created!"
    )


    # ==================================================
    # 15. USER QUESTION
    # ==================================================

    query = st.text_input(

        "Ask a question about the document:"

    )


    if query:

        # ----------------------------------------------
        # Semantic Retrieval
        # ----------------------------------------------

        semantic_results = semantic_retrieval(

            query,

            faiss_index,

            chunks,

            top_k=10

        )


        # ----------------------------------------------
        # Keyword Retrieval
        # ----------------------------------------------

        keyword_results = keyword_retrieval(

            query,

            vectorizer,

            document_vectors,

            chunks,

            top_k=10

        )


        # ----------------------------------------------
        # RRF Fusion
        # ----------------------------------------------

        reranked_results = reciprocal_rank_fusion(

            semantic_results,

            keyword_results,

            k=60

        )


        # Take final top 3

        final_results = reranked_results[:3]


        # ----------------------------------------------
        # Generate Answer
        # ----------------------------------------------

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(

                query,

                final_results

            )


        # ----------------------------------------------
        # Display Answer
        # ----------------------------------------------

        st.subheader(
            "Answer"
        )

        st.write(
            answer
        )


        # ----------------------------------------------
        # Display Retrieval Results
        # ----------------------------------------------

        st.subheader(
            "RRF Reranked Sources"
        )


        for i, chunk in enumerate(
            final_results
        ):

            with st.expander(

                f"Result {i + 1} "
                f"- Page {chunk['page']}"

            ):

                st.write(
                    chunk["text"]
                )

                st.write(

                    f"RRF Score: "
                    f"{chunk['rrf_score']:.6f}"

                )