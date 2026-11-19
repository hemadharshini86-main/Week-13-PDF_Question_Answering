import os
import pandas as pd
import chromadb

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PDF_PATH = os.path.join(DATASET_DIR, "sample.pdf")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# 2. Extract PDF
reader = PdfReader(PDF_PATH)
pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
print("Pages:", len(reader.pages))

# 3. Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)
chunks = splitter.split_text(pdf_text)
print("Total chunks:", len(chunks))

# 4. Load models
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatOllama(model="llama3.2:1b", temperature=0)

# 5. Store in ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DIR)

try:
    client.delete_collection("pdf_qa_collection")
except Exception:
    pass

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    metadatas=[{"chunk_id": i + 1} for i in range(len(chunks))],
    persist_directory=CHROMA_DIR,
    collection_name="pdf_qa_collection",
    collection_metadata={"hnsw:space": "cosine"}
)

# 6. Queries
queries = [
    "What is machine learning?",
    "What are supervised and unsupervised learning?",
    "What is reinforcement learning?",
    "What is linear regression?",
    "How can we evaluate a machine learning model?"
]

pd.DataFrame({
    "query_id": range(1, len(queries) + 1),
    "query": queries
}).to_csv(os.path.join(DATASET_DIR, "queries.csv"), index=False)

# 7. Semantic search
semantic_records = []

for query in queries:
    results = vectorstore.similarity_search_with_score(query, k=5)

    for rank, (doc, distance) in enumerate(results, 1):
        score = 1 - distance

        semantic_records.append({
            "query": query,
            "rank": rank,
            "chunk_id": doc.metadata["chunk_id"],
            "similarity_score": round(float(score), 4),
            "content": doc.page_content
        })

        print(f"Semantic | {rank} | Chunk {doc.metadata['chunk_id']} | {score:.4f}")

pd.DataFrame(semantic_records).to_csv(
    os.path.join(DATASET_DIR, "semantic_search_results.csv"),
    index=False
)

# 8. Keyword search
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(chunks)
keyword_records = []

for query in queries:
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    for rank, index in enumerate(scores.argsort()[::-1][:5], 1):
        keyword_records.append({
            "query": query,
            "rank": rank,
            "chunk_id": index + 1,
            "similarity_score": round(float(scores[index]), 4),
            "content": chunks[index]
        })

        print(f"Keyword | {rank} | Chunk {index + 1} | {scores[index]:.4f}")

pd.DataFrame(keyword_records).to_csv(
    os.path.join(DATASET_DIR, "keyword_search_results.csv"),
    index=False
)

# 9. PDF Question Answering
answer_records = []

for query in queries:
    docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
    Answer using only the PDF context.
    If the answer is unavailable, say so.

    Context:
    {context}

    Question:
    {query}
    """

    try:
        answer = llm.invoke(prompt).content.strip()
    except Exception as error:
        answer = f"Answer generation failed: {error}"

    print("\nQuestion:", query)
    print("Answer:", answer)

    answer_records.append({
        "query": query,
        "answer": answer,
        "retrieved_context": context
    })

# 10. Save answers
pd.DataFrame(answer_records).to_csv(
    os.path.join(DATASET_DIR, "answers.csv"),
    index=False
)

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("Chunks:", len(chunks))
print("Semantic results:", len(semantic_records))
print("Keyword results:", len(keyword_records))
print("Answers:", len(answer_records))