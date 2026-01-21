import pinecone
from sentence_transformers import SentenceTransformer
from config import PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def store_knowledge(text, doc_id):
    vector = embedder.encode(text).tolist()
    index.upsert([(doc_id, vector, {"text": text})])

def retrieve_context(query):
    vector = embedder.encode(query).tolist()
    result = index.query(vector=vector, top_k=3, include_metadata=True)
    return " ".join([m["metadata"]["text"] for m in result["matches"]])
