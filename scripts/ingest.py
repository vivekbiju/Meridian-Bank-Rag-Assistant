import os
import re
import psycopg
import pdfplumber
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Load lightweight 384-dimensional embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
DB_URI = os.getenv("DATABASE_URL")

def extract_pdf_text(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            # Strip running headers/footers without dropping body content
            text = re.sub(r"Meridian Bank plc.*?\n", "", text)
            text = re.sub(r"Valid from 1 March 2026.*?\n", "", text)
            text = re.sub(r"Page \d+", "", text)
            full_text += text + "\n"
    return full_text

def chunk_full_text(text: str, chunk_size=512, overlap=64):
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk_content = text[start:end].strip()
        
        if chunk_content:
            # Match section titles like "1. Your cards" or default to general context
            section_match = re.search(r"(\d+\.\s+[A-Za-z0-9\s]+)", chunk_content)
            section_title = section_match.group(1).strip() if section_match else "General Handbook Context"
            chunks.append((section_title, chunk_content))
            
        start += (chunk_size - overlap)
        
    return chunks

def ingest():
    if not DB_URI:
        raise ValueError("DATABASE_URL environment variable is not set.")

    print("Extracting and cleaning PDF...")
    pdf_text = extract_pdf_text("corpus/meridian-handbook.pdf")
    
    raw_chunks = chunk_full_text(pdf_text, chunk_size=512, overlap=64)
    print(f"Generated {len(raw_chunks)} chunks. Batch generating embeddings...")

    # Vectorized Batch Encoding (4x faster than looping single chunks)
    contents = [chunk[1] for chunk in raw_chunks]
    embeddings = embedder.encode(contents, batch_size=32, show_progress_bar=True)

    db_records = [
        (raw_chunks[i][0], i, contents[i], str(embeddings[i].tolist()))
        for i in range(len(raw_chunks))
    ]

    print("Ingesting chunks into pgvector...")

    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS chunks;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE chunks (
                    id BIGSERIAL PRIMARY KEY,
                    section TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR(384) NOT NULL,
                    UNIQUE (section, chunk_index)
                );
            """)
            cur.execute("""
                CREATE INDEX chunks_embedding_idx 
                ON chunks USING hnsw (embedding vector_cosine_ops);
            """)

            upsert_query = """
                INSERT INTO chunks (section, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (section, chunk_index) 
                DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding;
            """
            cur.executemany(upsert_query, db_records)
            conn.commit()

    print("Re-ingestion complete successfully.")

if __name__ == "__main__":
    ingest()