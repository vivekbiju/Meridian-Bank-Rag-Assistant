import os
import re
import psycopg
import pdfplumber
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
DB_URI = os.getenv("DATABASE_URL")

def extract_and_parse_sections(pdf_path: str):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            # Strip running headers and footers cleanly
            text = re.sub(r"Meridian Bank plc.*?\n", "", text)
            text = re.sub(r"Valid from 1 March 2026.*?\n", "", text)
            text = re.sub(r"Page \d+", "", text)
            full_text += text + "\n"

    # Regex to split strictly on section titles: e.g., "1. Your cards"
    # Matches a digit followed by a dot, space, and capital title
    pattern = r"(\n\d+\.\s+[A-Z][^\n]+)"
    parts = re.split(pattern, full_text)
    
    sections = []
    # Index 0 contains front matter/contents
    for i in range(1, len(parts), 2):
        sec_title = parts[i].strip()
        sec_body = parts[i+1].strip() if (i+1) < len(parts) else ""
        sections.append((sec_title, sec_body))
        
    return sections

def create_chunks(text: str, chunk_size=512, overlap=64):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def ingest():
    if not DB_URI:
        raise ValueError("DATABASE_URL environment variable is not set.")

    print("Extracting and cleaning PDF...")
    parsed_sections = extract_and_parse_sections("corpus/meridian-handbook.pdf")

    db_records = []
    for section_title, section_body in parsed_sections:
        chunks = create_chunks(section_body, chunk_size=512, overlap=64)
        for idx, chunk_content in enumerate(chunks):
            embedding = embedder.encode(chunk_content).tolist()
            db_records.append((section_title, idx, chunk_content, str(embedding)))

    print(f"Generated {len(db_records)} chunks across {len(parsed_sections)} sections. Ingesting...")

    with psycopg.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            # Drop table to clear corrupted section entries cleanly
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