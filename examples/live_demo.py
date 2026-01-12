"""Coarch demonstration script - shows semantic code search in action."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import faiss
import json
import http.client
from pathlib import Path

print("=" * 70)
print("Coarch - Semantic Code Search Engine Demo")
print("=" * 70)

DIM = 768

sample_chunks = [
    {
        "id": 1,
        "file": "src/auth.py",
        "lang": "python",
        "code": '''def authenticate_user(username, password):
    """Verify user credentials against database."""
    user = db.query("SELECT * FROM users WHERE username = ?", username)
    if user and verify_password(user.password_hash, password):
        return generate_token(user)
    return None''',
        "symbols": ["authenticate_user"],
    },
    {
        "id": 2,
        "file": "src/auth.py",
        "lang": "python",
        "code": """class AuthService:
    def __init__(self, db_connection):
        self.db = db_connection

    def login(self, email, password):
        user = self.db.find_user(email)
        if user and user.check_password(password):
            return jwt.encode({"sub": user.id})
        raise AuthError("Invalid credentials")""",
        "symbols": ["AuthService", "login"],
    },
    {
        "id": 3,
        "file": "src/parser.py",
        "lang": "python",
        "code": '''def parse_json(data):
    """Parse JSON string into Python dict."""
    import json
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        logging.error(f"Parse error: {e}")
        return None''',
        "symbols": ["parse_json"],
    },
    {
        "id": 4,
        "file": "src/parser.py",
        "lang": "python",
        "code": """class JSONParser:
    def __init__(self, strict_mode=False):
        self.strict = strict_mode

    def parse(self, data):
        import json
        return json.loads(data) if data else {}""",
        "symbols": ["JSONParser", "parse"],
    },
    {
        "id": 5,
        "file": "lib/utils.js",
        "lang": "javascript",
        "code": """function formatDate(date) {
    return new Date(date).toISOString();
}

function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

module.exports = { formatDate, validateEmail };""",
        "symbols": ["formatDate", "validateEmail"],
    },
    {
        "id": 6,
        "file": "src/database.py",
        "lang": "python",
        "code": """class Database:
    def __init__(self, connection_string):
        self.conn = connection_string

    def connect(self):
        return psycopg2.connect(self.conn)

    def query(self, sql, *args):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, args)
            return cur.fetchall()""",
        "symbols": ["Database", "connect", "query"],
    },
]


def simulate_embedding(text):
    """Simulate embedding using a simple hash-based approach."""
    import hashlib

    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    np.random.seed(hash_val % (2**31))
    return np.random.random(DIM).astype(np.float32)


print("\n[1/4] Creating semantic index...")
print(f"   Sample chunks: {len(sample_chunks)}")
print(f"   Embedding dimension: {DIM}")

embeddings = np.array([simulate_embedding(c["code"]) for c in sample_chunks])
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(DIM)
index.add(embeddings)

print(f"   Index created with {index.ntotal} vectors")

print("\n[2/4] Testing semantic search queries...")

queries = [
    ("authentication with JWT token", "semantic search: auth + JWT"),
    ("parsing JSON data", "semantic search: parse + JSON"),
    ("validate email format", "semantic search: validate + email"),
    ("database connection", "semantic search: database + connect"),
]

for query_text, description in queries:
    query_vec = simulate_embedding(query_text)
    faiss.normalize_L2(query_vec.reshape(1, -1))
    scores, ids = index.search(query_vec.reshape(1, -1), 3)

    print(f"\n   Query: '{query_text}'")
    print(f"   ({description})")
    for score, idx in zip(scores[0], ids[0]):
        chunk = sample_chunks[idx]
        print(f"   [{score:.3f}] {chunk['file']} - {chunk['code'][:50]}...")

print("\n[3/4] Testing language filtering...")

python_chunks = [i for i, c in enumerate(sample_chunks) if c["lang"] == "python"]
if python_chunks:
    python_embeddings = embeddings[python_chunks]
    python_index = faiss.IndexFlatIP(DIM)
    python_index.add(python_embeddings)

    query_vec = simulate_embedding("authentication function")
    faiss.normalize_L2(query_vec.reshape(1, -1))
    scores, ids = python_index.search(query_vec.reshape(1, -1), 3)

    print(f"   Python-only results: {len(ids[0])}")
    for score, local_idx in zip(scores[0], ids[0]):
        global_idx = python_chunks[local_idx]
        chunk = sample_chunks[global_idx]
        print(f"   [{score:.3f}] {chunk['file']}")

print("\n[4/4] Example API usage...")

api_example = {
    "endpoint": "POST /search",
    "request": {"query": "authentication function", "language": "python", "limit": 10},
    "response": {
        "results": [
            {
                "file": "src/auth.py",
                "score": 0.892,
                "code": "def authenticate_user(username, password):...",
                "symbols": ["authenticate_user"],
            }
        ],
        "total": 1,
        "query_time_ms": 12.5,
    },
}

print(f"\n   {json.dumps(api_example, indent=2)}")

print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
print("\nTo run Coarch server:")
print("   1. pip install -r requirements.txt")
print("   2. coarch init")
print("   3. coarch index /path/to/your/repo")
print("   4. coarch serve")
print('   5. curl -X POST http://localhost:8000/search -d \'{"query": "..."}\'')
print("=" * 70)
