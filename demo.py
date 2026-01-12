"""Full Coarch demo with simulated embeddings."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Coarch - Full Search Demo")
print("=" * 60)

print("\n[1/4] Creating index and sample data...")

import numpy as np
import faiss
from backend.hybrid_indexer import HybridIndexer, CodeChunk
from backend.ast_analyzer import TreeSitterAnalyzer

indexer = HybridIndexer(db_path=":memory:")
analyzer = TreeSitterAnalyzer()

sample_chunks = [
    CodeChunk(
        file_path="src/auth.py",
        start_line=1,
        end_line=20,
        code='''def authenticate_user(username, password):
    """Verify user credentials against database."""
    user = db.query("SELECT * FROM users WHERE username = ?", username)
    if user and verify_password(user.password_hash, password):
        return generate_token(user)
    return None''',
        language="python",
        symbols=["authenticate_user"],
        ast_hash="hash1",
    ),
    CodeChunk(
        file_path="src/auth.py",
        start_line=50,
        end_line=70,
        code="""class AuthService:
    def __init__(self, db_connection):
        self.db = db_connection

    def login(self, email, password):
        user = self.db.find_user(email)
        if user and user.check_password(password):
            return jwt.encode({"sub": user.id})
        raise AuthError("Invalid credentials")""",
        language="python",
        symbols=["AuthService", "login"],
        ast_hash="hash2",
    ),
    CodeChunk(
        file_path="src/parser.py",
        start_line=1,
        end_line=30,
        code='''def parse_json(data):
    """Parse JSON string into Python dict."""
    import json
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        logging.error(f"Parse error: {e}")
        return None''',
        language="python",
        symbols=["parse_json"],
        ast_hash="hash3",
    ),
    CodeChunk(
        file_path="src/parser.py",
        start_line=100,
        end_line=120,
        code="""class JSONParser:
    def __init__(self, strict_mode=False):
        self.strict = strict_mode

    def parse(self, data):
        import json
        return json.loads(data) if data else {}""",
        language="python",
        symbols=["JSONParser", "parse"],
        ast_hash="hash4",
    ),
    CodeChunk(
        file_path="lib/utils.js",
        start_line=1,
        end_line=15,
        code="""function formatDate(date) {
    return new Date(date).toISOString();
}

function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

module.exports = { formatDate, validateEmail };""",
        language="javascript",
        symbols=["formatDate", "validateEmail"],
        ast_hash="hash5",
    ),
]

for chunk in sample_chunks:
    print(f"   Chunk: {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
    print(f"   Language: {chunk.language}, Symbols: {chunk.symbols}")

print("\n[2/4] Generating simulated embeddings...")

dim = 768
n_chunks = len(sample_chunks)

fake_embeddings = np.random.random((n_chunks, dim)).astype(np.float32)
faiss.normalize_L2(fake_embeddings)

print(f"   Generated {n_chunks} embeddings of dimension {dim}")

index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
index.add(fake_embeddings)

metadata = [
    {
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "code": chunk.code,
        "language": chunk.language,
    }
    for chunk in sample_chunks
]

print("   Index ready with", index.ntotal, "vectors")

print("\n[3/4] Testing searches...")

search_queries = [
    "authentication with JWT token",
    "parsing JSON data",
    "validate email format",
]

for query in search_queries:
    query_embedding = np.random.random((1, dim)).astype(np.float32)
    faiss.normalize_L2(query_embedding)

    scores, ids = index.search(query_embedding, 3)

    print(f"\n   Query: '{query}'")
    print(f"   Results:")

    for score, id_ in zip(scores[0], ids[0]):
        if id_ < 0:
            continue
        meta = metadata[id_]
        print(
            f"      [{score:.3f}] {meta['file_path']}:{meta['start_line']}-{meta['end_line']}"
        )
        print(f"           Lang: {meta['language']}")
        preview = meta["code"][:60].replace("\n", " ")
        print(f"           {preview}...")

print("\n[4/4] Testing language filtering...")

query_embedding = np.random.random((1, dim)).astype(np.float32)
faiss.normalize_L2(query_embedding)
scores, ids = index.search(query_embedding, 5)

python_results = []
for score, id_ in zip(scores[0], ids[0]):
    if id_ < 0:
        continue
    meta = metadata[id_]
    if meta["language"] == "python":
        python_results.append((score, meta))

print(f"   Python-only results: {len(python_results)}")
for score, meta in python_results:
    print(f"      [{score:.3f}] {meta['file_path']}")

print("\n" + "=" * 60)
print("Demo completed successfully!")
print("=" * 60)
print(
    """
To run with real embeddings:
1. pip install -r requirements.txt
2. coarch serve
3. coarch index /path/to/your/repo
4. coarch search "your query"
"""
)
