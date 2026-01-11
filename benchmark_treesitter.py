"""Benchmark Tree-sitter C bindings vs regex parsing."""

import os
import sys
import time
import re
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Tree-sitter C Bindings Benchmark")
print("=" * 70)

SAMPLE_CODE = '''
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

class DataProcessor:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.cache = {}

    def process(self, data: List[str]) -> Dict[str, int]:
        results = {}
        for item in data:
            key = self._normalize(item)
            if key not in results:
                results[key] = 0
            results[key] += 1
        return results

    def _normalize(self, text: str) -> str:
        return text.lower().strip()

    def load_file(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

class APIHandler:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None

    async def fetch(self, endpoint: str) -> Dict:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/{endpoint}")
            return response.json()

    def parse_response(self, data: Dict) -> List[str]:
        return data.get("items", [])

def main():
    processor = DataProcessor({"debug": "true"})
    handler = APIHandler("https://api.example.com")
    print("Initialized")

if __name__ == "__main__":
    main()
'''

print("\n[1/3] Testing regex-based parser (baseline)")
print("-" * 50)

start = time.time()
for _ in range(1000):
    classes = re.findall(r"class\s+(\w+)", SAMPLE_CODE)
    funcs = re.findall(r"def\s+(\w+)\s*\(", SAMPLE_CODE)
    imports = re.findall(r"import\s+(\w+)", SAMPLE_CODE)
regex_time = time.time() - start

print(f"   Classes: {classes}")
print(f"   Functions: {funcs}")
print(f"   Imports: {imports}")
print(f"   Time for 1000 iterations: {regex_time*1000:.1f}ms ({regex_time:.3f}ms per iteration)")

print("\n[2/3] Testing Tree-sitter C bindings")
print("-" * 50)

tree_sitter_installed = False
ts_time = regex_time

try:
    import tree_sitter
    from tree_sitter_languages import get_language

    tree_sitter_installed = True
    print("   Tree-sitter C bindings detected")

    python_lang = get_language("python")
    parser = tree_sitter.Parser()
    parser.set_language(python_lang)

    ts_parse_time = 0
    for _ in range(1000):
        t_start = time.time()
        tree = parser.parse(bytes(SAMPLE_CODE, "utf-8"))
        ts_parse_time += time.time() - t_start

    def extract_symbols(node, symbols=None):
        if symbols is None:
            symbols = []
        if node.type in ("function_definition", "class_definition"):
            for child in node.named_children:
                if child.type == "name":
                    symbols.append(("function" if node.type == "function_definition" else "class", child.text.decode()))
                    break
        for child in node.children:
            extract_symbols(child, symbols)
        return symbols

    symbols = extract_symbols(tree.root_node)
    print(f"   Symbols: {symbols}")
    print(f"   Parse time for 1000 iterations: {ts_parse_time*1000:.1f}ms ({ts_parse_time:.3f}ms per iteration)")
    ts_time = ts_parse_time

except ImportError as e:
    print(f"   Tree-sitter not installed")
    print("   Install with: pip install tree-sitter tree-sitter-languages")

print("\n[3/3] Performance comparison")
print("-" * 50)

print(f"   Regex parser:     {regex_time*1000:7.1f}ms (baseline)")
print(f"   Tree-sitter C:    {ts_time*1000:7.1f}ms")
if tree_sitter_installed and ts_time < regex_time:
    speedup = regex_time / ts_time
    print(f"   Speedup:          {speedup:.1f}x faster")
    print(f"   Time saved:       {(regex_time - ts_time)*1000:.1f}ms per 1000 parses")
elif tree_sitter_installed:
    print("   Tree-sitter is slower for this simple case")
else:
    print("   Install tree-sitter to compare")

print("\n" + "=" * 70)
print("Benchmark Complete")
print("=" * 70)
print("""
Installation:
   pip install tree-sitter tree-sitter-languages

Note: Tree-sitter provides accurate AST parsing with:
   - Proper nesting and scope analysis
   - Language-specific syntax trees
   - Better symbol extraction accuracy
   - Support for 30+ languages

The C bindings are significantly faster than regex for:
   - Complex codebases
   - Nested structures
   - Language-specific patterns
""")
