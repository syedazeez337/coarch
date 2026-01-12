"""Tree-sitter C bindings for high-performance AST parsing."""

import os
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TreeSitterSymbol:
    """A symbol extracted from Tree-sitter AST."""

    name: str
    type: str
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass
class TreeSitterParseResult:
    """Result of parsing a file with Tree-sitter."""

    file_path: str
    language: str
    root_node: Any
    symbols: List[TreeSitterSymbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    parse_time_ms: float = 0.0
    error: Optional[str] = None


class TreeSitterCAnalyzer:
    """High-performance AST analyzer using Tree-sitter C bindings."""

    LANGUAGE_MAP = {
        "python": "python",
        "javascript": "javascript",
        "typescript": "typescript",
        "java": "java",
        "cpp": "cpp",
        "c": "c",
        "go": "go",
        "rust": "rust",
        "ruby": "ruby",
        "php": "php",
        "swift": "swift",
        "kotlin": "kotlin",
        "scala": "scala",
        "tsx": "tsx",
        "jsx": "jsx",
    }

    NODE_TYPES = {
        "function_definition",
        "function_declaration",
        "method_definition",
        "class_definition",
        "class_declaration",
        "struct_declaration",
        "import_statement",
        "import_from_statement",
        "call_expression",
        "method_call_expression",
        "variable_declarator",
        "const_declarator",
        "let_declarator",
    }

    def __init__(self, use_c_binding: bool = True, cache_dir: Optional[str] = None):
        """Initialize the Tree-sitter analyzer.

        Args:
            use_c_binding: Use C bindings (faster) or pure Python (fallback)
            cache_dir: Directory for cached parsers
        """
        self.use_c_binding = use_c_binding
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/tree-sitter")
        self._parser_cache: Dict[str, Any] = {}
        self._language_cache: Dict[str, Any] = {}

        if use_c_binding:
            self._init_c_bindings()
        else:
            self._init_pure_python()

    def _init_c_bindings(self):
        """Initialize Tree-sitter C bindings."""
        try:
            import tree_sitter

            self.ts = tree_sitter
            self._use_native = True
            print(
                "[OK] Tree-sitter C bindings loaded (using regex fallback for symbols)"
            )
        except ImportError:
            print("[WARN] Tree-sitter C bindings not available, using pure Python")
            self._init_pure_python()

    def _init_pure_python(self):
        """Initialize pure Python fallback."""
        self._use_native = False
        self._regex_symbols = self._build_regex_extractors()

    def _build_regex_extractors(self) -> Dict[str, dict]:
        """Build regex-based extractors as fallback."""
        import re

        return {
            "python": {
                "class": re.compile(r"class\s+(\w+)"),
                "function": re.compile(r"def\s+(\w+)\s*\("),
                "import": re.compile(r"(?:from\s+)?import\s+(\w+)"),
            },
            "javascript": {
                "class": re.compile(r"class\s+(\w+)"),
                "function": re.compile(r"(?:function|const|let|var)\s+(\w+)\s*="),
                "import": re.compile(r"import\s+.*\s+from\s+['\"](\w+)"),
            },
        }

    def _get_language(self, file_path: str) -> Optional[str]:
        """Get Tree-sitter language for a file."""
        ext = Path(file_path).suffix.lstrip(".").lower()
        return self.LANGUAGE_MAP.get(ext)

    def is_supported(self, file_path: str) -> bool:
        """Check if a file type is supported."""
        return self._get_language(file_path) is not None

    def _load_language(self, language: str):
        """Load a Tree-sitter language parser."""
        if language in self._language_cache:
            return self._language_cache[language]

        try:
            from tree_sitter import Language

            lang_path = os.path.join(self.cache_dir, f"languages-{language}.so")
            os.makedirs(self.cache_dir, exist_ok=True)

            if not os.path.exists(lang_path):
                from tree_sitter_languages import get_language

                lang = get_language(language)
                with open(lang_path, "wb") as f:
                    f.write(lang.sp_bin)

            self._language_cache[language] = Language(lang_path)
            return self._language_cache[language]
        except Exception as e:
            print(f"Failed to load language {language}: {e}")
            return None

    def parse_file(
        self, file_path: str, timeout_ms: int = 30000
    ) -> TreeSitterParseResult:
        """Parse a file using Tree-sitter.

        Args:
            file_path: Path to file
            timeout_ms: Parse timeout in milliseconds

        Returns:
            TreeSitterParseResult with symbols and metadata
        """
        start_time = time.time()

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
        except Exception as e:
            return TreeSitterParseResult(
                file_path=file_path, language="", root_node=None, error=str(e)
            )

        language = self._get_language(file_path)
        if not language:
            return TreeSitterParseResult(
                file_path=file_path,
                language="",
                root_node=None,
                error="Unsupported language",
            )

        parse_time_ms = (time.time() - start_time) * 1000

        if self._use_native:
            return self._parse_native(
                file_path, code, language, timeout_ms, parse_time_ms
            )
        else:
            return self._parse_regex(file_path, code, language, parse_time_ms)

    def _parse_native(
        self,
        file_path: str,
        code: str,
        language: str,
        timeout_ms: int,
        parse_time_ms: float,
    ) -> TreeSitterParseResult:
        """Parse using Tree-sitter C bindings."""
        try:
            lang = self._load_language(language)
            if lang is None:
                return TreeSitterParseResult(
                    file_path=file_path,
                    language=language,
                    root_node=None,
                    error="Language not loaded",
                )

            parser = self.ts.Parser()
            parser.set_language(lang)  # type: ignore[attr-defined]

            tree = parser.parse(bytes(code, "utf-8"))
            root_node = tree.root_node

            symbols = self._extract_symbols_native(root_node, language)
            imports = self._extract_imports_native(root_node, language)
            function_calls = self._extract_calls_native(root_node, language)

            parse_time_ms = (time.time() - (time.time() - parse_time_ms / 1000)) * 1000

            return TreeSitterParseResult(
                file_path=file_path,
                language=language,
                root_node=root_node,
                symbols=symbols,
                imports=imports,
                function_calls=function_calls,
                parse_time_ms=parse_time_ms,
            )
        except Exception as e:
            return TreeSitterParseResult(
                file_path=file_path, language=language, root_node=None, error=str(e)
            )

    def _extract_symbols_native(
        self, node: Any, language: str
    ) -> List[TreeSitterSymbol]:
        """Extract symbols from native Tree-sitter AST."""
        symbols = []

        def traverse(node):
            if node.type in self.NODE_TYPES:
                symbol = TreeSitterSymbol(
                    name=self._get_node_name(node),
                    type=self._map_node_type(node.type, language),
                    start_point=node.start_point,
                    end_point=node.end_point,
                    children=[c.type for c in node.children if c.type != "_empty"],
                )
                symbols.append(symbol)

            for child in node.children:
                traverse(child)

        traverse(node)
        return symbols

    def _get_node_name(self, node: Any) -> str:
        """Get the name/identifier from a node."""
        if hasattr(node, "named_children"):
            for child in node.named_children:
                if child.type in ("identifier", "name", "function_name"):
                    return (
                        child.text.decode()
                        if isinstance(child.text, bytes)
                        else str(child.text)
                    )
        if hasattr(node, "text"):
            text = node.text
            return text.decode() if isinstance(text, bytes) else str(text)
        return ""

    def _map_node_type(self, node_type: str, language: str) -> str:
        """Map Tree-sitter node types to symbol types."""
        type_map = {
            "function_definition": "function",
            "function_declaration": "function",
            "method_definition": "method",
            "class_definition": "class",
            "class_declaration": "class",
            "struct_declaration": "class",
            "import_statement": "import",
            "import_from_statement": "import",
            "call_expression": "call",
            "method_call_expression": "call",
        }
        return type_map.get(node_type, node_type)

    def _extract_imports_native(self, node: Any, language: str) -> List[str]:
        """Extract import statements from native AST."""
        imports = []

        def traverse(node):
            if node.type in ("import_statement", "import_from_statement"):
                for child in node.named_children:
                    if child.type == "module_name" or child.type == "dotted_name":
                        imports.append(child.text.decode())

            for child in node.children:
                traverse(child)

        traverse(node)
        return list(set(imports))

    def _extract_calls_native(self, node: Any, language: str) -> List[str]:
        """Extract function calls from native AST."""
        calls = []

        def traverse(node):
            if node.type in ("call_expression", "method_call_expression"):
                for child in node.named_children:
                    if child.type == "function" or child.type == "method":
                        calls.append(child.text.decode())

            for child in node.children:
                traverse(child)

        traverse(node)
        return list(set(calls))

    def _parse_regex(
        self, file_path: str, code: str, language: str, parse_time_ms: float
    ) -> TreeSitterParseResult:
        """Parse using regex fallback (slower but no dependencies)."""
        extractors = self._regex_symbols.get(language, {})
        symbols = []

        for sym_type, pattern in extractors.items():
            for match in pattern.finditer(code):
                line = code[: match.start()].count("\n") + 1
                symbols.append(
                    TreeSitterSymbol(
                        name=match.group(1),
                        type=sym_type,
                        start_point=(line, 0),
                        end_point=(line, 0),
                    )
                )

        return TreeSitterParseResult(
            file_path=file_path,
            language=language,
            root_node=None,
            symbols=symbols,
            parse_time_ms=parse_time_ms,
        )

    def parse_code(self, code: str, language: str) -> TreeSitterParseResult:
        """Parse a code string directly."""
        start_time = time.time()

        if self._use_native:
            try:
                lang = self._load_language(language)
                if lang is None:
                    return TreeSitterParseResult(
                        file_path="",
                        language=language,
                        root_node=None,
                        error="Language not loaded",
                    )

                parser = self.ts.Parser()
                parser.set_language(lang)  # type: ignore[attr-defined]
                tree = parser.parse(bytes(code, "utf-8"))

                symbols = self._extract_symbols_native(tree.root_node, language)
                imports = self._extract_imports_native(tree.root_node, language)
                function_calls = self._extract_calls_native(tree.root_node, language)

                parse_time_ms = (time.time() - start_time) * 1000

                return TreeSitterParseResult(
                    file_path="",
                    language=language,
                    root_node=tree.root_node,
                    symbols=symbols,
                    imports=imports,
                    function_calls=function_calls,
                    parse_time_ms=parse_time_ms,
                )
            except Exception as e:
                return TreeSitterParseResult(
                    file_path="", language=language, root_node=None, error=str(e)
                )
        else:
            return self._parse_regex("", code, language, 0)

    def batch_parse(
        self, file_paths: List[str], show_progress: bool = True
    ) -> List[TreeSitterParseResult]:
        """Parse multiple files."""
        from tqdm import tqdm

        results = []
        for path in tqdm(file_paths, desc="Parsing") if show_progress else file_paths:
            result = self.parse_file(path)
            results.append(result)

        return results

    def get_symbols_by_type(
        self, symbols: List[TreeSitterSymbol]
    ) -> Dict[str, List[str]]:
        """Group symbols by type."""
        grouped: Dict[str, List[str]] = {}
        for sym in symbols:
            if sym.type not in grouped:
                grouped[sym.type] = []
            grouped[sym.type].append(sym.name)
        return grouped

    def calculate_complexity(self, code: str, language: str) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        keywords = ["if", "elif", "else", "for", "while", "case", "&&", "||", "?"]

        if self._use_native:
            result = self.parse_code(code, language)
            if result.root_node:
                for kw in keywords:
                    complexity += code.count(kw)
        else:
            for kw in keywords:
                complexity += code.count(kw)

        return complexity
