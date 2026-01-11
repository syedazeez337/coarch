"""Tree-sitter based structural analysis for code chunks."""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class CodeSymbol:
    """A symbol extracted from code."""

    name: str
    type: str  # function, class, method, variable, import
    start_line: int
    end_line: int
    parent: Optional[str] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None


@dataclass
class StructuralInfo:
    """Structural analysis result for a code chunk."""

    file_path: str
    symbols: List[CodeSymbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    ast_hash: str = ""
    complexity: int = 0


class TreeSitterAnalyzer:
    """Structural code analysis using Tree-sitter."""

    LANGUAGE_MAP = {
        "python": "python",
        "javascript": "javascript",
        "typescript": "typescript",
        "jsx": "javascript",
        "tsx": "typescript",
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
    }

    def __init__(self):
        """Initialize the analyzer."""
        self.parser_cache = {}

    def get_language(self, file_path: str) -> Optional[str]:
        """Get Tree-sitter language for a file."""
        ext = Path(file_path).suffix.lstrip(".").lower()
        return self.LANGUAGE_MAP.get(ext)

    def is_supported(self, file_path: str) -> bool:
        """Check if a file type is supported."""
        return self.get_language(file_path) is not None

    def extract_symbols(self, code: str, language: str) -> List[CodeSymbol]:
        """Extract symbols from code using Tree-sitter queries."""
        symbols = []

        if language == "python":
            symbols = self._extract_python_symbols(code)
        elif language in ("javascript", "typescript", "jsx", "tsx"):
            symbols = self._extract_js_symbols(code, language)
        elif language in ("java",):
            symbols = self._extract_java_symbols(code)
        elif language in ("cpp", "c"):
            symbols = self._extract_cpp_symbols(code)
        elif language in ("go",):
            symbols = self._extract_go_symbols(code)
        elif language in ("rust",):
            symbols = self._extract_rust_symbols(code)
        else:
            symbols = self._extract_generic_symbols(code, language)

        return symbols

    def _extract_python_symbols(self, code: str) -> List[CodeSymbol]:
        """Extract Python symbols."""
        import re

        symbols = []

        # Class definitions
        class_pattern = r"class\s+(\w+)(?:\s*\(\s*([\w,\s]+)\s*\))?\s*:"
        for match in re.finditer(class_pattern, code):
            symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    type="class",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                    parent=match.group(2) if match.group(2) else None,
                )
            )

        # Function definitions
        func_pattern = r"def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*[\w\[\], ]+)?\s*:"
        for match in re.finditer(func_pattern, code):
            params = match.group(2).strip()
            symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    type="function",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                    signature=f"def {match.group(1)}({params})",
                )
            )

        # Async functions
        async_func_pattern = (
            r"async\s+def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*[\w\[\], ]+)?\s*:"
        )
        for match in re.finditer(async_func_pattern, code):
            params = match.group(2).strip()
            symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    type="function",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                    signature=f"async def {match.group(1)}({params})",
                )
            )

        # Import statements
        import_pattern = r"^(?:from\s+([\w.]+)\s+)?import\s+(.+)"
        for match in re.finditer(import_pattern, code, re.MULTILINE):
            imports = match.group(2).split(",")
            for imp in imports:
                imp = imp.strip().split(" as ")[0]
                if imp:
                    symbols.append(
                        CodeSymbol(
                            name=imp,
                            type="import",
                            start_line=code[: match.start()].count("\n") + 1,
                            end_line=0,
                        )
                    )

        return symbols

    def _extract_js_symbols(self, code: str, language: str) -> List[CodeSymbol]:
        """Extract JavaScript/TypeScript symbols."""
        import re

        symbols = []

        # Class definitions
        class_pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{"
        for match in re.finditer(class_pattern, code):
            symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    type="class",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                    parent=match.group(2),
                )
            )

        # Function declarations
        func_pattern = r"(?:function\s+)?(\w+)\s*\(([^)]*)\)\s*\{"
        for match in re.finditer(func_pattern, code):
            name = match.group(1)
            if name in ("if", "else", "for", "while", "switch", "return"):
                continue
            params = match.group(2).strip()
            symbols.append(
                CodeSymbol(
                    name=name,
                    type="function",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                    signature=f"{name}({params})",
                )
            )

        # Arrow functions
        arrow_pattern = r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>"
        for match in re.finditer(arrow_pattern, code):
            symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    type="function",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                )
            )

        # Import statements
        import_pattern = r"import\s+(?:\{([^}]+)\}|\*\s+as\s+(\w+)|(\w+))\s+from\s+['\"]([^'\"]+)['\"]"
        for match in re.finditer(import_pattern, code):
            imports = match.group(1) or match.group(2) or match.group(3)
            for imp in imports.split(","):
                imp = imp.strip().split(" as ")[-1].strip()
                if imp:
                    symbols.append(
                        CodeSymbol(
                            name=imp,
                            type="import",
                            start_line=code[: match.start()].count("\n") + 1,
                            end_line=0,
                        )
                    )

        return symbols

    def _extract_java_symbols(self, code: str) -> List[CodeSymbol]:
        """Extract Java symbols."""
        import re

        symbols = []

        # Class definitions
        class_pattern = (
            r"(?:public|private|protected|static)?\s*(?:class|interface)\s+(\w+)"
        )
        for match in re.finditer(class_pattern, code):
            symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    type="class",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                )
            )

        # Method definitions
        method_pattern = r"(?:public|private|protected|static|final|synchronized)?\s*[\w<>[\]]+\s+(\w+)\s*\(([^)]*)\)"
        for match in re.finditer(method_pattern, code):
            name = match.group(1)
            params = match.group(2).strip()
            if not name[0].isupper():
                symbols.append(
                    CodeSymbol(
                        name=name,
                        type="method",
                        start_line=code[: match.start()].count("\n") + 1,
                        end_line=0,
                        signature=f"{name}({params})",
                    )
                )

        # Import statements
        import_pattern = r"import\s+([\w.]+)(?:\.\*)?;"
        for match in re.finditer(import_pattern, code):
            symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    type="import",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                )
            )

        return symbols

    def _extract_cpp_symbols(self, code: str) -> List[CodeSymbol]:
        """Extract C++ symbols."""
        import re

        symbols = []

        # Class definitions
        class_pattern = r"(?:class|struct)\s+(\w+)"
        for match in re.finditer(class_pattern, code):
            symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    type="class",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                )
            )

        # Function definitions
        func_pattern = (
            r"(?:void|int|bool|std::\w+|auto)\s+(\w+)\s*\(([^)]*)\)\s*(?:const)?\s*\{"
        )
        for match in re.finditer(func_pattern, code):
            name = match.group(1)
            if name in ("if", "while", "for", "switch", "return"):
                continue
            params = match.group(2).strip()
            symbols.append(
                CodeSymbol(
                    name=name,
                    type="function",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                    signature=f"{name}({params})",
                )
            )

        # Include statements
        include_pattern = r"#include\s+[<\"]([^\">]+)[>\"]"
        for match in re.finditer(include_pattern, code):
            symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    type="import",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                )
            )

        return symbols

    def _extract_go_symbols(self, code: str) -> List[CodeSymbol]:
        """Extract Go symbols."""
        import re

        symbols = []

        # Function definitions
        func_pattern = r"func\s+(?:\([^)]+\)\s*)?(\w+)\s*\(([^)]*)\)"
        for match in re.finditer(func_pattern, code):
            name = match.group(1)
            params = match.group(2).strip()
            symbols.append(
                CodeSymbol(
                    name=name,
                    type="function",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                    signature=f"func {name}({params})",
                )
            )

        # Method definitions
        method_pattern = r"func\s+\(([^)]+)\)\s*(\w+)\s*\(([^)]*)\)"
        for match in re.finditer(method_pattern, code):
            receiver = match.group(1)
            name = match.group(2)
            params = match.group(3).strip()
            symbols.append(
                CodeSymbol(
                    name=name,
                    type="method",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                    parent=receiver,
                    signature=f"func ({receiver}) {name}({params})",
                )
            )

        # Type definitions
        type_pattern = r"type\s+(\w+)\s+(?:struct|interface)"
        for match in re.finditer(type_pattern, code):
            symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    type="class",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                )
            )

        # Import statements
        import_pattern = r"\"([^\"]+)\""
        in_import = False
        for i, line in enumerate(code.split("\n"), 1):
            if "import" in line and "{" in line:
                in_import = True
            if in_import:
                for match in re.finditer(import_pattern, line):
                    symbols.append(
                        CodeSymbol(
                            name=match.group(1), type="import", start_line=i, end_line=0
                        )
                    )
            if in_import and "}" in line:
                in_import = False

        return symbols

    def _extract_rust_symbols(self, code: str) -> List[CodeSymbol]:
        """Extract Rust symbols."""
        import re

        symbols = []

        # Function definitions
        func_pattern = r"fn\s+(\w+)\s*<[^>]*>?\s*\(([^)]*)\)"
        for match in re.finditer(func_pattern, code):
            name = match.group(1)
            params = match.group(2).strip()
            symbols.append(
                CodeSymbol(
                    name=name,
                    type="function",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                    signature=f"fn {name}({params})",
                )
            )

        # Struct definitions
        struct_pattern = r"struct\s+(\w+)"
        for match in re.finditer(struct_pattern, code):
            symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    type="class",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                )
            )

        # Impl blocks
        impl_pattern = r"impl\s+(?:<[^>]+>\s+)?(\w+)"
        for match in re.finditer(impl_pattern, code):
            symbols.append(
                CodeSymbol(
                    name=f"impl_{match.group(1)}",
                    type="class",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                )
            )

        # Use statements
        use_pattern = r"use\s+([\w:]+)(?:\s+as\s+(\w+))?"
        for match in re.finditer(use_pattern, code):
            name = match.group(2) or match.group(1).split("::")[-1]
            symbols.append(
                CodeSymbol(
                    name=name,
                    type="import",
                    start_line=code[: match.start()].count("\n") + 1,
                    end_line=0,
                )
            )

        return symbols

    def _extract_generic_symbols(self, code: str, language: str) -> List[CodeSymbol]:
        """Extract symbols using generic patterns."""
        import re

        symbols = []

        # Common patterns
        patterns = [
            (r"(?:function|def|func|fn)\s+(\w+)", "function"),
            (r"(?:class|struct|interface|type)\s+(\w+)", "class"),
        ]

        for pattern, sym_type in patterns:
            for match in re.finditer(pattern, code):
                symbols.append(
                    CodeSymbol(
                        name=match.group(1),
                        type=sym_type,
                        start_line=code[: match.start()].count("\n") + 1,
                        end_line=0,
                    )
                )

        return symbols

    def analyze_file(self, file_path: str) -> StructuralInfo:
        """Analyze a file and extract structural information."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        language = self.get_language(file_path)
        if not language:
            return StructuralInfo(file_path=file_path)

        symbols = self.extract_symbols(code, language)
        imports = [s.name for s in symbols if s.type == "import"]
        function_calls = self._extract_function_calls(code, language)

        import hashlib

        ast_hash = hashlib.md5(code.encode()).hexdigest()

        return StructuralInfo(
            file_path=file_path,
            symbols=symbols,
            imports=imports,
            function_calls=function_calls,
            ast_hash=ast_hash,
            complexity=self._calculate_complexity(code),
        )

    def _extract_function_calls(self, code: str, language: str) -> List[str]:
        """Extract function calls from code."""
        import re

        calls = set()

        if language == "python":
            call_pattern = r"(\w+)\s*\("
            for match in re.finditer(call_pattern, code):
                name = match.group(1)
                keywords = {
                    "if",
                    "while",
                    "for",
                    "with",
                    "assert",
                    "return",
                    "yield",
                    "print",
                    "len",
                    "str",
                    "int",
                    "float",
                    "list",
                    "dict",
                    "set",
                    "open",
                    "range",
                    "enumerate",
                    "zip",
                }
                if name not in keywords:
                    calls.add(name)
        else:
            call_pattern = r"(\w+)\s*\("
            for match in re.finditer(call_pattern, code):
                calls.add(match.group(1))

        return list(calls)

    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        keywords = ["if", "elif", "else", "for", "while", "case", "&&", "||", "?"]
        for kw in keywords:
            complexity += code.count(kw)
        return complexity

    def extract_chunk_info(
        self, code: str, file_path: str, start_line: int, end_line: int
    ) -> StructuralInfo:
        """Extract structural info for a specific chunk."""
        language = self.get_language(file_path)
        symbols = self.extract_symbols(code, language)

        imports = [s.name for s in symbols if s.type == "import"]
        function_calls = self._extract_function_calls(code, language)

        import hashlib

        ast_hash = hashlib.md5(code.encode()).hexdigest()

        return StructuralInfo(
            file_path=file_path,
            symbols=symbols,
            imports=imports,
            function_calls=function_calls,
            ast_hash=ast_hash,
            complexity=self._calculate_complexity(code),
        )
