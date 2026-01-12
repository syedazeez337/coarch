//! Minimal Rust indexer - parallel file scanning and chunking

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use regex::Regex;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

const CHUNK_SIZE: usize = 50;
const OVERLAP: usize = 10;
const MIN_CHUNK_SIZE: usize = 10;
const MAX_FILE_SIZE: usize = 10 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub type_: String,
}

#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub code: String,
    pub language: String,
    pub symbols: Vec<Symbol>,
    pub ast_hash: String,
}

#[pyclass]
pub struct RustIndexer {
    func_regex: Regex,
    class_regex: Regex,
    import_regex: Regex,
}

#[pymethods]
impl RustIndexer {
    #[new]
    fn new() -> PyResult<Self> {
        let func_regex = Regex::new(r"(?:fn|def|function|func)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap();
        let class_regex = Regex::new(r"(class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap();
        let import_regex = Regex::new(r"(?:import|from)\s+([a-zA-Z0-9_.]+)").unwrap();
        Ok(Self { func_regex, class_regex, import_regex })
    }

    fn index_directory(&self, repo_path: &str) -> PyResult<Py<PyDict>> {
        let repo_path = PathBuf::from(repo_path);
        let start = std::time::Instant::now();
        let files = self.scan_files_inner(&repo_path);
        let chunks: Vec<CodeChunk> = files.par_iter().filter_map(|f| self.index_file(f, &repo_path)).collect();
        let elapsed = start.elapsed();
        
        Python::with_gil(|py| {
            let result = PyDict::new(py);
            result.set_item("files_indexed", files.len())?;
            result.set_item("chunks_created", chunks.len())?;
            result.set_item("time_ms", elapsed.as_secs_f64() * 1000.0)?;
            result.set_item("parallel_workers", rayon::current_num_threads())?;
            let chunks_list: &PyList = PyList::new(py, std::iter::empty::<&str>());
            for chunk in chunks {
                let safe_code = chunk.code.replace("\n", "\n").replace("|", "\\p");
                let symbols_str = chunk.symbols.iter().map(|s| format!("{}:{}", s.name, s.type_)).collect::<Vec<_>>().join(",");
                let json = format!("{}|{}|{}|{}|{}|{}|{}", 
                    chunk.file_path, chunk.start_line, chunk.end_line, 
                    chunk.language, chunk.ast_hash, safe_code, symbols_str
                );
                chunks_list.append(json.as_str())?;
            }
            result.set_item("chunks_data", chunks_list)?;
            Ok(result.into())
        })
    }

    fn scan_files(&self, repo_path: &str) -> usize {
        let repo_path = PathBuf::from(repo_path);
        self.scan_files_inner(&repo_path).len()
    }

    fn benchmark(&self, repo_path: &str, iterations: usize) -> PyResult<Py<PyDict>> {
        let repo_path = PathBuf::from(repo_path);
        let mut times = Vec::new();
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _files = self.scan_files_inner(&repo_path);
            times.push(start.elapsed().as_secs_f64());
        }
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        Python::with_gil(|py| {
            let stats = PyDict::new(py);
            stats.set_item("iterations", iterations)?;
            stats.set_item("avg_time_ms", avg * 1000.0)?;
            stats.set_item("parallel_workers", rayon::current_num_threads())?;
            Ok(stats.into())
        })
    }
}

impl RustIndexer {
    fn scan_files_inner(&self, repo_path: &Path) -> Vec<PathBuf> {
        let exts = ["py", "rs", "js", "ts", "go", "java", "cpp", "c", "h", "md", "json", "yaml", "yml"];
        let mut files = Vec::new();
        fn recurse(path: &Path, exts: &[&str], files: &mut Vec<PathBuf>) {
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() {
                        if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                            if exts.contains(&ext) && !p.file_name().unwrap().to_string_lossy().starts_with(".") {
                                files.push(p);
                            }
                        }
                    } else if p.is_dir() {
                        let name = p.file_name().unwrap_or_default().to_string_lossy();
                        if !name.starts_with(".") && name != "target" && name != "node_modules" && name != "build" && name != "dist" {
                            recurse(&p, exts, files);
                        }
                    }
                }
            }
        }
        recurse(repo_path, &exts, &mut files);
        files
    }

    fn index_file(&self, file_path: &Path, repo_path: &Path) -> Option<CodeChunk> {
        if file_path.metadata().ok()?.len() > MAX_FILE_SIZE as u64 { return None; }
        let rel_path = file_path.strip_prefix(repo_path).unwrap_or(file_path).to_string_lossy().into_owned();
        let content = fs::read_to_string(file_path).ok()?;
        let language = self.detect_language(file_path);
        let symbols = self.extract_symbols(&content);
        let ast_hash = self.hash_code(&content);
        let lines: Vec<&str> = content.lines().collect();
        let mut chunk = String::new();
        let start = 1;
        for (i, line) in lines.iter().enumerate() {
            chunk.push_str(line);
            chunk.push('\n');
            if (i + 1) % CHUNK_SIZE == 0 {
                let end = i + 1;
                return Some(CodeChunk { file_path: rel_path, start_line: start, end_line: end, code: chunk, language: language.clone(), symbols: symbols.clone(), ast_hash: ast_hash.clone() });
            }
        }
        if chunk.len() > MIN_CHUNK_SIZE {
            Some(CodeChunk { file_path: rel_path, start_line: start, end_line: lines.len(), code: chunk, language, symbols, ast_hash })
        } else { None }
    }

    fn detect_language(&self, path: &Path) -> String {
        match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
            "py" => "python", "rs" => "rust", "js" => "javascript", "ts" => "typescript",
            "go" => "go", "java" => "java", "cpp" | "cc" | "cxx" | "hpp" => "cpp",
            "c" | "h" => "c", "md" => "markdown", "json" => "json", "yaml" | "yml" => "yaml",
            _ => "text",
        }.to_string()
    }

    fn extract_symbols(&self, code: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        for cap in self.func_regex.captures_iter(code) {
            if let Some(m) = cap.get(1) { symbols.push(Symbol { name: m.as_str().to_string(), type_: "function".to_string() }); }
        }
        for cap in self.class_regex.captures_iter(code) {
            if let Some(m) = cap.get(2) { symbols.push(Symbol { name: m.as_str().to_string(), type_: "class".to_string() }); }
        }
        for cap in self.import_regex.captures_iter(code) {
            if let Some(m) = cap.get(1) { symbols.push(Symbol { name: m.as_str().to_string(), type_: "import".to_string() }); }
        }
        symbols
    }

    fn hash_code(&self, code: &str) -> String {
        let mut h = DefaultHasher::new();
        code.hash(&mut h);
        format!("{:x}", h.finish())
    }
}

#[pymodule]
fn coarch_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustIndexer>()?;
    Ok(())
}
