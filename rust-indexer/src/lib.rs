//! High-performance code indexing for Coarch
//! Target: 100x faster than Python

use pyo3::prelude::*;
use pyo3::types::*;
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use regex::Regex;
use dashmap::DashMap;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

#[pyclass]
pub struct RustIndexer {
    index_dir: PathBuf,
    db_path: PathBuf,
    files: Arc<DashMap<String, f64>>,
    chunks: Arc<DashMap<String, ChunkData>>,
    embedding_cache: Arc<Mutex<LruCache<String, Vec<f32>>>>,
    func_regex: Regex,
    class_regex: Regex,
    import_regex: Regex,
}

#[derive(Clone, Debug)]
pub struct ChunkData {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub code: String,
    pub language: String,
    pub symbols: Vec<Symbol>,
    pub ast_hash: String,
}

#[derive(Clone, Debug)]
pub struct Symbol {
    pub name: String,
    pub type_: String,
    pub start_line: usize,
    pub end_line: usize,
}

#[pymethods]
impl RustIndexer {
    #[new]
    fn new(index_path: &str, db_path: &str) -> PyResult<Self> {
        let index_dir = PathBuf::from(index_path);
        let db_path = PathBuf::from(db_path);
        
        if !index_dir.exists() {
            fs::create_dir_all(&index_dir)?;
        }
        
        let func_regex = Regex::new(r"(?:fn|def|function|func|let|const|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap();
        let class_regex = Regex::new(r"(?:class|struct|type)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap();
        let import_regex = Regex::new(r"(?:import|from|require|using)\s+([a-zA-Z0-9_.]+)").unwrap();
        
        Ok(Self {
            index_dir,
            db_path,
            files: Arc::new(DashMap::new()),
            chunks: Arc::new(DashMap::new()),
            embedding_cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(10000).unwrap()))),
            func_regex,
            class_regex,
            import_regex,
        })
    }

    fn index_directory(&mut self, repo_path: &str, _repo_name: &str) -> PyResult<Py<PyDict>> {
        let repo_path = PathBuf::from(repo_path);
        let start = std::time::Instant::now();
        
        // Phase 1: Parallel file scanning
        let files: Vec<PathBuf> = self.scan_files_parallel(&repo_path);
        
        // Phase 2: Parallel file indexing
        let chunks: Vec<ChunkData> = files
            .par_iter()
            .filter_map(|f| self.index_file(f, &repo_path))
            .collect();
        
        // Store chunks
        for chunk in &chunks {
            let key = format!("{}:{}:{}", chunk.file_path, chunk.start_line, chunk.end_line);
            self.chunks.insert(key, chunk.clone());
        }
        
        // Track files
        for f in &files {
            if let Ok(metadata) = f.metadata() {
                if let Ok(mtime) = metadata.modified() {
                    if let Ok(duration) = mtime.elapsed() {
                        let rel_path = f.strip_prefix(&repo_path)
                            .unwrap_or(f)
                            .to_string_lossy()
                            .into_owned();
                        self.files.insert(rel_path, duration.as_secs_f64());
                    }
                }
            }
        }
        
        let elapsed = start.elapsed();
        
        Python::with_gil(|py| {
            let stats = PyDict::new(py);
            stats.set_item("files_indexed", files.len())?;
            stats.set_item("chunks_created", chunks.len())?;
            stats.set_item("time_ms", elapsed.as_secs_f64() * 1000.0)?;
            stats.set_item("parallel_workers", rayon::current_num_threads())?;
            Ok(stats.into())
        })
    }

    fn get_chunk_count(&self) -> usize {
        self.chunks.len()
    }

    fn search(&self, query: &str, limit: usize) -> PyResult<Py<PyList>> {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        
        // Collect chunks to vector for parallel processing
        let chunks_vec: Vec<ChunkData> = self.chunks.iter().map(|e| e.value().clone()).collect();
        
        let mut results: Vec<(f32, ChunkData)> = chunks_vec
            .par_iter()
            .map(|chunk| {
                let score = self.calculate_relevance(&query_words, chunk);
                (score, chunk.clone())
            })
            .filter(|(score, _)| *score > 0.0)
            .collect();
        
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        results.truncate(limit);
        
        Python::with_gil(|py| {
            let results_list: &PyList = PyList::new(py, std::iter::empty::<&PyDict>());
            for (score, chunk) in results {
                let dict = PyDict::new(py);
                dict.set_item("file", &chunk.file_path)?;
                dict.set_item("score", score)?;
                dict.set_item("code", &chunk.code)?;
                dict.set_item("language", &chunk.language)?;
                dict.set_item("start_line", chunk.start_line)?;
                dict.set_item("end_line", chunk.end_line)?;
                let symbol_names: Vec<&str> = chunk.symbols.iter().map(|s| s.name.as_str()).collect();
                dict.set_item("symbols", symbol_names)?;
                results_list.append(dict)?;
            }
            Ok(results_list.into())
        })
    }

    fn get_stats(&self) -> PyResult<Py<PyDict>> {
        let chunks_vec: Vec<ChunkData> = self.chunks.iter().map(|e| e.value().clone()).collect();
        
        let mut lang_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut type_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        
        for chunk in &chunks_vec {
            *lang_counts.entry(chunk.language.clone()).or_insert(0) += 1;
            for sym in &chunk.symbols {
                *type_counts.entry(sym.type_.clone()).or_insert(0) += 1;
            }
        }
        
        Python::with_gil(|py| {
            let stats = PyDict::new(py);
            stats.set_item("total_chunks", chunks_vec.len())?;
            stats.set_item("total_files", self.files.len())?;
            stats.set_item("by_language", lang_counts)?;
            stats.set_item("by_symbol_type", type_counts)?;
            stats.set_item("parallel_workers", rayon::current_num_threads())?;
            Ok(stats.into())
        })
    }

    fn benchmark(&self, repo_path: &str, iterations: usize) -> PyResult<Py<PyDict>> {
        let repo_path = PathBuf::from(repo_path);
        let mut times = Vec::new();
        
        for _ in 0..iterations {
            self.chunks.clear();
            self.files.clear();
            
            let start = std::time::Instant::now();
            let files: Vec<PathBuf> = self.scan_files_parallel(&repo_path);
            let chunks: Vec<ChunkData> = files
                .par_iter()
                .filter_map(|f| self.index_file(f, &repo_path))
                .collect();
            for chunk in &chunks {
                let key = format!("{}:{}:{}", chunk.file_path, chunk.start_line, chunk.end_line);
                self.chunks.insert(key, chunk.clone());
            }
            times.push(start.elapsed().as_secs_f64());
        }
        
        let avg_time = times.iter().sum::<f64>() / times.len() as f64;
        let min_time = times.iter().cloned().fold(f64::MAX, f64::min);
        let max_time = times.iter().cloned().fold(f64::MIN, f64::max);
        
        Python::with_gil(|py| {
            let stats = PyDict::new(py);
            stats.set_item("iterations", iterations)?;
            stats.set_item("avg_time_ms", avg_time * 1000.0)?;
            stats.set_item("min_time_ms", min_time * 1000.0)?;
            stats.set_item("max_time_ms", max_time * 1000.0)?;
            stats.set_item("chunks_created", self.chunks.len())?;
            stats.set_item("parallel_workers", rayon::current_num_threads())?;
            Ok(stats.into())
        })
    }
}

impl RustIndexer {
    fn scan_files_parallel(&self, repo_path: &Path) -> Vec<PathBuf> {
        let extensions = ["py", "rs", "js", "ts", "go", "java", "cpp", "c", "h", "md", "json", "yaml"];
        let mut files = Vec::new();
        
        fn scan_dir(path: &Path, extensions: &[&str], files: &mut Vec<PathBuf>) {
            if let Ok(entries) = fs::read_dir(path) {
                entries.flatten().for_each(|entry| {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                            if extensions.contains(&ext) {
                                files.push(path);
                            }
                        }
                    } else if path.is_dir() && !path.to_string_lossy().contains(".git") {
                        scan_dir(&path, extensions, files);
                    }
                });
            }
        }
        
        scan_dir(repo_path, &extensions, &mut files);
        files
    }
    
    fn index_file(&self, file_path: &Path, repo_path: &Path) -> Option<ChunkData> {
        let rel_path = file_path.strip_prefix(repo_path)
            .unwrap_or(file_path)
            .to_string_lossy()
            .into_owned();
        
        let language = self.detect_language(file_path);
        let content = fs::read_to_string(file_path).ok()?;
        let lines: Vec<&str> = content.lines().collect();
        
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut start_line = 1;
        let mut line_count = 0;
        
        for (i, line) in lines.iter().enumerate() {
            current_chunk.push_str(line);
            current_chunk.push('\n');
            line_count += 1;
            
            if line_count >= 100 || current_chunk.len() > 5000 {
                let chunk = self.create_chunk(&rel_path, start_line, i + 1, &current_chunk, &language);
                chunks.push(chunk);
                current_chunk.clear();
                start_line = i + 2;
                line_count = 0;
            }
        }
        
        if !current_chunk.is_empty() {
            let chunk = self.create_chunk(&rel_path, start_line, lines.len(), &current_chunk, &language);
            chunks.push(chunk);
        }
        
        chunks.into_iter().next()
    }
    
    fn create_chunk(&self, file_path: &str, start_line: usize, end_line: usize, code: &str, language: &str) -> ChunkData {
        let symbols = self.extract_symbols(code);
        let ast_hash = self.hash_code(code);
        
        ChunkData {
            file_path: file_path.to_string(),
            start_line,
            end_line,
            code: code.to_string(),
            language: language.to_string(),
            symbols,
            ast_hash,
        }
    }
    
    fn detect_language(&self, path: &Path) -> String {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        match ext.as_str() {
            "py" => "python",
            "rs" => "rust",
            "js" | "mjs" | "cjs" => "javascript",
            "ts" | "tsx" => "typescript",
            "go" => "go",
            "java" | "kt" => "java",
            "cpp" | "cc" | "cxx" | "hpp" => "cpp",
            "c" | "h" => "c",
            "md" => "markdown",
            "json" => "json",
            "yaml" | "yml" => "yaml",
            _ => "text",
        }.to_string()
    }
    
    fn extract_symbols(&self, code: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        
        for cap in self.func_regex.captures_iter(code) {
            if let Some(name) = cap.get(1) {
                symbols.push(Symbol {
                    name: name.as_str().to_string(),
                    type_: "function".to_string(),
                    start_line: 0,
                    end_line: 0,
                });
            }
        }
        
        for cap in self.class_regex.captures_iter(code) {
            if let Some(name) = cap.get(1) {
                symbols.push(Symbol {
                    name: name.as_str().to_string(),
                    type_: "class".to_string(),
                    start_line: 0,
                    end_line: 0,
                });
            }
        }
        
        for cap in self.import_regex.captures_iter(code) {
            if let Some(name) = cap.get(1) {
                symbols.push(Symbol {
                    name: name.as_str().to_string(),
                    type_: "import".to_string(),
                    start_line: 0,
                    end_line: 0,
                });
            }
        }
        
        symbols
    }
    
    fn hash_code(&self, code: &str) -> String {
        let mut hasher = DefaultHasher::new();
        code.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
    
    fn calculate_relevance(&self, query_words: &[&str], chunk: &ChunkData) -> f32 {
        let mut score = 0.0;
        let chunk_text = chunk.code.to_lowercase();
        let chunk_words: Vec<&str> = chunk_text.split_whitespace().collect();
        
        for query_word in query_words {
            let chunk_word_count = chunk_words.len();
            
            if chunk_word_count == 0 {
                continue;
            }
            
            let matches: usize = chunk_words.iter()
                .filter(|w| w.contains(query_word))
                .count();
            
            let match_ratio = matches as f32 / chunk_word_count as f32;
            
            if chunk_text.contains(query_word) {
                score += 1.0;
            }
            
            for sym in &chunk.symbols {
                if sym.name.to_lowercase().contains(query_word) {
                    score += 2.0;
                }
            }
            
            score += match_ratio * 0.5;
        }
        
        score
    }
}

/// Initialize the Rust module
#[pymodule]
fn coarch_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustIndexer>()?;
    Ok(())
}
