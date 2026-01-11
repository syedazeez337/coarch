#!/usr/bin/env python3
"""Coarch CLI entry point with production-ready features."""

import sys
import os

import click
from rich import print
from rich.panel import Panel
from rich.text import Text

from .logging_config import setup_logging, get_logger

__version__ = "1.0.0"

logger = get_logger(__name__)


def print_header():
    """Print the Coarch header."""
    header = Text()
    header.append("Coarch", style="bold green", justify="center")
    header.append(" - Local-first Semantic Code Search Engine", justify="center")
    print(Panel(header, title="[green]Coarch[/]", subtitle="v" + __version__))


@click.group()
@click.version_option(__version__, prog_name="coarch")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose (DEBUG) logging")
@click.option("--log-file", type=str, default=None, help="Path to log file")
def main(verbose: bool, log_file: str):
    """Coarch - Local-first code search engine."""
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(
        level=log_level,
        log_file=log_file,
        json_format=os.environ.get("COARCH_LOG_JSON", "false").lower() == "true",
    )

    logger.info(f"Coarch CLI started (v{__version__})")

    if verbose:
        logger.debug("Verbose logging enabled")


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=str))
@click.option("--name", "-n", type=str, default=None, help="Repository name")
@click.option(
    "--watch/--no-watch",
    default=False,
    help="Enable file watching for incremental indexing",
)
def index(path: str, name: str, watch: bool):
    """Index a repository for semantic search."""
    print_header()
    print(f"[green]Indexing repository:[/] {path}")

    try:
        from .indexer import RepositoryIndexer
        from .embeddings import CodeEmbedder
        from .faiss_index import FaissIndex
        from .file_watcher import IncrementalIndexer

        indexer = RepositoryIndexer()
        embedder = CodeEmbedder()

        index_path = os.environ.get("COARCH_INDEX_PATH", "coarch_index")
        faiss = FaissIndex(dim=embedder.get_dimension(), index_path=index_path)

        stats = indexer.index_repository(path, name)
        print(f"[green]Files indexed:[/] {stats['files_indexed']}")
        print(f"[green]Chunks created:[/] {stats['chunks_created']}")

        chunks = indexer.get_chunks_for_embedding()
        if chunks:
            print("[green]Generating embeddings...[/]")

            code_texts = [chunk.code for chunk in chunks]
            embeddings = embedder.embed(code_texts)

            metadata = [
                {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "code": chunk.code,
                    "language": chunk.language,
                }
                for chunk in chunks
            ]

            faiss.add(embeddings, metadata)
            faiss.save()

            print(f"[green]Total vectors indexed:[/] {faiss.count()}")

        if watch:
            print("[green]Starting file watcher for incremental indexing...[/]")
            incremental = IncrementalIndexer(indexer, embedder, faiss, {path})
            incremental.start()
            print("[green]File watcher started. Press Ctrl+C to stop.[/]")

            try:
                import time

                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                incremental.stop()
                print("\n[yellow]File watcher stopped.[/]")

    except Exception as e:
        logger.exception(f"Indexing failed: {e}")
        print(f"[red]Error: {e}[/]")
        sys.exit(1)


@main.command()
@click.argument("query", type=str)
@click.option("--limit", "-l", type=int, default=10, help="Maximum results to return")
@click.option("--language", "-L", type=str, default=None, help="Filter by language")
def search(query: str, limit: int, language: str):
    """Search for semantic similar code."""
    print_header()
    print(f"[green]Searching for:[/] {query}")

    if language:
        print(f"[green]Language filter:[/] {language}")

    try:
        from .embeddings import CodeEmbedder
        from .faiss_index import FaissIndex

        embedder = CodeEmbedder()
        index_path = os.environ.get("COARCH_INDEX_PATH", "coarch_index")
        faiss = FaissIndex(dim=embedder.get_dimension(), index_path=index_path)

        query_embedding = embedder.embed_query(query)
        results = faiss.search(query_embedding, k=limit)

        if not results:
            print("[yellow]No results found.[/]")
            return

        print(f"\n[green]Found {len(results)} results:[/]\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. [cyan]{result.file_path}[/]")
            print(f"   Lines {result.start_line}-{result.end_line}")
            print(f"   Score: {result.score:.4f}")
            print(f"   Language: {result.language}")
            code_preview = (
                result.code[:100] + "..." if len(result.code) > 100 else result.code
            )
            print(f"   Code: {code_preview}")
            print()

    except Exception as e:
        logger.exception(f"Search failed: {e}")
        print(f"[red]Error: {e}[/]")
        sys.exit(1)


@main.command()
@click.option("--port", "-p", type=int, default=8000, help="Server port")
@click.option("--host", "-H", type=str, default="0.0.0.0", help="Server host")
@click.option("--reload", "-r", is_flag=True, default=False, help="Enable auto-reload")
def serve(port: int, host: str, reload: bool):
    """Start the Coarch API server."""
    print_header()
    print(f"[green]Starting Coarch server...[/]")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Docs: http://{host}:{port}/api/docs")

    try:
        from .server import run_server

        run_server(host=host, port=port, reload=reload)
    except Exception as e:
        logger.exception(f"Server failed: {e}")
        print(f"[red]Error: {e}[/]")
        sys.exit(1)


@main.command()
def status():
    """Show index statistics."""
    print_header()

    try:
        from .indexer import RepositoryIndexer
        from .faiss_index import FaissIndex

        indexer = RepositoryIndexer()
        stats = indexer.get_stats()

        print("[green]Index Statistics:[/]")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Total repos: {stats['total_repos']}")

        if stats.get("by_language"):
            print("\n[green]By Language:[/]")
            for lang, count in stats["by_language"].items():
                print(f"  {lang}: {count}")

        index_path = os.environ.get("COARCH_INDEX_PATH", "coarch_index")
        faiss = FaissIndex(index_path=index_path)
        print(f"\n[green]Vector Index:[/]")
        print(f"  Vectors: {faiss.count()}")

    except Exception as e:
        logger.exception(f"Status check failed: {e}")
        print(f"[red]Error: {e}[/]")


@main.command()
@click.argument("repo_id", type=int)
def delete(repo_id: int):
    """Delete a repository from the index."""
    print_header()
    print(f"[yellow]Deleting repository {repo_id}...[/]")

    try:
        from .indexer import RepositoryIndexer

        indexer = RepositoryIndexer()
        chunks_deleted = indexer.delete_repo(repo_id)

        print(f"[green]Deleted {chunks_deleted} chunks.[/]")

    except Exception as e:
        logger.exception(f"Delete failed: {e}")
        print(f"[red]Error: {e}[/]")
        sys.exit(1)


@main.command()
@click.option("--port", "-p", type=int, default=8000, help="Health check port")
def health(port: int):
    """Check server health."""
    import urllib.request
    import json

    url = f"http://localhost:{port}/health"

    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            print("[green]Server is healthy[/]")
            print(f"  Status: {data.get('status')}")
            print(f"  Vectors: {data.get('vectors', 0)}")
    except Exception as e:
        print(f"[red]Server is unhealthy: {e}[/]")
        sys.exit(1)


@main.command()
def init():
    """Initialize Coarch configuration."""
    print_header()

    from .config import init_config

    try:
        config = init_config()
        print("[green]Configuration initialized successfully![/]")
        print(f"  Config path: {config.get_default_config_path()}")
        print(f"  Index path: {config.index_path}")
        print(f"  DB path: {config.db_path}")
    except Exception as e:
        logger.exception(f"Init failed: {e}")
        print(f"[red]Error: {e}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
