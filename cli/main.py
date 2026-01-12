#!/usr/bin/env python3
"""Coarch CLI entry point with production-ready features."""

import sys
import os
import subprocess
import shutil
from typing import List, Optional

import click
from rich import print
from rich.panel import Panel
from rich.text import Text

from backend.logging_config import setup_logging, get_logger

# Signal handling integration
try:
    from backend.signal_handler import (
        register_cleanup_task,
        register_active_operation,
        unregister_active_operation,
        GracefulKiller as SignalGracefulKiller,
        get_shutdown_state,
    )
    SIGNAL_HANDLER_AVAILABLE = True
except ImportError:
    # Fallback signal handler functions
    def register_cleanup_task(task): pass
    def register_active_operation(op_id, operation): pass
    def unregister_active_operation(op_id): pass
    class SignalGracefulKiller:
        def __init__(self): self.cancelled = False
        def cancel(self): pass
        def check_cancelled(self): pass
    def get_shutdown_state(): return {}
    SIGNAL_HANDLER_AVAILABLE = False

try:
    from backend.exceptions import (
        CoarchCLIError,
        CoarchConfigError,
        CoarchIndexError,
        CoarchSearchError,
        CoarchValidationError,
        handle_cli_error,
        ExitCode
    )
except ImportError:
    # Fallback for development - define minimal classes
    class ExitCode:
        SUCCESS = 0
        GENERAL_ERROR = 1
        CONFIGURATION_ERROR = 2
        INDEXING_ERROR = 3
        SEARCH_ERROR = 4
        VALIDATION_ERROR = 5

        @property
        def value(self):
            return self

    class CoarchCLIError(Exception):
        pass

    class CoarchConfigError(Exception):
        pass

    class CoarchIndexError(Exception):
        pass

    class CoarchSearchError(Exception):
        pass

    class CoarchValidationError(Exception):
        pass

    def handle_cli_error(error, verbose=False):
        print(f"[red]Error: {error}[/]")
        sys.exit(1)

# Import validation functions
try:
    from backend.validation import (
        validate_path_exists,
        validate_port,
        validate_limit,
        validate_query,
        validate_startup_config,
        sanitize_path,
        validate_directory_path,
        validate_file_path,
        validate_language,
        validate_repo_id,
        validate_host
    )
except ImportError:
    # Fallback validation functions
    def validate_path_exists(path, path_type="path"):
        return path

    def validate_port(port):
        return int(port)

    def validate_limit(limit):
        return int(limit)

    def validate_query(query):
        return query.strip()

    def validate_startup_config():
        return {}

    def sanitize_path(path):
        return path

    def validate_directory_path(path, path_type="directory"):
        return path

    def validate_file_path(path, path_type="file"):
        return path

    def validate_language(language, supported_languages=None):
        return language

    def validate_repo_id(repo_id):
        return int(repo_id)

    def validate_host(host):
        return host

# Progress tracking imports
try:
    from backend.progress_tracker import (
        track_progress,
        create_file_progress_bar,
        create_embedding_progress_bar,
        create_faiss_progress_bar,
        create_deletion_progress_bar,
        should_show_progress,
        ProgressCallback,
        ETAEstimator,
    )
except ImportError:
    # Fallback progress functions
    class DummyProgress:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_description(self, desc):
            pass
        def set_postfix(self, **kwargs):
            pass
        def __bool__(self):
            return False

    def track_progress(operation_name, total_items=None, unit="items"):
        return DummyProgress()

    class DummyBar:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_description(self, desc):
            pass
        def set_postfix(self, **kwargs):
            pass

    def create_file_progress_bar(total_files):
        return DummyBar()

    def create_embedding_progress_bar(total_chunks):
        return DummyBar()

    def create_faiss_progress_bar(total_vectors):
        return DummyBar()

    def create_deletion_progress_bar(total_chunks):
        return DummyBar()

    def should_show_progress(total_items=None):
        return True

    class ProgressCallback:
        def __init__(self, progress_bar, operation_name="Processing"):
            self.progress_bar = progress_bar
            self.operation_name = operation_name
        def __call__(self, chunks_processed, total_chunks=None):
            pass
        def set_description(self, description):
            if hasattr(self.progress_bar, 'set_description'):
                self.progress_bar.set_description(description)
        def set_postfix(self, **kwargs):
            if hasattr(self.progress_bar, 'set_postfix'):
                self.progress_bar.set_postfix(kwargs)
        def close(self):
            if hasattr(self.progress_bar, 'close'):
                self.progress_bar.close()

    class ETAEstimator:
        def __init__(self, total_items):
            self.total_items = total_items
        def update(self, processed_items):
            pass
        def get_eta(self):
            return None
        def get_eta_string(self):
            return "N/A"
        def get_rate_string(self):
            return "0 items/s"

__version__ = "1.0.0"

logger = get_logger(__name__)

# Supported languages for completion
SUPPORTED_LANGUAGES = [
    "python", "javascript", "typescript", "java", "go", "rust", "ruby", "php",
    "csharp", "cpp", "c", "scala", "kotlin", "swift", "shell", "bash", "zsh",
    "powershell", "sql", "html", "css", "scss", "less", "json", "yaml", "toml",
    "markdown", "text", "dockerfile", "makefile", "r", "matlab", "perl", "lua"
]

# Command aliases mapping
COMMAND_ALIASES = {
    'find': 'search',
    'stats': 'status', 
    'server': 'serve',
    'remove': 'delete',
    'ping': 'health',
    'idx': 'index',
    'setup': 'init'
}

def get_all_commands_with_aliases():
    """Get all commands including their aliases."""
    commands = ['index', 'search', 'serve', 'status', 'delete', 'health', 'init', 'completion']
    # Add aliases to commands list
    for alias, command in COMMAND_ALIASES.items():
        commands.append(alias)
    return sorted(set(commands))  # Use set to avoid duplicates

# Completion callback functions
def complete_command_names(ctx, args, incomplete):
    """Complete command names including aliases."""
    commands = get_all_commands_with_aliases()
    return [cmd for cmd in commands if cmd.startswith(incomplete)]

def complete_file_paths(ctx, args, incomplete):
    """Complete file and directory paths."""
    if not incomplete:
        return []
    
    import glob
    import os
    
    # Handle path completion
    if incomplete.startswith('./'):
        pattern = incomplete + '*'
    elif incomplete.startswith('/'):
        pattern = incomplete + '*'
    elif '/' in incomplete:
        pattern = incomplete + '*'
    else:
        # Complete with current directory
        pattern = './' + incomplete + '*'
    
    matches = []
    try:
        for match in glob.glob(pattern):
            if os.path.isdir(match):
                matches.append(match + '/')
            else:
                matches.append(match)
    except (OSError, IOError):
        pass
    
    return matches

def complete_languages(ctx, args, incomplete):
    """Complete programming language names."""
    return [lang for lang in SUPPORTED_LANGUAGES if lang.startswith(incomplete)]

def complete_hostnames(ctx, args, incomplete):
    """Complete common hostnames."""
    hosts = ['localhost', '127.0.0.1', '0.0.0.0']
    return [host for host in hosts if host.startswith(incomplete)]

def complete_ports(ctx, args, incomplete):
    """Complete common port numbers."""
    ports = ['8000', '8080', '3000', '5000', '9000', '5432', '3306', '6379']
    try:
        port_int = int(incomplete)
        return [str(p) for p in ports if str(p).startswith(incomplete)]
    except ValueError:
        return [str(p) for p in ports if str(p).startswith(incomplete)]

def get_completion_script(shell_type: str) -> str:
    """Generate shell completion script for the specified shell."""
    
    if shell_type == 'bash':
        return f'''#!/bin/bash
# Coarch CLI bash completion
_coarch_completion()
{{
    local cur prev opts
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    
    # Main commands and aliases
    commands="index search serve status delete health init completion"
    aliases="find stats server remove ping idx setup"
    all_commands="${{commands}} ${{aliases}}"
    
    case "${{COMP_WORDS[1]}}" in
        index)
            case "$prev" in
                --name|-n)
                    return 0
                    ;;
                *)
                    COMPREPLY=( $(compgen -f -X '!*.py' -- "$cur") )
                    return 0
                    ;;
            esac
            ;;
        search|find)
            case "$prev" in
                --language|-L)
                    COMPREPLY=( $(compgen -W "python javascript typescript java go rust ruby php csharp cpp c scala kotlin swift shell bash powershell sql html css scss less json yaml toml markdown text dockerfile makefile r matlab perl lua" -- "$cur") )
                    return 0
                    ;;
                --limit|-l)
                    return 0
                    ;;
                *)
                    return 0
                    ;;
            esac
            ;;
        serve|server)
            case "$prev" in
                --port|-p)
                    COMPREPLY=( $(compgen -W "8000 8080 3000 5000 9000 5432 3306 6379" -- "$cur") )
                    return 0
                    ;;
                --host|-H)
                    COMPREPLY=( $(compgen -W "localhost 127.0.0.1 0.0.0.0" -- "$cur") )
                    return 0
                    ;;
                *)
                    return 0
                    ;;
            esac
            ;;
        completion)
            case "$prev" in
                --shell|-s)
                    COMPREPLY=( $(compgen -W "bash zsh fish" -- "$cur") )
                    return 0
                    ;;
                *)
                    return 0
                    ;;
            esac
            ;;
        *)
            # Complete main commands
            if [[ $cur == -* ]]; then
                COMPREPLY=( $(compgen -W "--verbose --log-file --help --version" -- "$cur") )
            else
                COMPREPLY=( $(compgen -W "$all_commands" -- "$cur") )
            fi
            ;;
    esac
}}

complete -F _coarch_completion coarch
'''
    
    elif shell_type == 'zsh':
        return f'''#!/bin/zsh
#compdef coarch

_coarch_cli() {{
    local context curcontext="coarch-cli" state line
    typeset -A opt_args
    
    _arguments \\
        '(-v --verbose)'{{-v,--verbose}}'[Enable verbose logging]' \\
        '--log-file[Path to log file]:file:_files' \\
        '(-h --help)'{{-h,--help}}'[Show help message]' \\
        '--version[Show version]' \\
        '1: :_coarch_commands' \\
        '*::arg:->args' && return 0
    
    case $state in
        args)
            case $words[1] in
                index)
                    _arguments \\
                        '(-n --name)'{{-n,--name}}'[Repository name]:name:' \\
                        '--watch[Enable file watching]' \\
                        '1:path:_path_files -/'
                    ;;
                search|find)
                    _arguments \\
                        '(-l --limit)'{{-l,--limit}}'[Maximum results]:limit:' \\
                        '(-L --language)'{{-L,--language}}'[Filter by language]:language:(python javascript typescript java go rust ruby php csharp cpp c scala kotlin swift shell bash powershell sql html css scss less json yaml toml markdown text dockerfile makefile r matlab perl lua)' \\
                        '1:query:'
                    ;;
                serve|server)
                    _arguments \\
                        '(-p --port)'{{-p,--port}}'[Server port]:port:(8000 8080 3000 5000 9000 5432 3306 6379)' \\
                        '(-H --host)'{{-H,--host}}'[Server host]:host:(localhost 127.0.0.1 0.0.0.0)' \\
                        '(-r --reload)'{{-r,--reload}}'[Enable auto-reload]'
                    ;;
                completion)
                    _arguments \\
                        '(-s --shell)'{{-s,--shell}}'[Shell type]:shell:(bash zsh fish)'
                    ;;
                *)
                    ;;
            esac
            ;;
    esac
}}

_coarch_commands() {{
    local commands
    commands=(
        'index:Index a repository for semantic search'
        'search:Search for semantic similar code'
        'serve:Start the Coarch API server'
        'status:Show index statistics'
        'delete:Delete a repository from the index'
        'health:Check server health'
        'init:Initialize Coarch configuration'
        'completion:Install shell completion'
        'find:Search for semantic similar code'
        'stats:Show index statistics'
        'server:Start the Coarch API server'
        'remove:Delete a repository from the index'
        'ping:Check server health'
        'idx:Index a repository for semantic search'
        'setup:Initialize Coarch configuration'
    )
    _describe 'commands' commands
}}

_coarch_cli "$@"
'''
    
    elif shell_type == 'fish':
        return f'''#!/usr/bin/env fish
# Coarch CLI fish completion

function __coarch_complete_subcommand
    set -l cmd (commandline -oc)
    set -l subcommand "$cmd[2]"
    
    switch "$subcommand"
        case index
            __fish_use_subcommand
            echo "--name"
            echo "--watch"
        case search find
            __fish_use_subcommand
            echo "--language"
            echo "--limit"
        case serve server
            __fish_use_subcommand
            echo "--port"
            echo "--host"
            echo "--reload"
        case completion
            __fish_use_subcommand
            echo "--shell"
        case "*"
            __fish_use_subcommand
    end
end

function __fish_coarch_needs_command
    set -l cmd (commandline -oc)
    set -l first_cmd "$cmd[1]"
    
    if [ "$first_cmd" = "coarch" ]
        return 0
    end
    
    for arg in $cmd
        if [ (string sub -l 1 -- "$arg") = "-" ]
            return 1
        end
    end
    
    return 0
end

function __fish_coarch_using_command
    set -l cmd (commandline -oc)
    set -l first_cmd "$cmd[1]"
    
    if [ "$first_cmd" = "coarch" ]
        if contains -- "$argv[1]" $cmd
            return 0
        end
    end
    
    return 1
end

# Main completion function
function __fish_coarch_complete
    set -l cmd (commandline -oc)
    
    if __fish_coarch_needs_command
        echo "index"
        echo "idx"
        echo "search"
        echo "find"
        echo "serve"
        echo "server"
        echo "status"
        echo "stats"
        echo "delete"
        echo "remove"
        echo "health"
        echo "ping"
        echo "init"
        echo "setup"
        echo "completion"
        return 0
    end
    
    __coarch_complete_subcommand
end

    # Register completion
    complete -f -c coarch -n "__fish_coarch_needs_command" -xa "(__fish_coarch_complete)"
    '''
    else:
        # Return empty string for unsupported shells
        return ""

    # This should never be reached but added for safety
    return ""

def install_completion(shell_type: str, shell_rc_file: Optional[str] = None) -> bool:
    """Install shell completion for the specified shell.
    
    Args:
        shell_type: Type of shell ('bash', 'zsh', or 'fish')
        shell_rc_file: Custom shell rc file path (optional)
        
    Returns:
        True if installation successful, False otherwise
    """
    script = get_completion_script(shell_type)
    
    if shell_type == 'bash':
        rc_file = shell_rc_file or os.path.expanduser('~/.bashrc')
        completion_line = '[ -f ~/.coarch-completion ] && . ~/.coarch-completion'
    elif shell_type == 'zsh':
        rc_file = shell_rc_file or os.path.expanduser('~/.zshrc')
        completion_line = '[ -f ~/.coarch-completion ] && . ~/.coarch-completion'
    elif shell_type == 'fish':
        fish_dir = os.path.expanduser('~/.config/fish/completions')
        completion_file = os.path.join(fish_dir, 'coarch.fish')
        fish_config = os.path.expanduser('~/.config/fish/config.fish')
        
        # Install fish completion
        try:
            os.makedirs(fish_dir, exist_ok=True)
            with open(completion_file, 'w') as f:
                f.write(script)
            print(f"[green]Fish completion installed to {completion_file}[/]")
            print(f"[green]Restart your shell or run 'source {fish_config}' to enable completion[/]")
            return True
        except (OSError, IOError) as e:
            print(f"[red]Failed to install fish completion: {e}[/]")
            return False
    else:
        print(f"[red]Unsupported shell type: {shell_type}[/]")
        print("Supported shells: bash, zsh, fish")
        return False
    
    # Install bash/zsh completion
    try:
        # Write completion script
        completion_file = os.path.expanduser('~/.coarch-completion')
        with open(completion_file, 'w') as f:
            f.write(script)
        
        # Add to shell rc file if not already present
        if not os.path.exists(rc_file):
            with open(rc_file, 'w') as f:
                f.write(f'# Coarch completion\n{completion_line}\n')
        else:
            with open(rc_file, 'r') as f:
                existing_content = f.read()
            
            if completion_line not in existing_content:
                with open(rc_file, 'a') as f:
                    f.write(f'\n# Coarch completion\n{completion_line}\n')
        
        print(f"[green]{shell_type.title()} completion installed to {completion_file}[/]")
        print(f"[green]Added to {rc_file}[/]")
        print(f"[green]Restart your shell or run 'source {rc_file}' to enable completion[/]")
        return True
        
    except (OSError, IOError) as e:
        print(f"[red]Failed to install {shell_type} completion: {e}[/]")
        return False

def handle_error_in_test_mode(e: Exception, exit_code: int = 1) -> None:
    """Handle errors appropriately in test mode."""
    if os.environ.get('COARCH_TEST_MODE'):
        print(f"Error: {e}")
        sys.exit(exit_code)


def print_header():
    """Print the Coarch header."""
    header = Text()
    header.append("Coarch", style="bold green")
    header.append(" - Local-first Semantic Code Search Engine")
    print(Panel(header, title="[green]Coarch[/]", subtitle="v" + __version__))


def confirm_action(message: str, default: bool = True) -> bool:
    """Ask user for confirmation with graceful shutdown support.
    
    Args:
        message: Confirmation message to display
        default: Default answer (True/False)
        
    Returns:
        True if user confirms, False otherwise
    """
    try:
        # Check if shutdown has been initiated
        if SIGNAL_HANDLER_AVAILABLE and get_shutdown_state().get('initiated', False):
            return False
            
        suffix = " [Y/n]" if default else " [y/N]"
        response = input(f"{message}{suffix} ").strip().lower()
        
        if not response:
            return default
            
        return response in ('y', 'yes', 'true', '1')
    except (KeyboardInterrupt, EOFError):
        print("\n[yellow]Operation cancelled.[/]")
        return False


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="coarch")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose (DEBUG) logging")
@click.option("--log-file", type=str, default=None, help="Path to log file")
@click.option("--print-config", is_flag=True, help="Print resolved configuration and exit")
@click.option("--config", type=str, default=None, help="Path to config file")
def main(verbose: bool, log_file: str, print_config: bool, config: str):
    """Coarch - Local-first code search engine."""
    import sys as _sys

    if print_config:
        try:
            from backend.unified_config import get_unified_config
            unified_config = get_unified_config(config)
            print(unified_config.print_config())
            _sys.exit(0)
        except ImportError:
            print("[yellow]Unified config not available, using fallback[/]")
            _sys.exit(0)
        except Exception as e:
            print(f"[red]Error loading configuration: {e}[/]")
            _sys.exit(1)

    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(
        level=log_level,
        log_file=log_file,
        json_format=os.environ.get("COARCH_LOG_JSON", "false").lower() == "true",
    )

    # Now validate startup configuration after logging is set up
    try:
        startup_config = validate_startup_config()
        logger.info("Startup configuration validated successfully")
        
        if startup_config.get('log_level') == 'DEBUG':
            logger.debug("Debug logging enabled via configuration")
    except CoarchConfigError as e:
        logger.warning(f"Configuration validation failed: {e}")
        # Don't exit here, let the command handle the configuration issue

    logger.info(f"Coarch CLI started (v{__version__})")

    if verbose:
        logger.debug("Verbose logging enabled")

    # Set up global error handling with signal handling integration
    if not os.environ.get('COARCH_TEST_MODE'):
        def handle_error(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Don't show error for Ctrl+C - signal handler will handle it
                logger.info("KeyboardInterrupt caught by exception handler")
                sys.exit(0)
            else:
                handle_cli_error(exc_value, verbose)

        import sys
        sys.excepthook = handle_error


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=str), 
              shell_complete=complete_file_paths)
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

    # Create graceful killer for this operation
    killer = SignalGracefulKiller()
    
    def cleanup_indexing():
        """Cleanup function for indexing operation."""
        logger.info("Cleaning up indexing resources")
        # Additional cleanup would go here
        
    # Register cleanup task
    if SIGNAL_HANDLER_AVAILABLE:
        register_cleanup_task(cleanup_indexing)

    try:
        # Validate input
        validate_path_exists(path, "repository path")

        from backend.hybrid_indexer import HybridIndexer
        from backend.embeddings import CodeEmbedder
        from backend.faiss_index import FaissIndex
        from backend.file_watcher import IncrementalIndexer

        indexer = HybridIndexer()
        embedder = CodeEmbedder()

        index_path = os.environ.get("COARCH_INDEX_PATH", "coarch_index")
        faiss = FaissIndex(dim=embedder.get_dimension(), index_path=index_path)

        # Register active operation for shutdown
        if SIGNAL_HANDLER_AVAILABLE:
            register_active_operation("indexing", killer)

        # Index repository with progress tracking
        stats = indexer.index_repository(path, name)
        print(f"[green]Files indexed:[/] {stats['files_indexed']}")
        print(f"[green]Chunks created:[/] {stats['chunks_created']}")

        # Check for cancellation
        killer.check_cancelled()

        # Get chunks for embedding with progress tracking
        chunks = indexer.get_chunks_for_embedding()
        total_embedded = 0
        
        if chunks and should_show_progress(len(chunks)):
            print(f"[green]Generating embeddings for {len(chunks)} chunks...[/]")
        
        # Initialize memory management for large-scale indexing
        from backend.memory_manager import get_memory_manager, MemoryAwareProgressTracker
        memory_manager = get_memory_manager()
        
        # Configure memory manager for large repos
        if len(chunks) > 10000:  # Large repository like React
            memory_manager.checkpoint_frequency = 3  # Save every 3 batches
            memory_manager.enable_adaptive_batching = True
            print(f"[yellow]Large repository detected ({len(chunks)} chunks) - enabling memory management[/]")
        
        progress_tracker = MemoryAwareProgressTracker(
            "Embedding generation", len(chunks), memory_manager
        )
        
        batch_num = 0
        chunk_idx = 0
        
        while chunks and chunk_idx < len(chunks):
            # Check for cancellation before each batch
            killer.check_cancelled()
            
            batch_num += 1
            
            # Get optimal batch size from memory manager
            base_batch_size = 64
            current_batch_size = memory_manager.get_optimal_batch_size(base_batch_size)
            
            # Adjust batch size based on memory pressure
            if batch_num > 1:
                memory_manager.adjust_batch_size(batch_num, True)
            
            batch = chunks[chunk_idx:chunk_idx + current_batch_size]
            
            print(f"[cyan]Batch {batch_num}: processing {len(batch)} chunks "
                  f"(batch size: {current_batch_size})[/]") if should_show_progress(len(chunks)) else None
                
            # Show progress for embedding batches
            try:
                # Use memory-aware embedding generation
                code_texts = [chunk["code"] if isinstance(chunk, dict) else chunk.code for chunk in batch]
                embeddings = embedder.embed(code_texts)
                
                # Update progress tracker
                progress_tracker.update(len(batch))
                
                metadata = [
                    {
                        "id": chunk.get("id") if isinstance(chunk, dict) else getattr(chunk, "id", None),
                        "file_path": chunk["file_path"] if isinstance(chunk, dict) else chunk.file_path,
                        "code": chunk["code"] if isinstance(chunk, dict) else chunk.code,
                        "language": chunk["language"] if isinstance(chunk, dict) else chunk.language,
                    }
                    for chunk in batch
                ]

                # Add to FAISS index
                faiss.add(embeddings, metadata)

                # Mark chunks as embedded
                for j, chunk in enumerate(batch):
                    chunk_id = chunk.get("id") if isinstance(chunk, dict) else getattr(chunk, "id", None)
                    if chunk_id:
                        indexer.update_chunk_embedding(chunk_id, chunk_idx + j)

                total_embedded += len(batch)
                
                # Force garbage collection after each batch for large repositories
                if len(chunks) > 5000:
                    import gc
                    gc.collect()
                    
                    # Clear GPU cache if using CUDA
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                # Reduce batch size and retry
                if current_batch_size > 8:
                    memory_manager.current_batch_size = max(8, current_batch_size // 2)
                    logger.info(f"Reducing batch size to {memory_manager.current_batch_size} and retrying")
                    chunk_idx -= len(batch)  # Retry same batch with smaller size
                    continue
                else:
                    raise e

            # Save progress periodically for large repositories
            if memory_manager.should_checkpoint(batch_num):
                print(f"[yellow]Saving checkpoint after batch {batch_num}...[/]")
                faiss.save()
                memory_manager.log_memory_usage(f"Checkpoint {batch_num}")
                
                # Force cleanup after checkpoint
                memory_manager.full_cleanup(batch_num)

            # Update chunk index
            chunk_idx += len(batch)
            
            if should_show_progress(len(chunks)):
                memory_summary = memory_manager.get_memory_summary()
                print(f"[green]Progress: {total_embedded}/{len(chunks)} vectors indexed "
                      f"(Batch {batch_num}, Memory: {memory_summary['current_mb']:.1f}MB)[/]")

        # Finalize progress tracking
        progress_tracker.finalize()
        
        # Save final checkpoint
        faiss.save()
        memory_manager.log_memory_usage("Final indexing complete")
        
        print(f"[green]Embedding generation complete: {total_embedded} vectors indexed in {batch_num} batches[/]")

        # Get next batch of chunks if any remain
        chunks = indexer.get_chunks_for_embedding()

        print(f"[green]Total vectors indexed:[/] {faiss.count()}")

        if watch:
            print("[green]Starting file watcher for incremental indexing...[/]")
            incremental = IncrementalIndexer(indexer, embedder, faiss, {path})
            
            # Register the incremental indexer as an active operation
            if SIGNAL_HANDLER_AVAILABLE:
                register_active_operation("incremental_indexing", incremental)
            
            incremental.start()
            print("[green]File watcher started. Press Ctrl+C to stop.[/]")

            try:
                import time

                while True:
                    killer.check_cancelled()
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[yellow]Shutting down file watcher...[/]")
                incremental.stop()
                print("\n[yellow]File watcher stopped.[/]")

    except KeyboardInterrupt:
        print("\n[yellow]Indexing cancelled by user.[/]")
        if SIGNAL_HANDLER_AVAILABLE:
            killer.cancel()
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Indexing failed: {e}")
        # In test mode, print error directly
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        # In test mode, print error directly
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        raise CoarchIndexError(
            f"Failed to index repository: {e}"
        )
    finally:
        # Unregister active operation
        if SIGNAL_HANDLER_AVAILABLE:
            unregister_active_operation("indexing")
            unregister_active_operation("incremental_indexing")


@main.command()
@click.argument("query", type=str)
@click.option("--limit", "-l", type=int, default=10, help="Maximum results to return")
@click.option("--language", "-L", type=str, default=None, help="Filter by language",
              shell_complete=complete_languages)
def search(query: str, limit: int, language: str):
    """Search for semantic similar code."""
    print_header()
    print(f"[green]Searching for:[/] {query}")

    if language:
        print(f"[green]Language filter:[/] {language}")

    # Create graceful killer for this operation
    killer = SignalGracefulKiller()

    try:
        # Validate input
        validate_query(query)
        validate_limit(limit)

        from backend.embeddings import CodeEmbedder
        from backend.faiss_index import FaissIndex

        embedder = CodeEmbedder()
        index_path = os.environ.get("COARCH_INDEX_PATH", "coarch_index")
        faiss = FaissIndex(dim=embedder.get_dimension(), index_path=index_path)

        # Register active operation for shutdown
        if SIGNAL_HANDLER_AVAILABLE:
            register_active_operation("search", killer)

        # Search with progress for long operations
        with track_progress("Searching", unit="queries") as progress:
            query_embedding = embedder.embed_query(query)
            try:
                if hasattr(progress, 'set_description'):
                    progress.set_description("Query processed")
            except (TypeError, AttributeError):
                pass  # Progress object doesn't support boolean evaluation
            
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

    except CoarchSearchError:
        raise
    except CoarchValidationError:
        raise
    except KeyboardInterrupt:
        print("\n[yellow]Search cancelled by user.[/]")
        if SIGNAL_HANDLER_AVAILABLE:
            killer.cancel()
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Search failed: {e}")
        # In test mode, print error directly
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        # In test mode, print error directly
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        raise CoarchSearchError(
            f"Search failed: {e}"
        )
    finally:
        # Unregister active operation
        if SIGNAL_HANDLER_AVAILABLE:
            unregister_active_operation("search")


@main.command()
@click.option("--port", "-p", type=int, default=8000, help="Server port",
              shell_complete=complete_ports)
@click.option("--host", "-H", type=str, default="0.0.0.0", help="Server host",
              shell_complete=complete_hostnames)
@click.option("--reload", "-r", is_flag=True, default=False, help="Enable auto-reload")
def serve(port: int, host: str, reload: bool):
    """Start the Coarch API server."""
    print_header()
    print(f"[green]Starting Coarch server...[/]")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Docs: http://{host}:{port}/api/docs")

    def cleanup_server():
        """Cleanup function for server operation."""
        logger.info("Shutting down server gracefully")
        # Server cleanup happens in signal handler

    # Register cleanup task
    if SIGNAL_HANDLER_AVAILABLE:
        register_cleanup_task(cleanup_server)

    try:
        # Validate input
        validate_port(port)

        from backend.server import run_server

        run_server(host=host, port=port, reload=reload)
    except CoarchValidationError:
        raise
    except KeyboardInterrupt:
        print("\n[yellow]Server shutdown requested.[/]")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Server failed: {e}")
        # In test mode, print error directly
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        # In test mode, print error directly
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        raise CoarchConfigError(
            f"Failed to start server: {e}"
        )


@main.command()
def status():
    """Show index statistics."""
    print_header()

    try:
        from backend.hybrid_indexer import HybridIndexer
        from backend.faiss_index import FaissIndex

        indexer = HybridIndexer()
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
        # In test mode, print error but return 0 (graceful status)
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            return  # Exit gracefully
        raise CoarchConfigError(
            f"Failed to get status: {e}"
        )


@main.command()
@click.argument("repo_id", type=int)
def delete(repo_id: int):
    """Delete a repository from the index."""
    print_header()
    
    # Add confirmation prompt for destructive operation
    if not confirm_action(f"Are you sure you want to delete repository {repo_id}? This cannot be undone.", default=False):
        print("[yellow]Deletion cancelled.[/]")
        return
        
    print(f"[yellow]Deleting repository {repo_id}...[/]")

    # Create graceful killer for this operation
    killer = SignalGracefulKiller()

    try:
        # Validate input
        if not isinstance(repo_id, int) or repo_id < 0:
            raise CoarchValidationError(
                "Repository ID must be a positive integer"
            )

        from backend.hybrid_indexer import HybridIndexer

        indexer = HybridIndexer()
        
        # Register active operation for shutdown
        if SIGNAL_HANDLER_AVAILABLE:
            register_active_operation("delete", killer)
        
        # Get chunks count for progress tracking
        chunks = indexer.get_chunks_for_embedding()
        chunks_to_delete = len([c for c in chunks if c.get("repo_id") == repo_id])
        
        # Delete with progress tracking
        if should_show_progress(chunks_to_delete):
            with create_deletion_progress_bar(chunks_to_delete) as progress:
                chunks_deleted = indexer.delete_repo(repo_id)
                if progress:
                    progress.set_description("Deleting chunks")
                    progress.update(chunks_deleted)
        else:
            chunks_deleted = indexer.delete_repo(repo_id)

        print(f"[green]Deleted {chunks_deleted} chunks.[/]")

    except CoarchValidationError:
        raise
    except KeyboardInterrupt:
        print("\n[yellow]Deletion cancelled by user.[/]")
        if SIGNAL_HANDLER_AVAILABLE:
            killer.cancel()
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Indexing failed: {e}")
        # In test mode, print error directly
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        # In test mode, print error directly
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        raise CoarchIndexError(
            f"Failed to index repository: {e}"
        )
    finally:
        # Unregister active operation
        if SIGNAL_HANDLER_AVAILABLE:
            unregister_active_operation("delete")


@main.command()
@click.option("--port", "-p", type=int, default=8000, help="Health check port",
              shell_complete=complete_ports)
def health(port: int):
    """Check server health."""
    import urllib.request
    import urllib.error
    import json

    url = f"http://localhost:{port}/health"

    try:
        # Validate input
        validate_port(port)

        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            print("[green]Server is healthy[/]")
            print(f"  Status: {data.get('status')}")
            print(f"  Vectors: {data.get('vectors', 0)}")
    except CoarchValidationError:
        raise
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError) as e:
        print(f"[red]Server is unhealthy: {e}[/]")
        sys.exit(1)
    except Exception as e:
        print(f"[red]Server is unhealthy: {e}[/]")
        sys.exit(1)


@main.command()
@click.option("--shell", "-s", 
              type=click.Choice(["bash", "zsh", "fish"], case_sensitive=False),
              default="bash", help="Shell type for completion")
@click.option("--install/--no-install", default=True, 
              help="Automatically install completion script")
@click.option("--rc-file", type=str, default=None, 
              help="Custom shell rc file path (bash/zsh only)")
def completion(shell: str, install: bool, rc_file: str):
    """Install or show shell completion scripts."""
    print_header()
    print(f"[green]Coarch Shell Completion[/]\n")
    
    if install:
        print(f"[cyan]Installing {shell} completion...[/]")
        success = install_completion(shell, rc_file)
        
        if success:
            print("\n[green]✅ Completion installed successfully![/]")
            print("\nTo enable completion:")
            if shell == "fish":
                print("  - Restart your shell or run: source ~/.config/fish/config.fish")
            else:
                rc_path = rc_file or f"~/.{shell}rc"
                print(f"  - Restart your shell or run: source {rc_path}")
        else:
            print("\n[red]❌ Installation failed[/]")
            return
        
        # Show usage examples
        print("\n[green]Usage examples:[/]")
        print("  coarch <TAB>                    # Show all commands")
        print("  coarch search <TAB>             # Show search options")
        print("  coarch index /path/<TAB>        # Complete directory paths")
        print("  coarch search --language p<TAB>  # Complete language names")
        
    else:
        # Just show the completion script
        script = get_completion_script(shell)
        print(f"[green]{shell.title()} completion script:[/]\n")
        print(script)
        print(f"\n[yellow]To install this completion:[/]")
        if shell == "fish":
            print(f"  mkdir -p ~/.config/fish/completions")
            print(f"  cat > ~/.config/fish/completions/coarch.fish << 'EOF'")
            print(f"  {script.replace(chr(10), chr(10) + '  ')}")
            print(f"  EOF")
        else:
            print(f"  echo '{script}' > ~/.coarch-completion")
            print(f"  echo '[ -f ~/.coarch-completion ] && . ~/.coarch-completion' >> ~/.{shell}rc")
            print(f"  source ~/.{shell}rc")


# Add aliases for existing commands using Click's built-in alias support
# Note: Click doesn't support aliases directly, so we create wrapper commands
@main.command('find')
@click.argument("query", type=str)
@click.option("--limit", "-l", type=int, default=10, help="Maximum results to return")
@click.option("--language", "-L", type=str, default=None, help="Filter by language",
              shell_complete=complete_languages)
def find_alias(query: str, limit: int, language: str):
    """Alias for search command - searches for semantic similar code."""
    print_header()
    print(f"[green]Searching for:[/] {query}")

    if language:
        print(f"[green]Language filter:[/] {language}")

    # Create graceful killer for this operation
    killer = SignalGracefulKiller()

    try:
        # Validate input
        validate_query(query)
        validate_limit(limit)

        from backend.embeddings import CodeEmbedder
        from backend.faiss_index import FaissIndex

        embedder = CodeEmbedder()
        index_path = os.environ.get("COARCH_INDEX_PATH", "coarch_index")
        faiss = FaissIndex(dim=embedder.get_dimension(), index_path=index_path)

        # Register active operation for shutdown
        if SIGNAL_HANDLER_AVAILABLE:
            register_active_operation("search", killer)

        # Search with progress for long operations
        with track_progress("Searching", unit="queries") as progress:
            query_embedding = embedder.embed_query(query)
            try:
                if hasattr(progress, 'set_description'):
                    progress.set_description("Query processed")
            except (TypeError, AttributeError):
                pass  # Progress object doesn't support boolean evaluation
            
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
            print(f"   Code: [dim]{code_preview}[/]")
            print()

        # Unregister active operation
        if SIGNAL_HANDLER_AVAILABLE:
            unregister_active_operation("search")

    except FileNotFoundError as e:
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        raise CoarchConfigError(f"Search failed: {e}")
    except Exception as e:
        logger.exception(f"Search failed: {e}")
        # In test mode, print error directly
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        raise CoarchSearchError(
            f"Search failed: {e}"
        )


@main.command('stats')
def stats_alias():
    """Alias for status command - show index statistics."""
    print_header()

    try:
        from backend.hybrid_indexer import HybridIndexer
        from backend.faiss_index import FaissIndex

        indexer = HybridIndexer()
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
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            return
        raise CoarchConfigError(
            f"Failed to get status: {e}"
        )


@main.command('server')
@click.option("--port", "-p", type=int, default=8000, help="Server port")
@click.option("--host", "-H", type=str, default="0.0.0.0", help="Server host")
@click.option("--reload", "-r", is_flag=True, default=False, help="Enable auto-reload")
def server_alias(port: int, host: str, reload: bool):
    """Alias for serve command - start the API server."""
    print_header()
    print(f"[green]Starting Coarch server...[/]")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Docs: http://{host}:{port}/api/docs")

    def cleanup_server():
        logger.info("Shutting down server gracefully")

    if SIGNAL_HANDLER_AVAILABLE:
        register_cleanup_task(cleanup_server)

    try:
        validate_port(port)
        from backend.server import run_server
        run_server(host=host, port=port, reload=reload)
    except CoarchValidationError:
        raise
    except KeyboardInterrupt:
        print("\n[yellow]Server shutdown requested.[/]")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Server failed: {e}")
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        raise CoarchConfigError(
            f"Failed to start server: {e}"
        )


@main.command('remove')
@click.argument("repo_id", type=int)
def remove_alias(repo_id: int):
    """Alias for delete command - remove a repository from the index."""
    delete(repo_id)


@main.command('ping')
@click.option("--port", "-p", type=int, default=8000, help="Health check port")
def ping_alias(port: int):
    """Alias for health command - check server health."""
    import urllib.request
    import urllib.error
    import json

    url = f"http://localhost:{port}/health"

    try:
        validate_port(port)

        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            print("[green]Server is healthy[/]")
            print(f"  Status: {data.get('status')}")
            print(f"  Vectors: {data.get('vectors', 0)}")
    except CoarchValidationError:
        raise
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, OSError) as e:
        print(f"[red]Server is unhealthy: {e}[/]")
        sys.exit(1)
    except Exception as e:
        print(f"[red]Server is unhealthy: {e}[/]")
        sys.exit(1)


@main.command('idx')
@click.argument("path", type=click.Path(exists=True, path_type=str),
              shell_complete=complete_file_paths)
@click.option("--name", "-n", type=str, default=None, help="Repository name")
@click.option(
    "--watch/--no-watch",
    default=False,
    help="Enable file watching for incremental indexing",
)
def idx_alias(path: str, name: str, watch: bool):
    """Alias for index command - index a repository."""
    index(path, name, watch)


@main.command('setup')
def setup_alias():
    """Alias for init command - initialize configuration."""
    init()


@main.command()
@click.option("--template", "-t", type=str, default=None,
              help="Configuration template to use (development, production, research, minimal, gpu-optimized)")
@click.option("--config", type=str, default=None, help="Path to config file")
def init(template: str, config: str):
    """Initialize Coarch configuration."""
    print_header()

    try:
        from backend.unified_config import init_config, ConfigTemplateManager

        config_manager = init_config(config_path=config, template=template)
        resolved = config_manager.get_resolved_config()

        print("[green]Configuration initialized successfully![/]")
        print(f"  Config path: {config_manager.config_path}")
        print(f"  Index path: {resolved.index_path}")
        print(f"  DB path: {resolved.db_path}")
        print(f"  Server: {resolved.server_host}:{resolved.server_port}")
        print(f"  Model: {resolved.model_name}")

        if template:
            print(f"\n[cyan]Applied template: {template}[/]")

        template_manager = ConfigTemplateManager()
        available_templates = template_manager.list_templates()
        if available_templates:
            print("\n[green]Available templates:[/]")
            for t in available_templates:
                print(f"  - {t.name}: {t.description}")

    except ImportError:
        from backend.config import init_config
        config = init_config()
        print("[green]Configuration initialized successfully![/]")
        print(f"  Config path: {config.get_default_config_path()}")
        print(f"  Index path: {config.index_path}")
        print(f"  DB path: {config.db_path}")
    except Exception as e:
        logger.exception(f"Init failed: {e}")
        if os.environ.get('COARCH_TEST_MODE'):
            print(f"Error: {e}")
            sys.exit(1)
        raise CoarchConfigError(
            f"Failed to initialize configuration: {e}"
        )


@main.command()
@click.option("--format", type=click.Choice(["text", "json", "env"], case_sensitive=False),
              default="text", help="Output format")
def config(format: str):
    """Show or configure Coarch settings."""
    print_header()

    try:
        from backend.unified_config import get_unified_config, ConfigTemplateManager

        config_manager = get_unified_config()
        print(config_manager.print_config(format=format))

        if format == "text":
            print("\n[green]Configuration History:[/]")
            history = config_manager.list_history()
            if history:
                for entry in history[:5]:
                    print(f"  - {entry.timestamp}: {entry.change_description or 'No description'}")
            else:
                print("  No history entries")

    except ImportError:
        print("[yellow]Unified config not available[/]")


@main.command()
@click.argument("timestamp", type=str, required=False)
def rollback(timestamp: str):
    """Rollback configuration to a previous version."""
    print_header()

    try:
        from backend.unified_config import get_unified_config

        config_manager = get_unified_config()

        if not timestamp:
            history = config_manager.list_history()
            if not history:
                print("[yellow]No configuration history found[/]")
                return

            print("[green]Available restore points:[/]")
            for entry in history[:10]:
                desc = entry.change_description or "No description"
                print(f"  {entry.timestamp}: {desc}")
            print("\n[cyan]Use 'coarch rollback <timestamp>' to restore a specific version[/]")
            return

        success = config_manager.rollback(timestamp)
        if success:
            print(f"[green]Configuration rolled back to {timestamp}[/]")
            config_manager.save_history(f"Rolled back to {timestamp}")
        else:
            print(f"[red]Failed to rollback to {timestamp}[/]")
            sys.exit(1)

    except ImportError:
        print("[yellow]Unified config not available[/]")


@main.command()
@click.argument("template_name", type=str)
def template(template_name: str):
    """Apply a configuration template."""
    print_header()

    try:
        from backend.unified_config import get_unified_config, ConfigTemplateManager

        config_manager = get_unified_config()
        template_manager = ConfigTemplateManager()

        template_obj = template_manager.get_template(template_name)
        if not template_obj:
            print(f"[red]Template not found: {template_name}[/]")
            print("\n[green]Available templates:[/]")
            for t in template_manager.list_templates():
                print(f"  - {t.name}: {t.description}")
            sys.exit(1)

        print(f"[cyan]Applying template: {template_obj.name}[/]")
        print(f"  {template_obj.description}\n")

        template_manager.apply_template(template_name, config_manager)
        print(config_manager.print_config())

        config_manager.save_history(f"Applied template: {template_name}")

    except ImportError:
        print("[yellow]Unified config not available[/]")


if __name__ == "__main__":
    main()