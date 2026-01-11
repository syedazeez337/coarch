#!/usr/bin/env python3
"""Coarch CLI entry point."""

import click
from rich import print

__version__ = "0.1.0"


@click.group()
@click.version_option(__version__)
def main():
    """Coarch - Local-first code search engine."""
    pass


@main.command()
@click.argument("path", type=click.Path(exists=True))
def index(path):
    """Index a repository."""
    print(f"Indexing repository: {path}")


@main.command()
@click.argument("query", type=str)
@click.option("--limit", default=10, help="Maximum results to return")
def search(query, limit):
    """Search for code."""
    print(f"Searching for: {query} (limit: {limit})")


@main.command()
def serve():
    """Start the Coarch API server."""
    print("Starting Coarch server...")


@main.command()
def status():
    """Show index statistics."""
    print("Index status: Ready")


if __name__ == "__main__":
    main()
