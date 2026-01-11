"""Setup script for Coarch."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="coarch",
    version="0.1.0",
    author="syedazeez337",
    author_email="syedazeez337@github.com",
    description="Local-first semantic code search engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/syedazeez337/coarch",
    packages=find_packages(exclude=["tests*", "benchs*", "demos*"]),
    python_requires=">=3.8",
    install_requires=[
        "faiss-cpu>=1.7.4",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.2",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "sqlite-utils>=3.34.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "tqdm>=4.65.0",
        "onnxruntime>=1.15.0",
    ],
    extras_require={
        "gpu": ["faiss-gpu", "torch"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "coarch=cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="code search, semantic search, embeddings, faiss, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/syedazeez337/coarch/issues",
        "Source": "https://github.com/syedazeez337/coarch",
        "Documentation": "https://github.com/syedazeez337/coarch/wiki",
    },
)
