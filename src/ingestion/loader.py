import os
from typing import List, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class DocumentLoader:
    """Load documents from various file formats"""

    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf'}

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """Load a single file

        Args:
            file_path: Path to the file

        Returns:
            List of dicts with 'text' and 'metadata' keys
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        # Choose loader based on file type
        if path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding='utf-8')

        documents = loader.load()

        # Clean metadata to only include simple types (str, int, float, bool)
        cleaned_docs = []
        for doc in documents:
            clean_metadata = {"source": str(path)}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool, list)):
                    # Convert to string if it's a list, otherwise keep as is
                    if isinstance(value, list):
                        clean_metadata[key] = str(value)
                    else:
                        clean_metadata[key] = value

            cleaned_docs.append({
                "text": doc.page_content,
                "metadata": clean_metadata
            })

        return cleaned_docs

    def load_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Load all supported files from a directory

        Args:
            directory_path: Path to the directory

        Returns:
            List of dicts with 'text' and 'metadata' keys
        """
        path = Path(directory_path)

        if not path.exists() or not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        all_documents = []

        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    documents = self.load(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    # Skip files that fail to load
                    print(f"Warning: Failed to load {file_path}: {e}")
                    continue

        return all_documents


def load_documents_from_path(path: str) -> List[Dict[str, Any]]:
    """Convenience function to load from file or directory

    Args:
        path: Path to file or directory

    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    loader = DocumentLoader()

    if os.path.isfile(path):
        return loader.load(path)
    elif os.path.isdir(path):
        return loader.load_directory(path)
    else:
        raise ValueError(f"Path is neither file nor directory: {path}")
