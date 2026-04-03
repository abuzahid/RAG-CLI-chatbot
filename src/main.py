#!/usr/bin/env python3
"""
Life Insurance Support AI Agent - CLI Interface
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from src.chat.session import ChatSession
from src.chain.retrieval_chain import RetrievalChain
from src.vectorstore.store import VectorStore
from src.ingestion.loader import load_documents_from_path
from src.ingestion.chunker import DocumentChunker
from src.config import Config, reset_config


class CLI:
    """Command-line interface for the insurance agent"""

    def __init__(self):
        self.session: Optional[ChatSession] = None
        self.chain: Optional[RetrievalChain] = None
        self.vectorstore = VectorStore()

    def initialize(self):
        """Initialize the chat session and chain"""
        try:
            self.session = ChatSession.from_config()
            self.chain = RetrievalChain(session=self.session, vectorstore=self.vectorstore)
            return True
        except Exception as e:
            print(f"Error initializing: {e}")
            return False

    def ingest_documents(self, path: str):
        """Ingest documents from path into vector store"""
        try:
            print(f"Loading documents from: {path}")
            documents = load_documents_from_path(path)

            if not documents:
                print("No documents found.")
                return

            print(f"Loaded {len(documents)} document(s).")

            # Chunk documents
            chunker = DocumentChunker.from_config()
            chunks = chunker.chunk_documents(documents)
            print(f"Created {len(chunks)} chunks.")

            # Add to vector store
            print("Adding to vector store...")
            self.vectorstore.add_documents(chunks)
            print("Documents ingested successfully!")

        except Exception as e:
            print(f"Error during ingestion: {e}")

    def chat_loop(self):
        """Main chat REPL loop"""
        if not self.chain:
            if not self.initialize():
                return

        print("\n" + "="*50)
        print("Life Insurance Support Assistant")
        print("="*50)
        print("Type your questions below.")
        print("Commands: /clear, /help, /exit")
        print("-"*50 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    self.handle_command(user_input)
                    continue

                # Process query
                response = self.chain.query(user_input)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

    def handle_command(self, command: str):
        """Handle CLI commands"""
        cmd = command.lower()

        if cmd == "/exit" or cmd == "/quit":
            print("Goodbye!")
            sys.exit(0)
        elif cmd == "/clear":
            self.session.clear()
            print("Conversation history cleared.\n")
        elif cmd.startswith("/ingest "):
            # Extract path from command
            path = command[8:].strip()  # Remove "/ingest "
            if path:
                self.ingest_documents(path)
            else:
                print("Usage: /ingest <path>\n")
        elif cmd == "/help":
            self.show_help()
        else:
            print(f"Unknown command: {command}")
            print("Type /help for available commands.\n")

    def show_help(self):
        """Show help message"""
        print("\nAvailable commands:")
        print("  /ingest <path>  - Ingest documents from file or directory")
        print("  /clear          - Clear conversation history")
        print("  /help           - Show this help message")
        print("  /exit           - Exit the application")
        print()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Life Insurance Support AI Agent"
    )
    parser.add_argument(
        "--ingest", type=str, metavar="PATH",
        help="Ingest documents from path before starting chat"
    )
    parser.add_argument(
        "--chat", action="store_true",
        help="Start chat mode"
    )

    args = parser.parse_args()

    cli = CLI()

    # Initialize first to validate config
    if not cli.initialize():
        sys.exit(1)

    # Handle ingestion if requested
    if args.ingest:
        cli.ingest_documents(args.ingest)
        if not args.chat:
            return  # Exit after ingestion

    # Either chat mode or default to interactive
    if args.chat or not args.ingest:
        cli.chat_loop()


if __name__ == "__main__":
    main()
