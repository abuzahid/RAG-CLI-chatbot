# Life Insurance Support AI Agent

A conversational AI assistant that helps users with life insurance-related inquiries using RAG (Retrieval-Augmented Generation).

## Features

- **Conversational CLI Interface**: Interactive chat experience with context awareness
- **RAG-based Answers**: Accurate responses from ingested knowledge base
- **Configurable Data Ingestion**: Add your own documents (PDF, TXT, MD)
- **Conversation Memory**: Maintains context across multiple turns
- **Modular Architecture**: Clean, extensible codebase

## Tech Stack

- Python 3.10+
- LangChain 0.3+
- OpenAI API (GPT-4o-mini)
- ChromaDB (Vector Store)
- PyPDF (Document Loading)

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Installation

### Windows

1. **Clone or download the repository**

```bash
cd RAG-CLI-chatbot
```

2. **Create virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-api-key-here
```

Or use `.env.example` as a template:

```bash
copy .env.example .env
# Then edit .env with your actual API key
```

### Linux/Mac

1. **Clone the repository**

```bash
git clone <repository-url>
cd life-insurance-agent
```

2. **Create virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

```bash
cp .env.example .env
# Then edit .env with your actual API key
```

## Usage

### Quick Start

1. **Ingest the sample knowledge base**

```bash
python src/main.py --ingest data/knowledge/sample.txt
```

2. **Start chatting**

```bash
python src/main.py --chat
```

### One-line command (ingest + chat)

```bash
python src/main.py --ingest data/knowledge/sample.txt --chat
```

### CLI Commands During Chat

| Command | Description |
|---------|-------------|
| `/ingest <path>` | Ingest documents from file or directory |
| `/clear` | Clear conversation history |
| `/help` | Show available commands |
| `/exit` | Exit the application |

### Example Conversation

```
==================================================
Life Insurance Support Assistant
==================================================
Type your questions below.
Commands: /clear, /help, /exit
--------------------------------------------------

You: What is term life insurance?

Assistant: Term life insurance provides coverage for a specific period,
typically 10, 20, or 30 years. It is the most affordable type of life
insurance and pays a death benefit if the insured passes away during the term.

Key features include:
- Fixed premium payments for the term period
- Death benefit paid to beneficiaries if the insured dies during the term
- No cash value accumulation

You: How does it differ from whole life?

Assistant: Whole life insurance is a type of permanent insurance that provides
coverage for your entire lifetime (as long as premiums are paid). Key differences:

- Term life: Temporary coverage (10-30 years), more affordable, no cash value
- Whole life: Lifetime coverage, level premiums, builds cash value over time
```

## Configuration

Edit `config.yaml` to customize settings:

```yaml
llm:
  model: gpt-4o-mini       # OpenAI model to use
  temperature: 0.7          # Response randomness (0-1)
  max_tokens: 500           # Maximum response length

retrieval:
  chunk_size: 1000          # Document chunk size
  chunk_overlap: 200        # Overlap between chunks
  top_k: 3                  # Number of relevant chunks to retrieve

conversation:
  max_history: 10           # Number of conversation turns to remember
```

## Adding Your Own Documents

1. Place your documents (PDF, TXT, or MD) in any directory
2. Ingest them:

```bash
python src/main.py --ingest path/to/your/documents
```

3. Start chatting - the AI will use your documents!

## Project Structure

```
life-insurance-agent/
├── src/
│   ├── main.py                 # CLI entry point
│   ├── config.py               # Configuration loader
│   ├── chain/                  # LangChain components
│   │   ├── retrieval_chain.py  # RAG chain implementation
│   │   └── prompts.py          # System prompts
│   ├── vectorstore/            # ChromaDB wrapper
│   │   ├── embedder.py         # OpenAI embeddings
│   │   └── store.py            # Vector database operations
│   ├── ingestion/              # Document processing
│   │   ├── loader.py           # File loaders (PDF, TXT, MD)
│   │   └── chunker.py          # Text splitting
│   └── chat/                   # Conversation management
│       └── session.py          # In-memory history
├── data/
│   └── knowledge/              # Knowledge base documents
│       └── sample.txt          # Sample insurance knowledge
├── tests/                      # Unit tests
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variables template
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Test Coverage

- **Config tests**: YAML and environment variable loading
- **Vector store tests**: ChromaDB operations
- **Ingestion tests**: Document loading and chunking
- **Chat tests**: Conversation history management
- **Chain tests**: RAG pipeline

### Architecture Overview

```
User Query
    ↓
Chat Session (maintains history)
    ↓
Retrieval Chain (LCEL)
    ├─→ Vector Store (ChromaDB) → Retrieves relevant chunks
    └─→ LLM (OpenAI) → Generates response with context
    ↓
Response + Update History
```

## Troubleshooting

### "OPENAI_API_KEY not found"
- Ensure your `.env` file exists in the project root
- Verify your API key is set correctly

### Import errors
- Make sure the virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### Documents not being found
- Check the file path is correct
- Verify file format is supported (.pdf, .txt, .md)
- Ensure you have read permissions for the file

### ChromaDB errors
- Delete the `data/chroma/` directory and re-ingest documents
- Ensure you have write permissions in the project directory

## License

MIT License

## Author

Built as an AI Engineer Skill Test submission.
