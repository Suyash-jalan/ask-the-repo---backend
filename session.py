"""
RepoSession — ephemeral, per‑user codebase analysis session.

Wraps the clone → index → agent pipeline so every user gets an isolated
ChromaDB EphemeralClient, a private temp directory, and a dedicated
LangGraph agent. Call destroy() (or let the janitor do it) to wipe
everything.
"""

import os
import re
import json
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


CODE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.scss',
    '.java', '.cpp', '.c', '.h', '.go', '.php', '.rb', '.rs',
    '.swift', '.kt', '.dart', '.lua',
}

IGNORE_DIRS = {
    '.git', '.venv', 'venv', 'node_modules', 'dist', 'build', '__pycache__',
}

docs_take = {'.md', '.txt', '.rst'}

# Module‑level singleton — loaded lazily on first use, reused by every session
_EMBED_FN = None
_embed_lock = threading.Lock()


def get_embed_fn():
    """Lazy-load the embedding function so the server starts fast."""
    global _EMBED_FN
    if _EMBED_FN is None:
        with _embed_lock:
            if _EMBED_FN is None:
                _EMBED_FN = SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
    return _EMBED_FN

# Tree‑sitter Python parser (language object is cheap & thread‑safe)
PY_LANGUAGE = Language(tspython.language())

# Limits
MAX_FILES = 500
MAX_FILE_BYTES = 200_000
FALLBACK_BLOCK_LINES = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relpath_posix(abs_path: str, base: str) -> str:
    """Return a forward‑slash relative path, never leaking server dirs."""
    return os.path.relpath(abs_path, base).replace("\\", "/")


def _walk_repo(path: str) -> list:
    """Collect code files, respecting IGNORE_DIRS and MAX_FILES."""
    files = []
    for root, dirs, filenames in os.walk(path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in CODE_EXTENSIONS:
                full = os.path.join(root, fname)
                if os.path.getsize(full) <= MAX_FILE_BYTES:
                    files.append(full)
                    if len(files) >= MAX_FILES:
                        return files
    return files


def _extract_python_chunks(file_path: str, repo_path: str) -> list:
    """Parse a Python file with tree‑sitter and return function/class chunks."""
    parser = Parser(PY_LANGUAGE)
    with open(file_path, 'rb') as f:
        source_code = f.read()
    tree = parser.parse(source_code)
    chunks = []

    def _node_name(node):
        ident = node.child_by_field_name("name")
        return ident.text.decode("utf-8") if ident else "<unknown>"

    def _walk(node):
        if node.type in ("function_definition", "class_definition"):
            code = source_code[node.start_byte:node.end_byte].decode('utf-8')
            rel = _relpath_posix(file_path, repo_path)
            chunks.append({
                'file': rel,
                'type': node.type,
                'name': _node_name(node),
                'code': code,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
            })
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return chunks


def _extract_fallback_chunks(file_path: str, repo_path: str) -> list:
    """Fallback chunker: 60‑line blocks for non‑Python code files."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception:
        return []

    rel = _relpath_posix(file_path, repo_path)
    chunks = []
    for i in range(0, len(lines), FALLBACK_BLOCK_LINES):
        block = lines[i:i + FALLBACK_BLOCK_LINES]
        code = ''.join(block)
        if not code.strip():
            continue
        chunks.append({
            'file': rel,
            'type': 'block',
            'name': f'block_{i // FALLBACK_BLOCK_LINES}',
            'code': code,
            'start_line': i + 1,
            'end_line': min(i + FALLBACK_BLOCK_LINES, len(lines)),
        })
    return chunks


def split_by_markdown_headers(text: str) -> list:
    """Split markdown text into sections by headers."""
    sections = []
    current_heading = "Introduction"
    current_content = []

    for line in text.split('\n'):
        header_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if header_match:
            if current_content:
                sections.append({
                    'heading': current_heading,
                    'content': '\n'.join(current_content).strip(),
                })
            current_heading = header_match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append({
            'heading': current_heading,
            'content': '\n'.join(current_content).strip(),
        })
    return sections


def _get_commit_history(repo_path: str, max_commits: int = 500) -> list:
    """Retrieve commit history via subprocess (no GitPython)."""
    result = subprocess.run(
        ['git', '-C', repo_path, 'log', f'-{max_commits}',
         '--pretty=format:%H|%an|%ad|%s', '--date=short'],
        capture_output=True, text=True, timeout=30,
    )
    commits = []
    for line in result.stdout.split('\n'):
        parts = line.split('|', 3)
        if len(parts) == 4:
            sha, author, date, message = parts
            commits.append({'sha': sha, 'author': author, 'date': date, 'message': message})
    return commits


# ---------------------------------------------------------------------------
# RepoSession
# ---------------------------------------------------------------------------

class RepoSession:
    """One user's ephemeral codebase‑analysis session."""

    def __init__(self, repo_url: str, groq_api_key: str,
                 groq_model: str = "qwen/qwen3.8-27b"):
        self.id: str = uuid.uuid4().hex
        self.repo_url: str = repo_url
        self.groq_api_key: str = groq_api_key
        self.groq_model: str = groq_model

        self.workdir: str = tempfile.mkdtemp(prefix=f"repo_{self.id}_")
        self.repo_path: str = os.path.join(self.workdir, "repo")

        self.client: chromadb.ClientAPI | None = None
        self.collection = None
        self.all_chunks: list = []
        self.agent = None

        self.status: str = "cloning"   # cloning | indexing | ready | error
        self.error: str | None = None
        self.last_active: float = time.time()

    # ------------------------------------------------------------------ build
    def build(self):
        """Clone → index → build agent.  Runs in a background thread."""
        try:
            self._clone()
            self.status = "indexing"
            self._index_code()
            self._index_docs()
            self._index_commits()
            self._build_agent()
            self.status = "ready"
        except Exception as exc:
            self.status = "error"
            self.error = str(exc)

    def _clone(self):
        subprocess.run(
            ['git', 'clone', '--filter=blob:none', '--depth', '300',
             '--', self.repo_url, self.repo_path],
            capture_output=True, text=True, timeout=180, check=True,
        )

    # ------------------------------------------------------------ index code
    def _index_code(self):
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection(
            name=f"codebase_{self.id}",
            embedding_function=get_embed_fn(),
        )

        code_files = _walk_repo(self.repo_path)
        for fpath in code_files:
            try:
                ext = os.path.splitext(fpath)[1].lower()
                if ext == '.py':
                    chunks = _extract_python_chunks(fpath, self.repo_path)
                else:
                    chunks = _extract_fallback_chunks(fpath, self.repo_path)
                self.all_chunks.extend(chunks)
            except Exception:
                pass  # skip unparseable files

        # Batched add (100 per batch)
        seen_ids: set = set()
        batch_ids, batch_docs, batch_metas = [], [], []

        for chunk in self.all_chunks:
            cid = f"{chunk['file']}:{chunk['start_line']}:{chunk['name']}"
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            batch_ids.append(cid)
            batch_docs.append(chunk['code'])
            batch_metas.append({
                'file': chunk['file'],
                'name': chunk['name'],
                'type': chunk['type'],
                'start_line': str(chunk['start_line']),
                'end_line': str(chunk['end_line']),
            })

            if len(batch_ids) >= 100:
                self.collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                batch_ids, batch_docs, batch_metas = [], [], []

        if batch_ids:
            self.collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    # ------------------------------------------------------------- index docs
    def _index_docs(self):
        seen_ids: set = set()
        batch_ids, batch_docs, batch_metas = [], [], []

        for root, dirs, filenames in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() not in docs_take:
                    continue
                file_path = os.path.join(root, fname)
                try:
                    text = open(file_path, encoding='utf-8', errors='replace').read()
                except Exception:
                    continue

                rel = _relpath_posix(file_path, self.repo_path)
                sections = split_by_markdown_headers(text)
                for n, section in enumerate(sections):
                    if not section['content'].strip():
                        continue
                    doc_id = f"doc:{rel}:{n}"
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)

                    batch_ids.append(doc_id)
                    batch_docs.append(section['content'])
                    batch_metas.append({
                        'file': rel,
                        'heading': section['heading'],
                        'source_type': 'doc',
                    })

                    if len(batch_ids) >= 100:
                        self.collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                        batch_ids, batch_docs, batch_metas = [], [], []

        if batch_ids:
            self.collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    # ---------------------------------------------------------- index commits
    def _index_commits(self):
        commits = _get_commit_history(self.repo_path)
        batch_ids, batch_docs, batch_metas = [], [], []

        for commit in commits:
            text = f"{commit['message']} by {commit['author']} on {commit['date']}"
            cid = f"commit:{commit['sha']}"

            batch_ids.append(cid)
            batch_docs.append(text)
            batch_metas.append({
                'author': commit['author'],
                'date': commit['date'],
                'source_type': 'commit',
            })

            if len(batch_ids) >= 100:
                self.collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                batch_ids, batch_docs, batch_metas = [], [], []

        if batch_ids:
            self.collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    # ----------------------------------------------------------- build agent
    def _build_agent(self):
        llm = ChatGroq(model=self.groq_model, api_key=self.groq_api_key)
        tools = self._build_tools()

        system_prompt = (
            "You are a codebase assistant. Always cite the exact "
            "file path, line numbers, or commit hash your answer is based on. "
            "Never answer without grounding your claims in the retrieved information. "
            "Always use relative file paths (never absolute server paths)."
            "And whatever you are returning return in proper formating like dont use ** this type of things"
        )

        self.agent = create_react_agent(llm, tools, prompt=system_prompt)

    def _build_tools(self):
        repo_path = self.repo_path
        collection = self.collection
        all_chunks = self.all_chunks

        @tool
        def vector_search_tool(query: str) -> str:
            """Semantic search over code, docs, and commit messages."""
            results = collection.query(query_texts=[query], n_results=3)
            docs = results.get('documents', [[]])[0] if results.get('documents') else []
            metas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
            snippets = []
            for d, m in zip(docs[:3], metas[:3]):
                src = m.get('file') or m.get('source_type') or 'code'
                name = m.get('name') or ''
                snippet = d[:600]
                snippets.append(f"[{src} {name}]\n{snippet}")
            return "\n---\n".join(snippets) if snippets else "No relevant matches found."

        @tool
        def grep_tool(search_term: str) -> str:
            """Exact string/symbol search across the repo."""
            result = subprocess.run(
                ['git', '-C', repo_path, 'grep', '-n', '-F', '--', search_term],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
            )
            output = result.stdout or "No matches found."
            lines = output.splitlines()[:25]
            return "\n".join(lines)[:1500]

        @tool
        def git_log_tool(file_path: str) -> str:
            """Get commit history for a specific file."""
            result = subprocess.run(
                ['git', '-C', repo_path, 'log', '-5',
                 '--pretty=format:%h|%an|%ad|%s', '--date=short', '--', file_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
            )
            return (result.stdout or "No commits found.")[:1000]

        @tool
        def git_blame_tool(file_path: str, line_number: int) -> str:
            """Find who last changed a specific line and when."""
            result = subprocess.run(
                ['git', '-C', repo_path, 'blame', '-L',
                 f'{line_number},{line_number}', '--', file_path],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
            )
            return (result.stdout or "No blame info found.")[:500]

        @tool
        def find_definition_tool(name: str) -> str:
            """Find where a function or class is defined by exact name."""
            matches = [c for c in all_chunks if c['name'] == name][:2]
            if not matches:
                return f"No definition found for '{name}'."
            formatted = []
            for m in matches:
                formatted.append(f"File: {m['file']} (lines {m['start_line']}-{m['end_line']})\n{m['code'][:500]}")
            return "\n---\n".join(formatted)

        return [vector_search_tool, grep_tool, git_log_tool, git_blame_tool, find_definition_tool]

    # ------------------------------------------------------------------- ask
    def ask(self, question: str) -> str:
        """Ask the agent a question about the codebase with automatic rate limit backoff."""
        self.last_active = time.time()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.agent.invoke({
                    "messages": [{"role": "user", "content": question}]
                })
                return response["messages"][-1].content
            except Exception as e:
                err_msg = str(e).lower()
                if any(x in err_msg for x in ["413", "429", "rate limit", "itpm", "tokens per minute"]):
                    if attempt < max_retries - 1:
                        time.sleep(6)
                        continue
                raise

    # --------------------------------------------------------------- destroy
    def destroy(self):
        """Wipe temp directory and release ChromaDB resources."""
        try:
            if self.client and self.collection:
                self.client.delete_collection(f"codebase_{self.id}")
        except Exception:
            pass
        try:
            shutil.rmtree(self.workdir, ignore_errors=True)
        except Exception:
            pass
