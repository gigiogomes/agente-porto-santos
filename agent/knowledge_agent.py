import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """
    Busca semântica nos documentos de docs/ (RAG). Devolve os trechos mais
    relevantes para a pergunta; quem escreve a resposta é o agente principal,
    que recebe estes trechos como resultado da ferramenta `buscar_documentos`.
    """

    def __init__(self, docs_dir: str = "docs", persist_dir: str = "chroma_db"):
        logger.info("Iniciando busca em documentos (RAG)...")

        self.docs_dir = docs_dir
        self.persist_dir = persist_dir
        self.meta_path = os.path.join(self.persist_dir, "_index_meta.json")

        self.ready = False
        self.status_message = "Não inicializado"
        self.vector_store = None

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.status_message = "Indisponível: OPENAI_API_KEY não configurada."
            return

        try:
            from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma

            from langchain_openai import OpenAIEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception as exc:
            self.status_message = f"Indisponível: dependências do RAG ausentes ({exc})."
            return

        self._DirectoryLoader = DirectoryLoader
        self._TextLoader = TextLoader
        self._PyPDFLoader = PyPDFLoader
        self._Chroma = Chroma
        self._RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter

        try:
            self.embeddings = OpenAIEmbeddings(api_key=api_key)
        except Exception as exc:
            self.status_message = f"Falha ao inicializar embeddings: {exc}"
            return

        self.vector_store = self._init_or_load_db()
        if self.vector_store is None:
            if self.status_message == "Não inicializado":
                self.status_message = "Sem documentos válidos para indexação."
            return

        self.ready = True
        self.status_message = "Operacional"

    def _build_docs_fingerprint(self) -> str:
        docs_path = Path(self.docs_dir)
        if not docs_path.exists():
            return ""

        parts = []
        for file in sorted(docs_path.rglob("*")):
            if file.is_file():
                stat = file.stat()
                relative_path = str(file.relative_to(docs_path))
                parts.append(f"{relative_path}|{stat.st_size}|{int(stat.st_mtime)}")

        if not parts:
            return ""

        return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()

    def _read_saved_fingerprint(self) -> str:
        if not os.path.exists(self.meta_path):
            return ""
        try:
            with open(self.meta_path, "r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
            return data.get("fingerprint", "")
        except Exception:
            return ""

    def _save_fingerprint(self, fingerprint: str) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8") as file_obj:
            json.dump({"fingerprint": fingerprint}, file_obj, ensure_ascii=False, indent=2)

    def _init_or_load_db(self):
        if not os.path.exists(self.docs_dir):
            self.status_message = "Pasta de documentos não encontrada."
            return None

        fingerprint = self._build_docs_fingerprint()
        if not fingerprint:
            self.status_message = "Nenhum documento encontrado para indexação."
            return None

        should_rebuild = (
            (not os.path.exists(self.persist_dir))
            or (self._read_saved_fingerprint() != fingerprint)
        )

        if should_rebuild:
            logger.info("Reindexando documentos do knowledge agent...")
            try:
                # Loaders explícitos por extensão. O padrão do DirectoryLoader
                # (unstructured) é pesado e trava o processo no Windows.
                docs = []
                for pattern, loader_cls, loader_kwargs in [
                    ("**/*.txt", self._TextLoader, {"encoding": "utf-8"}),
                    ("**/*.md", self._TextLoader, {"encoding": "utf-8"}),
                    ("**/*.pdf", self._PyPDFLoader, {}),
                ]:
                    loader = self._DirectoryLoader(
                        self.docs_dir, glob=pattern, loader_cls=loader_cls, loader_kwargs=loader_kwargs
                    )
                    docs.extend(loader.load())
            except Exception as exc:
                self.status_message = f"Erro ao carregar documentos: {exc}"
                return None

            if not docs:
                self.status_message = "Nenhum documento válido carregado."
                return None

            splitter = self._RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(docs)

            try:
                vector_store = self._Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory=self.persist_dir,
                )
                self._save_fingerprint(fingerprint)
                return vector_store
            except Exception as exc:
                self.status_message = f"Erro ao criar índice vetorial: {exc}"
                return None

        logger.info("Carregando índice vetorial existente...")
        try:
            return self._Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
        except Exception as exc:
            self.status_message = f"Erro ao carregar índice vetorial existente: {exc}"
            return None

    def search(self, query: str, k: int = 4) -> str:
        """Trechos mais relevantes para a pergunta, identificados pelo arquivo de origem."""
        if not self.ready:
            return f"Busca em documentos indisponível. Status: {self.status_message}"

        try:
            docs = self.vector_store.similarity_search(query, k=k)
        except Exception:
            logger.exception("Erro na busca em documentos.")
            return "Erro ao buscar nos documentos."

        if not docs:
            return "Nenhum trecho relevante encontrado nos documentos."
        return "\n\n".join(
            f"[Trecho {i} - {Path(doc.metadata.get('source', '')).name}]\n{doc.page_content}"
            for i, doc in enumerate(docs, 1)
        )


if __name__ == "__main__":
    agent = KnowledgeAgent()
    question = "Quais são os planos de investimento e expansão para o Porto de Santos?"
    print(f"\nPergunta: {question}")
    print(f"\nTrechos:\n{agent.search(question)}")
