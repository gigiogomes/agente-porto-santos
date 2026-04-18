import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class KnowledgeAgent:
    def __init__(self, docs_dir: str = "docs", persist_dir: str = "chroma_db"):
        print("Iniciando Agente de Conhecimento (RAG)...")

        self.docs_dir = docs_dir
        self.persist_dir = persist_dir
        self.meta_path = os.path.join(self.persist_dir, "_index_meta.json")

        self.ready = False
        self.status_message = "Não inicializado"
        self.vector_store = None
        self.rag_chain = None

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.status_message = "Indisponível: OPENAI_API_KEY não configurada."
            return

        try:
            from langchain_classic.chains import create_retrieval_chain
            from langchain_classic.chains.combine_documents import create_stuff_documents_chain
            from langchain_community.document_loaders import DirectoryLoader
            
            # --- NOVA IMPORTAÇÃO DO CHROMA COM FALLBACK ---
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma
            # ----------------------------------------------
            
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception as exc:
            self.status_message = f"Indisponível: dependências do RAG ausentes ({exc})."
            return

        self._DirectoryLoader = DirectoryLoader
        self._Chroma = Chroma
        self._ChatPromptTemplate = ChatPromptTemplate
        self._RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
        self._create_retrieval_chain = create_retrieval_chain
        self._create_stuff_documents_chain = create_stuff_documents_chain

        try:
            self.embeddings = OpenAIEmbeddings(api_key=api_key)
            self.llm = ChatOpenAI(model=os.getenv("KNOWLEDGE_AGENT_MODEL", "gpt-4o"), temperature=0)
        except Exception as exc:
            self.status_message = f"Falha ao inicializar embeddings/LLM: {exc}"
            return

        self.vector_store = self._init_or_load_db()
        if self.vector_store is None:
            if self.status_message == "Não inicializado":
                self.status_message = "Sem documentos válidos para indexação."
            return

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        system_prompt = (
            "Você é o Agente de Conhecimento do Porto de Santos. "
            "Responda perguntas qualitativas usando apenas o contexto recuperado. "
            "Se a resposta não estiver nos documentos, diga isso claramente.\n\n"
            "Contexto recuperado:\n{context}"
        )

        prompt = self._ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )
        question_answer_chain = self._create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = self._create_retrieval_chain(retriever, question_answer_chain)

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
            print("⚙️ Reindexando documentos do knowledge agent...")
            try:
                loader = self._DirectoryLoader(self.docs_dir, glob="**/*.*", show_progress=True)
                docs = loader.load()
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

        print("✅ Carregando índice vetorial existente...")
        try:
            return self._Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
        except Exception as exc:
            self.status_message = f"Erro ao carregar índice vetorial existente: {exc}"
            return None

    def ask(self, question: str, context: str = "") -> str:
        if not self.ready:
            return f"Knowledge agent indisponível. Status: {self.status_message}"

        full_question = question
        if context:
            full_question = f"Contexto recente:\n{context}\n\nPergunta atual:\n{question}"

        try:
            response = self.rag_chain.invoke({"input": full_question})
            return response.get("answer", "Não foi possível gerar uma resposta.")
        except Exception as exc:
            return f"Erro ao consultar o knowledge agent: {exc}"


if __name__ == "__main__":
    agent = KnowledgeAgent()
    question = "Quais são os planos de investimento e expansão para o Porto de Santos?"
    print(f"\nUsuário: {question}")
    answer = agent.ask(question)
    print(f"\nAgente RAG: {answer}")
