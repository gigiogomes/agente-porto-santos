import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class KnowledgeAgent:
    def __init__(self, docs_dir="docs", persist_dir="chroma_db"):
        print("Iniciando Agente de Conhecimento (RAG)...")

        self.docs_dir = docs_dir
        self.persist_dir = persist_dir
        self.meta_path = os.path.join(self.persist_dir, "_index_meta.json")

        self.ready = False
        self.status_message = "Não inicializado"

        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

        self.vector_store = self._init_or_load_db()

        if self.vector_store is None:
            self.status_message = "Sem documentos válidos para indexação."
            return

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        system_prompt = (
            "Você é o Agente de Conhecimento do Porto de Santos. "
            "Responda perguntas qualitativas usando apenas o contexto recuperado. "
            "Se a resposta não estiver nos documentos, diga isso claramente.\n\n"
            "Contexto recuperado:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

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
                parts.append(f"{file.name}|{stat.st_size}|{int(stat.st_mtime)}")

        if not parts:
            return ""

        return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()

    def _read_saved_fingerprint(self) -> str:
        if not os.path.exists(self.meta_path):
            return ""

        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("fingerprint", "")
        except Exception:
            return ""

    def _save_fingerprint(self, fingerprint: str):
        os.makedirs(self.persist_dir, exist_ok=True)

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fingerprint}, f, ensure_ascii=False, indent=2)

    def _init_or_load_db(self):
        if not os.path.exists(self.docs_dir):
            print("⚠️ Pasta de documentos não encontrada.")
            return None

        fingerprint = self._build_docs_fingerprint()

        if not fingerprint:
            print("⚠️ Nenhum documento encontrado para indexação.")
            return None

        should_rebuild = (
            (not os.path.exists(self.persist_dir))
            or (self._read_saved_fingerprint() != fingerprint)
        )

        if should_rebuild:
            print("⚙️ Reindexando documentos do knowledge agent...")

            try:
                loader = DirectoryLoader(
                    self.docs_dir,
                    glob="**/*.*",
                    show_progress=True
                )
                docs = loader.load()
            except Exception as e:
                print(f"❌ Erro ao carregar documentos: {e}")
                return None

            if not docs:
                print("⚠️ Nenhum documento válido carregado.")
                return None

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = splitter.split_documents(docs)

            try:
                vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory=self.persist_dir
                )
                self._save_fingerprint(fingerprint)
                print("✅ Índice vetorial recriado com sucesso.")
                return vector_store
            except Exception as e:
                print(f"❌ Erro ao criar índice vetorial: {e}")
                return None

        print("✅ Carregando índice vetorial existente...")

        try:
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"❌ Erro ao carregar índice vetorial existente: {e}")
            return None

    def ask(self, question: str, context: str = "") -> str:
        if not self.ready:
            return f"Knowledge agent indisponível. Status: {self.status_message}"

        full_question = question
        if context:
            full_question = (
                f"Contexto recente:\n{context}\n\n"
                f"Pergunta atual:\n{question}"
            )

        try:
            response = self.rag_chain.invoke({"input": full_question})
            return response.get("answer", "Não foi possível gerar uma resposta.")
        except Exception as e:
            return f"Erro ao consultar o knowledge agent: {str(e)}"


if __name__ == "__main__":
    agente_rag = KnowledgeAgent()

    pergunta = "Quais são os planos de investimento e expansão para o Porto de Santos?"
    print(f"\nUsuário: {pergunta}")

    resposta = agente_rag.ask(pergunta)
    print(f"\nAgente RAG: {resposta}")