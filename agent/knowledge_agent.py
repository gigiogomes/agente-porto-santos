import os
import sys
from dotenv import load_dotenv

# Bibliotecas do LangChain para o RAG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Garante que as variáveis do .env (como OPENAI_API_KEY) sejam carregadas
load_dotenv()

class KnowledgeAgent:
    def __init__(self, docs_dir="docs", persist_dir="chroma_db"):
        print("Iniciando Agente de Conhecimento (RAG)...")
        self.docs_dir = docs_dir
        self.persist_dir = persist_dir
        
        # 1. Configura o modelo de Embeddings e o LLM
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # 2. Inicializa ou carrega o banco de dados vetorial
        self.vector_store = self._init_or_load_db()
        
        # 3. Configura o "Retriever" (buscador) para trazer os 3 trechos mais relevantes
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 4. Prompt de Sistema do RAG
        # 4. Prompt de Sistema do RAG
        system_prompt = (
            "Você é o Agente de Conhecimento do Porto de Santos. "
            "Sua missão é responder a perguntas qualitativas usando APENAS o contexto fornecido abaixo.\n"
            "O ano atual é 2026. Trate documentos e eventos de 2025 e 2026 como fatos presentes ou passados, não como previsões futuras.\n\n"
            "Se a resposta não estiver no contexto, diga claramente que não encontrou a informação nos documentos oficiais.\n\n"
            "Contexto recuperado:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # 5. Monta a corrente (Chain) que junta a busca com o LLM
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    def _init_or_load_db(self):
        """Cria o banco vetorial a partir dos PDFs/Textos ou carrega um existente."""
        
        # Se a pasta do banco já existe, apenas carrega (para ser rápido)
        if os.path.exists(self.persist_dir):
            print("✅ Carregando banco de dados vetorial existente...")
            return Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        
        # Se não existe, lê os documentos na pasta 'docs'
        print("⚙️ Lendo documentos e gerando embeddings pela primeira vez...")
        
        # O DirectoryLoader vai procurar PDFs, TXTs, etc.
        loader = DirectoryLoader(self.docs_dir, glob="**/*.*", show_progress=True)
        docs = loader.load()
        
        if not docs:
            print("⚠️ Nenhum documento encontrado na pasta 'docs'.")
            # Retorna um banco vazio
            return Chroma(embedding_function=self.embeddings, persist_directory=self.persist_dir)

        # Picota o texto em pedaços (chunks) de 1000 caracteres
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Converte para vetores e salva no disco
        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings, 
            persist_directory=self.persist_dir
        )
        print("✅ Banco de dados vetorial criado com sucesso!")
        return vector_store

    def ask(self, question: str) -> str:
        """Busca a resposta nos documentos."""
        response = self.rag_chain.invoke({"input": question})
        return response["answer"]

# ==========================================
# TESTE DO RAG ISOLADO
# ==========================================
if __name__ == "__main__":
    agente_rag = KnowledgeAgent()
    
    pergunta = "Quais são os planos de investimento e expansão para o porto de santos?"
    print(f"\nUsuário: {pergunta}")
    
    resposta = agente_rag.ask(pergunta)
    print(f"\nAgente RAG: {resposta}")