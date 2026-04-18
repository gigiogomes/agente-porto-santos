import os
import sys
import uuid
from typing import Optional

import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from agent.coordinator_agent import CoordinatorAgent
except ImportError:
    from coordinator_agent import CoordinatorAgent


st.set_page_config(
    page_title="Assistente do Porto de Santos",
    page_icon="🚢",
    layout="centered",
)

# ==========================================
# 1. A MÁGICA DO CACHE ACONTECE AQUI
# ==========================================
@st.cache_resource
def get_global_coordinator():
    # Usamos um ID fixo para o agente. Assim o servidor cria a base pesada UMA única vez
    # e a reaproveita, evitando estourar a memória a cada F5 (refresh) na página.
    return CoordinatorAgent(session_id="sessao_global_porto")


st.title("🚢 Assistente de IA - Porto de Santos")
st.markdown(
    """
Bem-vindo! Sou um agente especialista na movimentação de cargas do porto.
Você pode me perguntar sobre **market share, crescimento, mix de cargas, volumes (TEUs)**
e também sobre temas qualitativos, caso o knowledge agent esteja disponível.
"""
)

def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "coordinator" not in st.session_state:
        with st.spinner("Inicializando agentes e carregando base de dados (Isso ocorre apenas uma vez)..."):
            # 2. Chamamos o agente blindado pelo cache
            st.session_state.coordinator = get_global_coordinator()

def rebuild_coordinator(clear_messages: bool = False) -> None:
    with st.spinner("Recarregando agentes e base de dados..."):
        if not clear_messages:
            # Se for recarregar o banco de dados/agentes (botão 🔄)
            get_global_coordinator.clear() # Limpa a memória do servidor
            st.session_state.coordinator = get_global_coordinator()

    if clear_messages:
        # Se for apenas limpar o chat (botão 🧹)
        st.session_state.messages = []
        try:
            st.session_state.coordinator.reset_memory()
        except Exception:
            pass


def render_sidebar() -> None:
    st.sidebar.header("Controles")

    if st.sidebar.button("🔄 Recarregar dados e agentes"):
        rebuild_coordinator(clear_messages=False)
        st.rerun()

    if st.sidebar.button("🧹 Nova conversa"):
        rebuild_coordinator(clear_messages=True)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Status do sistema")

    coordinator = st.session_state.coordinator

    try:
        data_status = coordinator.data_agent.get_status_summary()
        st.sidebar.code(data_status, language="text")
    except Exception as exc:
        st.sidebar.error(f"Erro ao obter status dos dados: {exc}")

    try:
        st.sidebar.write(f"**Knowledge agent:** {coordinator.knowledge_agent.status_message}")
    except Exception as exc:
        st.sidebar.error(f"Erro ao obter status do knowledge agent: {exc}")


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("image"):
                st.image(message["image"])


def _read_generated_chart_bytes(chart_path: Optional[str]) -> Optional[bytes]:
    if not chart_path:
        return None

    if not os.path.exists(chart_path):
        return None

    try:
        with open(chart_path, "rb") as file_obj:
            image_bytes = file_obj.read()
        return image_bytes
    except Exception:
        return None
    finally:
        try:
            os.remove(chart_path)
        except OSError:
            pass


def process_user_message(user_input: str) -> None:
    coordinator = st.session_state.coordinator

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            try:
                coordinator.data_agent.clear_last_generated_chart()
                response_text = coordinator.chat(user_input)
                chart_path = coordinator.data_agent.last_generated_chart_path
            except Exception as exc:
                response_text = (
                    "Ocorreu um erro ao processar a sua solicitação.\n\n"
                    f"Detalhe técnico: {exc}"
                )
                chart_path = None

        st.markdown(response_text)

        image_bytes = _read_generated_chart_bytes(chart_path)
        if image_bytes:
            st.image(image_bytes)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "image": image_bytes,
        }
    )


def main() -> None:
    try:
        init_session_state()
    except Exception as exc:
        st.error(f"Falha ao inicializar o sistema: {exc}")
        st.stop()

    render_sidebar()
    render_chat_history()

    user_input = st.chat_input("Ex: Qual o market share por terminal em 2025?")
    if user_input:
        process_user_message(user_input)


if __name__ == "__main__":
    main()