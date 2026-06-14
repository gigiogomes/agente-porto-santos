import logging
import os
import sys
import uuid

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

try:
    from agent.coordinator_agent import CoordinatorAgent, default_filters
except ImportError:
    from coordinator_agent import CoordinatorAgent, default_filters

st.set_page_config(
    page_title="Assistente do Porto de Santos",
    page_icon="🚢",
    layout="centered",
)


# ============================================================== #
# Recursos pesados e IMUTÁVEIS, criados uma única vez e
# compartilhados entre sessões com segurança (base de dados,
# índice vetorial, clientes). NENHUMA memória de conversa vive
# aqui — ela fica em st.session_state, por usuário.
# ============================================================== #
@st.cache_resource
def get_global_coordinator() -> CoordinatorAgent:
    return CoordinatorAgent(session_id="recursos_compartilhados_porto")


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []          # histórico exibido na tela
    if "history" not in st.session_state:
        st.session_state.history = []           # memória de conversa do LLM
    if "data_filters" not in st.session_state:
        st.session_state.data_filters = default_filters()  # filtros do analista
    if "coordinator" not in st.session_state:
        with st.spinner(
            "Inicializando agentes e carregando base de dados (apenas uma vez)..."
        ):
            st.session_state.coordinator = get_global_coordinator()


def start_new_conversation() -> None:
    """Reset real da conversa: limpa a tela, a memória do LLM e os filtros — só desta sessão."""
    st.session_state.messages = []
    st.session_state.history = []
    st.session_state.data_filters = default_filters()


def reload_shared_resources() -> None:
    """Recarrega os recursos pesados compartilhados (afeta todas as sessões)."""
    get_global_coordinator.clear()
    st.session_state.coordinator = get_global_coordinator()


def render_sidebar() -> None:
    st.sidebar.header("Controles")

    if st.sidebar.button("🧹 Nova conversa"):
        start_new_conversation()
        st.rerun()

    if st.sidebar.button("🔄 Recarregar dados e agentes"):
        with st.spinner("Recarregando base de dados e agentes..."):
            reload_shared_resources()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Status do sistema")

    coordinator = st.session_state.coordinator
    try:
        st.sidebar.code(coordinator.data_agent.get_status_summary(), language="text")
    except Exception:
        logger.exception("Erro ao obter status dos dados.")
        st.sidebar.error("Não foi possível obter o status dos dados.")

    try:
        st.sidebar.write(
            f"**Knowledge agent:** {coordinator.knowledge_agent.status_message}"
        )
    except Exception:
        logger.exception("Erro ao obter status do knowledge agent.")
        st.sidebar.error("Não foi possível obter o status do knowledge agent.")


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_user_message(user_input: str) -> None:
    coordinator = st.session_state.coordinator

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            try:
                response_text, updated_history, updated_filters = coordinator.chat(
                    user_input,
                    history=st.session_state.history,
                    data_filters=st.session_state.data_filters,
                )
                # Persiste o estado atualizado da sessão.
                st.session_state.history = updated_history
                st.session_state.data_filters = updated_filters
            except Exception:
                logger.exception("Erro inesperado ao processar a mensagem do usuário.")
                response_text = (
                    "Ocorreu um erro inesperado ao processar a sua solicitação. "
                    "Tente novamente em instantes."
                )
        st.markdown(response_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )


def main() -> None:
    try:
        init_session_state()
    except Exception:
        logger.exception("Falha ao inicializar o sistema.")
        st.error(
            "Falha ao inicializar o sistema. Verifique as configurações e tente novamente."
        )
        st.stop()

    st.title("🚢 Assistente de IA - Porto de Santos")
    st.markdown(
        """
        Bem-vindo! Sou um agente especialista na movimentação de cargas do porto.
        Você pode me perguntar sobre **market share, crescimento, mix de cargas, volumes (TEUs)**
        e também sobre temas qualitativos, caso o knowledge agent esteja disponível.
        """
    )

    render_sidebar()
    render_chat_history()

    user_input = st.chat_input("Ex: Qual o market share por terminal em 2025?")
    if user_input:
        process_user_message(user_input)


if __name__ == "__main__":
    main()
