import logging
import os
import sys
import uuid

import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from agent.port_agent import USER_FACING_ERROR, PortAgent, Turno  # noqa: E402
from tools.port_data_tools import load_port_data  # noqa: E402

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
def get_agent() -> PortAgent:
    return PortAgent()


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []          # histórico exibido na tela (texto + tabelas)
    if "history" not in st.session_state:
        st.session_state.history = []           # memória de conversa do LLM
    if "agent" not in st.session_state:
        with st.spinner("Inicializando o agente e carregando a base de dados..."):
            st.session_state.agent = get_agent()


def start_new_conversation() -> None:
    """Reset real da conversa: limpa a tela e a memória do LLM — só desta sessão."""
    st.session_state.messages = []
    st.session_state.history = []


def reload_shared_resources() -> None:
    """Recarrega dados e agente (afeta todas as sessões)."""
    load_port_data.clear()  # sem isto, o agente novo receberia os dados antigos do cache
    get_agent.clear()
    st.session_state.agent = get_agent()


def render_sidebar() -> None:
    st.sidebar.header("Controles")

    if st.sidebar.button("🧹 Nova conversa"):
        start_new_conversation()
        st.rerun()

    if st.sidebar.button("🔄 Recarregar dados e agente"):
        with st.spinner("Recarregando base de dados e agente..."):
            reload_shared_resources()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Status do sistema")

    agent = st.session_state.agent
    try:
        st.sidebar.code(agent.get_status_summary(), language="text")
    except Exception:
        logger.exception("Erro ao obter status dos dados.")
        st.sidebar.error("Não foi possível obter o status dos dados.")

    st.sidebar.write(f"**Busca em documentos:** {agent.knowledge_agent.status_message}")


def render_resultados(resultados) -> None:
    """Tabelas das consultas, formatadas pelo código, com a procedência de cada uma."""
    agent = st.session_state.agent
    for resultado in resultados:
        st.dataframe(resultado.tabela_exibicao(), hide_index=True, width="stretch")
        st.caption(agent.procedencia(resultado))


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_resultados(message.get("resultados", []))


def process_user_message(user_input: str) -> None:
    agent = st.session_state.agent

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        fim = {}

        def trechos_de_texto():
            for evento, valor in agent.responder(user_input, st.session_state.history):
                if evento == "texto":
                    yield valor
                else:
                    fim["turno"] = valor

        try:
            st.write_stream(trechos_de_texto())
            turno = fim["turno"]
            st.session_state.history = turno.historico
        except Exception:
            logger.exception("Erro inesperado ao processar a mensagem do usuário.")
            turno = Turno(USER_FACING_ERROR)
            st.markdown(turno.texto)
        render_resultados(turno.resultados)

    st.session_state.messages.append(
        {"role": "assistant", "content": turno.texto, "resultados": turno.resultados}
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
        Você pode me perguntar sobre **contêineres (TEUs), granéis sólidos e líquidos, carga geral
        e mercadorias (em toneladas)**: market share, crescimento, CAGR, mix de cargas, exportação
        e importação, por terminal e por período desde 2005.
        Também respondo sobre temas qualitativos, caso o knowledge agent esteja disponível.
        """
    )

    render_sidebar()
    render_chat_history()

    user_input = st.chat_input("Ex: Quantas toneladas de soja foram exportadas em 2025?")
    if user_input:
        process_user_message(user_input)


if __name__ == "__main__":
    main()
