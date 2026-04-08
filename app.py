import glob
import os
import sys
from typing import Optional

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.coordinator_agent import CoordinatorAgent


TEMP_GRAPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_graphs")


st.set_page_config(
    page_title="Assistente do Porto de Santos",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Assistente de IA - Porto de Santos")
st.markdown("""
Bem-vindo! Sou um agente especialista na movimentação de cargas do porto.  
Você pode me perguntar sobre **Market Share, Crescimento, Mix de Cargas, Volumes (TEUs)** e também sobre temas qualitativos, caso o knowledge agent esteja carregado.
""")


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "coordinator" not in st.session_state:
        with st.spinner("Inicializando agentes e carregando base de dados..."):
            st.session_state.coordinator = CoordinatorAgent()


def rebuild_coordinator(clear_messages: bool = False):
    with st.spinner("Recarregando agentes e base de dados..."):
        st.session_state.coordinator = CoordinatorAgent()

    if clear_messages:
        st.session_state.messages = []


def list_chart_files():
    os.makedirs(TEMP_GRAPH_DIR, exist_ok=True)
    return set(glob.glob(os.path.join(TEMP_GRAPH_DIR, "*.png")))


def find_new_chart(before_files: set) -> Optional[str]:
    after_files = list_chart_files()
    new_files = list(after_files - before_files)

    if not new_files:
        return None

    new_files.sort(key=os.path.getmtime, reverse=True)
    return new_files[0]


def render_sidebar():
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
    except Exception as e:
        st.sidebar.error(f"Erro ao obter status dos dados: {e}")

    try:
        st.sidebar.write(f"**Knowledge agent:** {coordinator.knowledge_agent.status_message}")
    except Exception as e:
        st.sidebar.error(f"Erro ao obter status do knowledge agent: {e}")


def render_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("image"):
                st.image(message["image"])


def process_user_message(user_input: str):
    coordinator = st.session_state.coordinator

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("assistant"):
        before_files = list_chart_files()

        with st.spinner("Analisando..."):
            try:
                resposta_agente = coordinator.chat(user_input)
            except Exception as e:
                resposta_agente = (
                    "Ocorreu um erro ao processar a sua solicitação.\n\n"
                    f"Detalhe técnico: {str(e)}"
                )

        st.markdown(resposta_agente)

        image_bytes = None
        chart_path = find_new_chart(before_files)

        if chart_path and os.path.exists(chart_path):
            try:
                with open(chart_path, "rb") as f:
                    image_bytes = f.read()

                st.image(image_bytes)

                try:
                    os.remove(chart_path)
                except OSError:
                    pass

            except Exception as e:
                st.warning(f"Não foi possível exibir o gráfico gerado: {e}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": resposta_agente,
        "image": image_bytes
    })


def main():
    init_session_state()
    render_sidebar()
    render_chat_history()

    user_input = st.chat_input("Ex: Qual o market share por terminal em 2025?")
    if user_input:
        process_user_message(user_input)


if __name__ == "__main__":
    main()