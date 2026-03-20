import streamlit as st
import sys
import os

# Garante que o Python encontre as nossas pastas agent/ e tools/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.coordinator_agent import CoordinatorAgent

# Configuração da página
st.set_page_config(
    page_title="Assistente do Porto de Santos", 
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Assistente de IA - Porto de Santos")
st.markdown("""
Bem-vindo! Sou um agente especialista na movimentação de cargas do porto. 
Você pode me perguntar sobre **Market Share, Crescimento, Mix de Cargas e Volumes (TEUs)** dos terminais.
""")

# 1. Inicializa o Agente Coordenador apenas uma vez por sessão
if "coordinator" not in st.session_state:
    with st.spinner("Inicializando agentes e carregando banco de dados da APS..."):
        st.session_state.coordinator = CoordinatorAgent()

# 2. Inicializa o histórico de mensagens para a interface visual
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Exibe o histórico de mensagens na tela (agora suporta imagens!)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Se houver uma imagem salva nesta mensagem do histórico, exibe-a
        if message.get("image"):
            st.image(message["image"])

# 4. Caixa de texto para o usuário digitar
if user_input := st.chat_input("Ex: Qual o market share do porto em 2026?"):
    
    # Exibe a mensagem do usuário imediatamente
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Exibe a resposta do assistente com um indicador de carregamento
    with st.chat_message("assistant"):
        with st.spinner("Analisando dados ..."):
            # Chama o nosso fluxo multiagente
            resposta_agente = st.session_state.coordinator.chat(user_input)
            st.markdown(resposta_agente)
            
            # --- LÓGICA DE CAPTURA DO GRÁFICO ---
            image_bytes = None
            if os.path.exists("grafico_temp.png"):
                # Lê a imagem para a memória antes de a apagar
                with open("grafico_temp.png", "rb") as f:
                    image_bytes = f.read()
                
                # Exibe a imagem no chat atual
                st.image(image_bytes)
                
                # Apaga o ficheiro temporário para não poluir o sistema
                os.remove("grafico_temp.png")
            
    # Salva a resposta (e a imagem, se houver) no histórico da interface
    st.session_state.messages.append({
        "role": "assistant", 
        "content": resposta_agente,
        "image": image_bytes  # Guarda a imagem na memória da sessão
    })