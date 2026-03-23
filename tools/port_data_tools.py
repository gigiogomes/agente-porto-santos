import pandas as pd
import requests
import io  # <--- Adicione esta linha!
from io import StringIO
import os
import matplotlib.pyplot as plt
import base64

# Configurações de URL e Caminho Local
URL_MENSARIO = "http://mensario.portodesantos.com.br/exportarcargas/csv"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CSV_PATH = "https://github.com/gigiogomes/agente-porto-santos/releases/download/v1.0/exportacao_cargas.csv"

# Mapeamento para que o agente entenda as siglas
TERMINAL_MAPPING = {
    "SBSA": "Santos Brasil",
    "DPW": "DP World",
    "BTP": "Brasil Terminal Portuário",
    "ECOPORTO": "Ecoporto Santos",
    "BTE": "BTE - Base de Transporte e Exportação"
}

def load_port_data() -> pd.DataFrame:
    """Carrega o CSV da URL da APS ou do backup local em caso de falha."""
    try:
        response = requests.get(URL_MENSARIO, timeout=10)
        response.raise_for_status() 
        # Trocamos sep=';' por sep=','
        df = pd.read_csv(StringIO(response.text), sep=',', encoding='utf-8')
        print("✅ Dados carregados da URL oficial da APS.")
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"⚠️ Erro ao acessar URL ({e}). Acionando fallback local...")
        try:
            # Trocamos sep=';' por sep=',' também no fallback
            df = pd.read_csv(LOCAL_CSV_PATH, sep=',', encoding='utf-8')
            print("✅ Dados carregados do arquivo local de backup.")
        except FileNotFoundError:
            raise Exception(f"Erro Crítico: Arquivo de backup não encontrado em {LOCAL_CSV_PATH}")
    
    print(f"🔍 [DEBUG] Colunas encontradas no CSV: {list(df.columns)}")
    
    # ==========================================
    # FILTRO DE REGRA DE NEGÓCIO: APENAS CONTÊINER
    # ==========================================
    if 'NATUREZA_CARGA' in df.columns:
        linhas_antes = len(df)
        df = df[df['NATUREZA_CARGA'] == "CARGA CONTEINERIZADA"].copy()
        linhas_depois = len(df)
        print(f"✅ Filtro de Carga Conteinerizada aplicado: de {linhas_antes} para {linhas_depois} linhas.")
    else:
        print("⚠️ AVISO: Coluna 'NATUREZA_CARGA' não encontrada. O filtro não pôde ser aplicado.")
    # ==========================================

    return _normalize_dataframe(df)

def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza os dados para evitar erros matemáticos do Agente."""
    
    # 1. Garante que o ANO e o MES sejam números
    if 'ANO' in df.columns:
        df['ANO'] = pd.to_numeric(df['ANO'], errors='coerce').fillna(0).astype(int)
    if 'MES' in df.columns:
        df['MES'] = pd.to_numeric(df['MES'], errors='coerce').fillna(0).astype(int)
        
    # 2. Limpa espaços invisíveis nos nomes dos terminais
    if 'TERMINAIS' in df.columns:
        df['TERMINAIS'] = df['TERMINAIS'].astype(str).str.strip().str.upper()
        
        # 🔍 DEBUG: Vai imprimir os primeiros 15 terminais que ele achar no CSV
        print(f"🔍 [DEBUG] Alguns terminais encontrados no CSV: {df['TERMINAIS'].unique()[:15]}")
        
        df['Nome_Terminal_Completo'] = df['TERMINAIS'].map(TERMINAL_MAPPING).fillna(df['TERMINAIS'])
    
    # 3. Garante que os valores de movimentação sejam floats validos
    for col in ['TOTAL_TEU', 'TOTAL_UNID', 'TOTAL_TONELADAS']:
        if col in df.columns:
            # Remove pontos de milhar, troca vírgula por ponto e converte
            df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df

# ==========================================
# FERRAMENTAS ANALÍTICAS (TOOLS PARA O AGENTE)
# ==========================================

def _encontrar_terminal(df: pd.DataFrame, termo_busca: str) -> pd.Series:
    """
    Função auxiliar inteligente: procura o terminal independentemente 
    de como está escrito no ficheiro original da APS.
    """
    # Tenta obter o nome por extenso do mapeamento (ex: SBSA -> Santos Brasil)
    nome_extenso = TERMINAL_MAPPING.get(termo_busca, termo_busca)
    
    # Procura na coluna TERMINAIS tanto a sigla quanto o nome extenso
    # case=False ignora maiúsculas/minúsculas.
    mask = df['TERMINAIS'].str.contains(termo_busca, case=False, na=False) | \
           df['TERMINAIS'].str.contains(nome_extenso, case=False, na=False)
           
    return mask

def get_volume(df: pd.DataFrame, terminal_sigla: str, ano: int, meses: list = None) -> str:
    """Calcula a movimentação total de um terminal em TEUs, Unidades e TEU Ratio."""
    
    mask = (df['ANO'] == ano)
    if meses and len(meses) > 0:
        mask &= (df['MES'].isin(meses))
        
    if terminal_sigla.upper() != "PORTO":
        mask &= _encontrar_terminal(df, terminal_sigla)
        
    df_filtrado = df[mask]

    if df_filtrado.empty:
        mes_str = f" nos meses {meses}" if meses else ""
        return f"Sem dados para {terminal_sigla} no ano de {ano}{mes_str}."

    total_teus = df_filtrado['TOTAL_TEU'].sum()
    
    # Coloque o nome EXATO da sua coluna de contêineres/unidades aqui:
    coluna_unidades = 'TOTAL_UNID' 
    
    if coluna_unidades in df_filtrado.columns:
        total_conteineres = df_filtrado[coluna_unidades].sum()
        teu_ratio = (total_teus / total_conteineres) if total_conteineres > 0 else 0
    else:
        total_conteineres = 0
        teu_ratio = 0
        return f"Erro: Coluna de unidades não encontrada. Total TEU: {total_teus:,.0f}"

    mes_str = f" (Meses: {meses})" if meses else ""
    return (
        f"Resultados de movimentação para {terminal_sigla} em {ano}{mes_str}:\n"
        f"- Total TEUs: {total_teus:,.0f}\n"
        f"- Total Contêineres (Unidades): {total_conteineres:,.0f}\n"
        f"- TEU Ratio: {teu_ratio:.2f} TEUs/Contêiner\n"
    )

def get_market_share(df: pd.DataFrame, ano: int, terminal_sigla: str, meses: list = None) -> str:
    """Calcula o market share (em TEUs) de um terminal, podendo filtrar por uma lista de meses."""
    df_filtrado = df[df['ANO'] == ano]
    
    if meses:
        df_filtrado = df_filtrado[df_filtrado['MES'].isin(meses)]
        periodo = f"nos meses {meses} de {ano}"
    else:
        periodo = f"no ano {ano}"
        
    total_porto = df_filtrado['TOTAL_TEU'].sum()
    mask_terminal = _encontrar_terminal(df_filtrado, terminal_sigla)
    total_terminal = df_filtrado[mask_terminal]['TOTAL_TEU'].sum()
    
    if total_porto == 0:
        return f"Não há dados de movimentação no porto para o período solicitado ({periodo})."
    
    market_share = (total_terminal / total_porto) * 100
    nome_terminal = TERMINAL_MAPPING.get(terminal_sigla, terminal_sigla)
    
    return f"O market share da {nome_terminal} {periodo} foi de {market_share:.2f}% (Volume do terminal: {total_terminal:,.0f} TEUs. Volume do porto: {total_porto:,.0f} TEUs)."

def get_cargo_mix(df: pd.DataFrame, terminal_sigla: str, ano: int, meses: list = None) -> str:
    """Retorna a quebra do tipo de carga de um terminal baseado nas regras de negócio (10 tipos)."""
    
    # 1. Filtra o ano e, opcionalmente, os meses solicitados
    mask = (df['ANO'] == ano)
    if meses and len(meses) > 0:
        mask &= (df['MES'].isin(meses))
        
    # Filtra o terminal (se não for o porto todo)
    if terminal_sigla.upper() != "PORTO":
        mask &= _encontrar_terminal(df, terminal_sigla)
        
    df_filtrado = df[mask].copy()

    if df_filtrado.empty:
        mes_str = f" nos meses {meses}" if meses else ""
        return f"Sem dados para {terminal_sigla} no ano de {ano}{mes_str}."

    # 2. Regra do Estado (Cheio ou Vazio)
    df_filtrado['ESTADO'] = 'Cheio'
    df_filtrado.loc[df_filtrado['MERCADORIAS'] == "SEM CARGAS", 'ESTADO'] = 'Vazio'

    # 3. Regra da Categoria (As 5 classificações com ordem de prioridade)
    df_filtrado['CATEGORIA'] = 'Outros' 
    df_filtrado.loc[df_filtrado['SENTIDO'] == "EMBARQUE", 'CATEGORIA'] = 'Exportação'
    df_filtrado.loc[df_filtrado['SENTIDO'] == "DESEMBARQUE", 'CATEGORIA'] = 'Importação'
    df_filtrado.loc[df_filtrado['TIPO_NAVEGACAO'] == "CABOTAGEM", 'CATEGORIA'] = 'Cabotagem' # Ajuste o nome da coluna aqui se o seu for diferente!
    df_filtrado.loc[df_filtrado['MOVIMENTO'] == "TRANSBORDO", 'CATEGORIA'] = 'Transbordo'
    df_filtrado.loc[df_filtrado['MOVIMENTO'] == "REMOÇÃO", 'CATEGORIA'] = 'Remoção'

    # 4. Junta Categoria + Estado
    df_filtrado['MIX_FINAL'] = df_filtrado['CATEGORIA'] + " " + df_filtrado['ESTADO']

    # 5. Agrupa, soma os TEUs e ordena
    resumo_mix = df_filtrado.groupby('MIX_FINAL')['TOTAL_TEU'].sum().sort_values(ascending=False)

    # 6. Monta o texto de resposta
    total_teus = resumo_mix.sum()
    mes_str = f" (Meses: {meses})" if meses else ""
    resultado = f"Mix de Cargas de {terminal_sigla} em {ano}{mes_str} - Total: {total_teus:,.0f} TEUs:\n"
    
    for categoria, volume in resumo_mix.items():
        percentual = (volume / total_teus) * 100 if total_teus > 0 else 0
        resultado += f"- {categoria}: {volume:,.0f} TEUs ({percentual:.2f}%)\n"

    return resultado

def get_growth(df: pd.DataFrame, terminal_sigla: str, ano_base: int, ano_comparacao: int, meses_base: list = None, meses_comparacao: list = None, cargo_mix: str = None) -> str:
    """Calcula a variação percentual em TEUs, com suporte opcional a filtro de mix de carga."""
    
    df_filtrado = df.copy()

    # Filtra o terminal (se não for o porto todo)
    if terminal_sigla.upper() != "PORTO":
        mask_terminal = _encontrar_terminal(df_filtrado, terminal_sigla)
        df_filtrado = df_filtrado[mask_terminal]

    # SE o usuário pedir um mix específico (ex: "Importação Cheio"), aplicamos a regra de classificação
    if cargo_mix:
        df_filtrado['ESTADO'] = 'Cheio'
        df_filtrado.loc[df_filtrado['MERCADORIAS'] == "SEM CARGAS", 'ESTADO'] = 'Vazio'

        df_filtrado['CATEGORIA'] = 'Outros' 
        df_filtrado.loc[df_filtrado['SENTIDO'] == "EMBARQUE", 'CATEGORIA'] = 'Exportação'
        df_filtrado.loc[df_filtrado['SENTIDO'] == "DESEMBARQUE", 'CATEGORIA'] = 'Importação'
        df_filtrado.loc[df_filtrado['TIPO_NAVEGACAO'] == "CABOTAGEM", 'CATEGORIA'] = 'Cabotagem' 
        df_filtrado.loc[df_filtrado['MOVIMENTO'] == "TRANSBORDO", 'CATEGORIA'] = 'Transbordo'
        df_filtrado.loc[df_filtrado['MOVIMENTO'] == "REMOÇÃO", 'CATEGORIA'] = 'Remoção'

        df_filtrado['MIX_FINAL'] = df_filtrado['CATEGORIA'] + " " + df_filtrado['ESTADO']
        
        # Filtra apenas as linhas que batem com o mix pedido (ignorando maiúsculas/minúsculas)
        df_filtrado = df_filtrado[df_filtrado['MIX_FINAL'].str.lower() == cargo_mix.lower()]

    # --- CÁLCULO DO VOLUME BASE ---
    mask_base = (df_filtrado['ANO'] == ano_base)
    if meses_base and len(meses_base) > 0:
        mask_base &= (df_filtrado['MES'].isin(meses_base))
    vol_base = df_filtrado[mask_base]['TOTAL_TEU'].sum()

    # --- CÁLCULO DO VOLUME COMPARAÇÃO ---
    mask_comp = (df_filtrado['ANO'] == ano_comparacao)
    if meses_comparacao and len(meses_comparacao) > 0:
        mask_comp &= (df_filtrado['MES'].isin(meses_comparacao))
    vol_comp = df_filtrado[mask_comp]['TOTAL_TEU'].sum()

    # --- CÁLCULO DO CRESCIMENTO ---
    if vol_base == 0:
        if vol_comp == 0:
            return f"Não houve movimentação em ambos os períodos para {terminal_sigla}."
        return f"Crescimento de 100% (volume base era zero, e agora é {vol_comp:,.0f} TEUs)."

    crescimento = ((vol_comp - vol_base) / vol_base) * 100
    
    # Textos formatados para a resposta
    mix_texto = f" na categoria '{cargo_mix}'" if cargo_mix else ""
    meses_base_str = f" nos meses {meses_base}" if meses_base else ""
    meses_comp_str = f" nos meses {meses_comparacao}" if meses_comparacao else ""
    
    resultado = (
        f"Análise de Crescimento para {terminal_sigla}{mix_texto}:\n"
        f"- Período Base ({ano_base}{meses_base_str}): {vol_base:,.0f} TEUs\n"
        f"- Período Atual ({ano_comparacao}{meses_comp_str}): {vol_comp:,.0f} TEUs\n"
        f"- Variação: {crescimento:.2f}%\n"
    )
    
    return resultado

def plotar_grafico(df: pd.DataFrame, tipo_grafico: str, tema: str = "market_share", anos: list = None, terminal_sigla: str = "PORTO") -> str:
    """Gera um gráfico visual e salva temporariamente no disco."""
    
    # Se o agente não souber o ano, usamos o ano mais recente (max) do DataFrame
    if anos is None or len(anos) == 0:
        anos = [int(df['ANO'].max())]
        
    # ... (o resto da função continua exatamente igual) ...
    
    # Prepara a figura (um pouco mais larga para caber 10 anos)
    plt.figure(figsize=(10, 5))
    
    # Cria o título lidando com um ano ou vários
    titulo_anos = f"{anos[0]} a {anos[-1]}" if len(anos) > 1 else str(anos[0])
    plt.title(f"{tema.replace('_', ' ').title()} - {terminal_sigla} ({titulo_anos})")
    
    # Filtra o DataFrame para pegar apenas a lista de anos solicitada
    df_filtrado = df[df['ANO'].isin(anos)].copy()

    # TEMA 1: Market Share (Proporção)
    if tema == "market_share":
        resumo = df_filtrado.groupby('TERMINAIS')['TOTAL_TEU'].sum().sort_values(ascending=False)
        top3 = resumo.head(3)
        outros = pd.Series({'OUTROS': resumo.iloc[3:].sum()}) if len(resumo) > 3 else pd.Series(dtype=float)
        dados_finais = pd.concat([top3, outros])

        if tipo_grafico in ['pizza', 'rosca']:
            plt.pie(dados_finais, labels=dados_finais.index, autopct='%1.1f%%', startangle=90)
            if tipo_grafico == 'rosca':
                centro = plt.Circle((0, 0), 0.70, fc='white')
                plt.gca().add_artist(centro)
        else:
            plt.bar(dados_finais.index, dados_finais.values, color='skyblue')

    # TEMA 2: Evolução Mensal (Tendência)
    elif tema == "evolucao_mensal":
        if terminal_sigla.upper() != "PORTO":
            mask = _encontrar_terminal(df_filtrado, terminal_sigla)
            df_filtrado = df_filtrado[mask]

        # Cria uma coluna 'ANO_MES' (ex: 2025-01) para a linha do tempo ficar contínua
        df_filtrado['ANO_MES'] = df_filtrado['ANO'].astype(str) + '-' + df_filtrado['MES'].astype(str).str.zfill(2)
        resumo_mensal = df_filtrado.groupby('ANO_MES')['TOTAL_TEU'].sum().sort_index()

        if tipo_grafico == 'linha':
            plt.plot(resumo_mensal.index, resumo_mensal.values, marker='o', linestyle='-', color='b')
            plt.grid(True, linestyle='--', alpha=0.6)
        else: 
            plt.bar(resumo_mensal.index, resumo_mensal.values, color='orange')
        
        # Se forem muitos meses (como 10 anos = 120 meses), rotacionamos os textos do eixo X
        plt.xticks(rotation=60 if len(resumo_mensal) > 12 else 0)
        plt.xlabel("Mês/Ano")
        plt.ylabel("TEUs")

    plt.tight_layout()

    caminho_arquivo = "grafico_temp.png"
    plt.savefig(caminho_arquivo, format="png", bbox_inches='tight')
    plt.close()

    return f"O gráfico foi gerado com sucesso e salvo em '{caminho_arquivo}'."
