import json
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import List, Dict

matplotlib.use("Agg")

# -----------------------------------------------------------
# Configuração geral
# -----------------------------------------------------------
st.set_page_config(
    page_title="Rede das Lutas, Brincadeiras e Habilidades",
    page_icon="🥋",
    layout="wide"
)
st.markdown(
    "<style> .stButton>button { font-weight: 600 } .stDownloadButton>button { font-weight: 600 } </style>",
    unsafe_allow_html=True
)

BASE_DIR = Path(__file__).parent
ARQ_DADOS = BASE_DIR / "dados.json"

# Arquivos de habilidades (um por categoria)
ARQ_TEC_OF  = BASE_DIR / "habilidades_tecnicas_ofensivas.txt"
ARQ_TEC_DEF = BASE_DIR / "habilidades_tecnicas_defensivas.txt"
ARQ_TAC     = BASE_DIR / "habilidades_taticas.txt"

# Cores
COR_LUTA         = "#1f77b4"   # azul
COR_BRINCADEIRA  = "#2ca02c"   # verde
COR_TECNICA      = "#ff7f0e"   # laranja
COR_TATICA       = "#9467bd"   # roxo

# -----------------------------------------------------------
# Utilidades
# -----------------------------------------------------------
def _ler_lista(arquivo: Path, fallback: List[str]) -> List[str]:
    if arquivo.exists():
        linhas = [l.strip() for l in arquivo.read_text(encoding="utf-8").splitlines()]
        return [l for l in linhas if l]
    return fallback

def carregar_habilidades_catalogo() -> Dict[str, List[str]]:
    """Retorna dict com 3 listas: tecnicas_of, tecnicas_def, taticas"""
    return {
        "tecnicas_of": _ler_lista(ARQ_TEC_OF,  ["projetar", "chutar", "golpear", "derrubar"]),
        "tecnicas_def": _ler_lista(ARQ_TEC_DEF, ["bloquear", "imobilizar", "defender", "segurar"]),
        "taticas": _ler_lista(ARQ_TAC, ["feintar", "marcação", "cobertura", "pressão", "movimentação coletiva"])
    }

def carregar_dados() -> List[dict]:
    if ARQ_DADOS.exists():
        try:
            return json.loads(ARQ_DADOS.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def salvar_dados(registros: List[dict]) -> None:
    ARQ_DADOS.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")

def init_state():
    if "view" not in st.session_state:
        st.session_state.view = "A"
    if "filters" not in st.session_state:
        st.session_state.filters = {"LB": True, "BH": True, "LH": True}

# -----------------------------------------------------------
# Estrutura dos registros
# -----------------------------------------------------------
def adicionar_ou_atualizar(registros: List[dict], novo: dict) -> List[dict]:
    luta = novo["luta"].strip().lower()
    brinc = novo["brincadeira"].strip().lower()
    for r in registros:
        if r["luta"].strip().lower() == luta and r["brincadeira"].strip().lower() == brinc:
            for k in ["hab_tecnicas_of", "hab_tecnicas_def", "hab_taticas"]:
                r[k] = sorted(set(r[k]).union(set(novo[k])))
            return registros
    registros.append(novo)
    return registros

# -----------------------------------------------------------
# Construção da Rede
# -----------------------------------------------------------
def build_graph_full(registros: List[dict]) -> nx.Graph:
    G = nx.Graph()
    for r in registros:
        luta = r["luta"].strip()
        brinc = r["brincadeira"].strip()
        G.add_node(luta, tipo="luta", label=luta)
        G.add_node(brinc, tipo="brincadeira", label=brinc)
        G.add_edge(luta, brinc, rel="LB")

        grupos = [
            ("tecnica", r["hab_tecnicas_of"]),
            ("tecnica", r["hab_tecnicas_def"]),
            ("tatica",  r["hab_taticas"])
        ]
        for sub_tipo, lista in grupos:
            for h in lista:
                nome = h.strip()
                if not nome:
                    continue
                G.add_node(nome, tipo="habilidade", sub_tipo=sub_tipo, label=nome)
                G.add_edge(brinc, nome, rel="BH")
                G.add_edge(luta, nome, rel="LH")
    return G

def subgraph_for_relation(G_full: nx.Graph, show_LB: bool, show_BH: bool, show_LH: bool) -> nx.Graph:
    allowed = set()
    if show_LB: allowed.add("LB")
    if show_BH: allowed.add("BH")
    if show_LH: allowed.add("LH")
    H = nx.Graph()
    for u, v, attrs in G_full.edges(data=True):
        if attrs.get("rel") in allowed:
            if u not in H:
                H.add_node(u, **G_full.nodes[u])
            if v not in H:
                H.add_node(v, **G_full.nodes[v])
            H.add_edge(u, v, **attrs)
    return H

def color_and_size_for_node(G: nx.Graph, n: str) -> Dict[str, str]:
    attrs = G.nodes[n]
    t = attrs.get("tipo", "")
    deg = G.degree(n)
    size = max(10, 10 + 4 * deg)
    if t == "luta":
        color = COR_LUTA
    elif t == "brincadeira":
        color = COR_BRINCADEIRA
    elif t == "habilidade":
        color = COR_TECNICA if attrs.get("sub_tipo") == "tecnica" else COR_TATICA
    else:
        color = "#888888"
    return {"color": color, "size": size, "title": f"{t} • grau: {deg}"}

def render_pyvis(G: nx.Graph, view="A", partition=None, height="720px") -> str:
    net = Network(height=height, width="100%", notebook=False, directed=False, bgcolor="#ffffff")
    net.barnes_hut()
    net.set_options('''
    {
      "nodes": {"shape": "dot", "scaling": {"min": 10, "max": 55}, "font": {"size": 18}},
      "edges": {"smooth": true},
      "physics": {"stabilization": true},
      "interaction": {"hover": true,"navigationButtons": true,"dragNodes": true}
    }
    ''')
    if view == "A":
        for n in G.nodes():
            sty = color_and_size_for_node(G, n)
            net.add_node(str(n), label=str(G.nodes[n].get("label", n)),
                         color=sty["color"], size=sty["size"], title=sty["title"])
        for u, v, attrs in G.edges(data=True):
            net.add_edge(str(u), str(v))
    else:
        palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
        for n in G.nodes():
            deg = G.degree(n)
            size = max(10, 10 + 4 * deg)
            g = partition.get(n) if partition else 0
            color = palette[g % len(palette)]
            net.add_node(str(n), label=str(G.nodes[n].get("label", n)),
                         color=color, size=size, title=f"grau: {deg}")
        for u, v, attrs in G.edges(data=True):
            net.add_edge(str(u), str(v))
    return net.generate_html(notebook=False)

# -----------------------------------------------------------
# Interface - Páginas
# -----------------------------------------------------------
def pagina_insercao():
    st.header("Área dos Alunos – Inserção")
    st.caption("Preencha os campos e adicione a brincadeira com suas habilidades.")
    cat = carregar_habilidades_catalogo()
    luta = st.text_input("Nome da Luta")
    brinc = st.text_input("Nome da Brincadeira")
    tec_of  = st.multiselect("Técnicas Ofensivas",  options=cat["tecnicas_of"])
    tec_def = st.multiselect("Técnicas Defensivas", options=cat["tecnicas_def"])
    taticas = st.multiselect("Habilidades Táticas", options=cat["taticas"])
    if st.button("➕ Adicionar Brincadeira", type="primary"):
        if not luta.strip() or not brinc.strip():
            st.error("Informe Luta e Brincadeira.")
        elif len(tec_of + tec_def + taticas) == 0:
            st.error("Selecione pelo menos uma habilidade.")
        else:
            registros = carregar_dados()
            novo = {"luta": luta.strip(),"brincadeira": brinc.strip(),
                    "hab_tecnicas_of": tec_of,"hab_tecnicas_def": tec_def,"hab_taticas": taticas}
            registros = adicionar_ou_atualizar(registros, novo)
            salvar_dados(registros)
            st.success(f"Brincadeira **{brinc}** adicionada à luta **{luta}**.")
            st.balloons()

    registros = carregar_dados()
    if registros:
        df = pd.DataFrame(registros)
        for k in ["hab_tecnicas_of","hab_tecnicas_def","hab_taticas"]:
            df[k]=df[k].apply(lambda x:", ".join(x))
        st.dataframe(df.rename(columns={"luta":"Luta","brincadeira":"Brincadeira",
                                        "hab_tecnicas_of":"Téc. Ofensivas","hab_tecnicas_def":"Téc. Defensivas",
                                        "hab_taticas":"Habilidades Táticas"}),
                     use_container_width=True,hide_index=True)
   
    st.subheader("✏️ Corrigir ou Atualizar Brincadeira")
    registros = carregar_dados()
    
    if registros:
        opcoes = [f"{r['luta']} – {r['brincadeira']}" for r in registros]
        escolha = st.selectbox("Selecione o registro para editar", opcoes)
    
        if escolha:
            idx = opcoes.index(escolha)
            r = registros[idx]
    
            nova_luta = st.text_input("Luta", r["luta"])
            nova_brinc = st.text_input("Brincadeira", r["brincadeira"])
            tec_of  = st.multiselect("Técnicas Ofensivas", cat["tecnicas_of"], default=r["hab_tecnicas_of"])
            tec_def = st.multiselect("Técnicas Defensivas", cat["tecnicas_def"], default=r["hab_tecnicas_def"])
            taticas = st.multiselect("Habilidades Táticas", cat["taticas"], default=r["hab_taticas"])
    
            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Atualizar Registro"):
                    registros[idx] = {
                        "luta": nova_luta.strip(),
                        "brincadeira": nova_brinc.strip(),
                        "hab_tecnicas_of": tec_of,
                        "hab_tecnicas_def": tec_def,
                        "hab_taticas": taticas
                    }
                    salvar_dados(registros)
                    st.success("Registro atualizado com sucesso.")
            with c2:
                if st.button("🗑️ Excluir Registro"):
                    registros.pop(idx)
                    salvar_dados(registros)
                    st.warning("Registro excluído.")
else:
    st.info("Nenhum registro disponível para edição.")

def pagina_visualizacao():
    st.header("Área do Professor – Visualização e Controles")
    with st.sidebar:
        st.subheader("Filtros de Visualização")
        show_LB = st.checkbox("Luta → Brincadeira", value=st.session_state.filters["LB"])
        show_BH = st.checkbox("Brincadeira → Habilidade", value=st.session_state.filters["BH"])
        show_LH = st.checkbox("Luta → Habilidade", value=st.session_state.filters["LH"])
        st.session_state.filters.update({"LB": show_LB, "BH": show_BH, "LH": show_LH})
        st.divider(); st.subheader("Legenda (Visual A)")
        st.markdown("- 🔵 **Lutas**\n- 🟢 **Brincadeiras**\n- 🟠 **Técnicas** (Of/Def)\n- 🟣 **Táticas**")
        st.divider()
        if st.button("♻️ Limpar Rede", type="secondary"):
            salvar_dados([]); st.toast("Rede limpa.", icon="✅")

    registros = carregar_dados()
    if not registros:
        st.warning("Sem registros ainda. Peça aos alunos para adicionar em `?page=insercao`.")
        return

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("🟦 Visualização A", use_container_width=True): st.session_state.view="A"
    with c2:
        if st.button("🌈 Visualização B", use_container_width=True): st.session_state.view="B"
    with c3:
        export_clicked=st.button("💾 Exportar PNG",use_container_width=True)

    G_full=build_graph_full(registros)
    f=st.session_state.filters
    G=subgraph_for_relation(G_full,f["LB"],f["BH"],f["LH"])
    if G.number_of_nodes()==0:
        st.warning("Nenhum nó com os filtros atuais."); return

    if st.session_state.view=="A":
        html=render_pyvis(G,"A")
        st.components.v1.html(html,height=720,scrolling=True)
    else:
        part=community_louvain.best_partition(G) if G.number_of_edges()>0 else {n:0 for n in G.nodes()}
        html=render_pyvis(G,"B",part)
        st.components.v1.html(html,height=720,scrolling=True)

# -----------------------------------------------------------
# Roteamento simples por query string (corrigido)
# -----------------------------------------------------------
init_state()
params = st.query_params
page = params.get("page", "visualizacao").lower()

if page == "insercao":
    pagina_insercao()
else:
    pagina_visualizacao()
