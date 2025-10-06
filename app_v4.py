
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

# Arquivos de habilidades (um por categoria/subcategoria)
ARQ_TEC_OF  = BASE_DIR / "habilidades_tecnicas_ofensivas.txt"
ARQ_TEC_DEF = BASE_DIR / "habilidades_tecnicas_defensivas.txt"
ARQ_TAC_OF  = BASE_DIR / "habilidades_taticas_ofensivas.txt"
ARQ_TAC_DEF = BASE_DIR / "habilidades_taticas_defensivas.txt"

# Cores
COR_LUTA         = "#1f77b4"   # azul
COR_BRINCADEIRA  = "#2ca02c"   # verde
COR_TECNICA      = "#ff7f0e"   # laranja (of + def)
COR_TATICA       = "#9467bd"   # roxo   (of + def)

# -----------------------------------------------------------
# Utilidades
# -----------------------------------------------------------
def _ler_lista(arquivo: Path, fallback: List[str]) -> List[str]:
    if arquivo.exists():
        linhas = [l.strip() for l in arquivo.read_text(encoding="utf-8").splitlines()]
        return [l for l in linhas if l]
    return fallback

def carregar_habilidades_catalogo() -> Dict[str, List[str]]:
    """Retorna dict com 4 listas: tecnicas_of, tecnicas_def, taticas_of, taticas_def"""
    return {
        "tecnicas_of": _ler_lista(ARQ_TEC_OF,  ["projetar", "chutar", "golpear", "derrubar"]),
        "tecnicas_def": _ler_lista(ARQ_TEC_DEF, ["bloquear", "imobilizar", "defender", "segurar"]),
        "taticas_of": _ler_lista(ARQ_TAC_OF,  ["feintar", "atacar ângulo", "combinação", "pressão ofensiva"]),
        "taticas_def": _ler_lista(ARQ_TAC_DEF, ["marcação", "cobertura", "controle de distância", "contra-golpe"])
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
        st.session_state.view = "A"  # A: por tipo   |   B: clusters
    if "filters" not in st.session_state:
        st.session_state.filters = {"LB": True, "BH": True, "LH": True}

# -----------------------------------------------------------
# Modelo de dados dos registros:
# { "luta": str,
#   "brincadeira": str,
#   "hab_tecnicas_of": [str],
#   "hab_tecnicas_def": [str],
#   "hab_taticas_of": [str],
#   "hab_taticas_def": [str] }
# -----------------------------------------------------------
def adicionar_ou_atualizar(registros: List[dict], novo: dict) -> List[dict]:
    """Se já existe (luta, brincadeira), faz união das listas de habilidades; senão, adiciona."""
    luta = novo["luta"].strip().lower()
    brinc = novo["brincadeira"].strip().lower()

    for r in registros:
        if r["luta"].strip().lower() == luta and r["brincadeira"].strip().lower() == brinc:
            # unir habilidades por categoria
            for k in ["hab_tecnicas_of", "hab_tecnicas_def", "hab_taticas_of", "hab_taticas_def"]:
                r[k] = sorted(set(r[k]).union(set(novo[k])))
            return registros

    registros.append(novo)
    return registros

# -----------------------------------------------------------
# Construção da Rede (grafo)
# -----------------------------------------------------------
def build_graph_full(registros: List[dict]) -> nx.Graph:
    """
    Nós:
      - tipo="luta" (azul)
      - tipo="brincadeira" (verde)
      - tipo="habilidade" com sub_tipo in {"tecnica", "tatica"} (cores laranja/roxo)
        e sub_cat in {"ofensiva","defensiva"}

    Arestas, com atributo rel:
      - "LB" (luta-brincadeira)
      - "BH" (brincadeira-habilidade)
      - "LH" (luta-habilidade)
    """
    G = nx.Graph()

    for r in registros:
        luta = r["luta"].strip()
        brinc = r["brincadeira"].strip()

        # nós principais
        G.add_node(luta, tipo="luta", label=luta)
        G.add_node(brinc, tipo="brincadeira", label=brinc)

        # habilidades agrupadas
        grupos = [
            ("tecnica", "ofensiva",  r["hab_tecnicas_of"]),
            ("tecnica", "defensiva", r["hab_tecnicas_def"]),
            ("tatica",  "ofensiva",  r["hab_taticas_of"]),
            ("tatica",  "defensiva", r["hab_taticas_def"]),
        ]

        # Luta -> Brincadeira
        G.add_edge(luta, brinc, rel="LB")

        for sub_tipo, sub_cat, lista in grupos:
            for h in lista:
                nome = h.strip()
                if not nome:
                    continue
                G.add_node(nome, tipo="habilidade", sub_tipo=sub_tipo, sub_cat=sub_cat, label=nome)
                # Brincadeira -> Habilidade
                G.add_edge(brinc, nome, rel="BH")
                # Luta -> Habilidade
                G.add_edge(luta, nome, rel="LH")
    return G

def subgraph_for_relation(G_full: nx.Graph, show_LB: bool, show_BH: bool, show_LH: bool) -> nx.Graph:
    """
    Filtra arestas pela relação e REMOVE NÓS não incidentes nas arestas resultantes.
    """
    allowed = set()
    if show_LB: allowed.add("LB")
    if show_BH: allowed.add("BH")
    if show_LH: allowed.add("LH")

    H = nx.Graph()
    # Adiciona somente nós que estiverem em arestas permitidas
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
        # técnica (laranja) / tática (roxo)
        color = COR_TECNICA if attrs.get("sub_tipo") == "tecnica" else COR_TATICA
    else:
        color = "#888888"

    return {"color": color, "size": size, "title": f"{t} • grau: {deg}"}

def render_pyvis(G: nx.Graph, view: str = "A", partition=None, height="720px") -> str:
    net = Network(height=height, width="100%", notebook=False, directed=False, bgcolor="#ffffff")
    net.barnes_hut()

    # JSON puro (não JS) para o Pyvis
    net.set_options('''
    {
      "nodes": {
        "shape": "dot",
        "scaling": { "min": 10, "max": 55 },
        "font": { "size": 18 }
      },
      "edges": { "smooth": true },
      "physics": { "stabilization": true },
      "interaction": {
        "hover": true,
        "multiselect": true,
        "navigationButtons": true,
        "dragNodes": true
      }
    }
    ''')

    if view == "A":
        # Cores por tipo (com técnica/tática já mapeadas)
        for n in G.nodes():
            sty = color_and_size_for_node(G, n)
            net.add_node(str(n), label=str(G.nodes[n].get("label", n)),
                         color=sty["color"], size=sty["size"], title=sty["title"])
        for u, v, attrs in G.edges(data=True):
            net.add_edge(str(u), str(v))
    else:
        # Cores por cluster (mantendo tamanhos proporcionais)
        default_color = "#777777"
        palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
                   "#e377c2","#7f7f7f","#bcbd22","#17becf"]
        for n in G.nodes():
            deg = G.degree(n)
            size = max(10, 10 + 4 * deg)
            g = partition.get(n) if partition else 0
            color = palette[g % len(palette)] if partition else default_color
            net.add_node(str(n), label=str(G.nodes[n].get("label", n)),
                         color=color, size=size, title=f"grau: {deg}")
        for u, v, attrs in G.edges(data=True):
            net.add_edge(str(u), str(v))

    return net.generate_html(notebook=False)

def export_png(G: nx.Graph, filename="rede_v4.png", view="A", partition=None) -> str:
    plt.figure(figsize=(13, 9))
    pos = nx.spring_layout(G, seed=42, k=0.75)
    labels = {n: str(G.nodes[n].get("label", n)) for n in G.nodes()}

    if view == "A":
        colors = []
        sizes = []
        for n in G.nodes():
            sty = color_and_size_for_node(G, n)
            colors.append(sty["color"])
            sizes.append(300 + 60 * G.degree(n))
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        plt.title("Visualização A – Por tipo (Luta, Brincadeira, Técnica, Tática)")
        plt.figtext(0.01, 0.01, "Azul=Lutas | Verde=Brincadeiras | Laranja=Técnicas | Roxo=Táticas",
                    ha="left", va="bottom")
    else:
        if not partition:
            partition = {n: 0 for n in G.nodes()}
        palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
                   "#e377c2","#7f7f7f","#bcbd22","#17becf"]
        for g in sorted(set(partition.values())):
            nodes_g = [n for n in G.nodes() if partition[n] == g]
            sizes = [300 + 60 * G.degree(n) for n in nodes_g]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_g,
                                   node_color=palette[g % len(palette)], node_size=sizes,
                                   alpha=0.9, label=f"Cluster {g}")
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        plt.title("Visualização B – Clusters (Louvain)")
        plt.legend(scatterpoints=1, fontsize=9, frameon=False, loc="lower right")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return filename

# -----------------------------------------------------------
# Páginas
# -----------------------------------------------------------
def pagina_insercao():
    st.header("Área dos Alunos – Inserção")
    st.caption("Preencha os campos e adicione a brincadeira com suas habilidades.")

    cat = carregar_habilidades_catalogo()

    c1, c2 = st.columns([1.2, 1])
    with c1:
        luta = st.text_input("Nome da Luta", placeholder="Ex.: Judô")
        brinc = st.text_input("Nome da Brincadeira", placeholder="Ex.: Derruba-mato")

        tec_of  = st.multiselect("Técnicas Ofensivas",  options=cat["tecnicas_of"])
        tec_def = st.multiselect("Técnicas Defensivas", options=cat["tecnicas_def"])
        tac_of  = st.multiselect("Tática Ofensiva",     options=cat["taticas_of"])
        tac_def = st.multiselect("Tática Defensiva",    options=cat["taticas_def"])

        if st.button("➕ Adicionar brincadeira", type="primary"):
            if not luta.strip() or not brinc.strip():
                st.error("Informe a Luta **e** a Brincadeira.")
            elif len(tec_of + tec_def + tac_of + tac_def) == 0:
                st.error("Selecione ao menos uma habilidade/tática.")
            else:
                registros = carregar_dados()
                novo = {
                    "luta": luta.strip(),
                    "brincadeira": brinc.strip(),
                    "hab_tecnicas_of": sorted(set([x.strip() for x in tec_of])),
                    "hab_tecnicas_def": sorted(set([x.strip() for x in tec_def])),
                    "hab_taticas_of": sorted(set([x.strip() for x in tac_of])),
                    "hab_taticas_def": sorted(set([x.strip() for x in tac_def]))
                }
                registros = adicionar_ou_atualizar(registros, novo)
                salvar_dados(registros)
                st.success(f"Brincadeira **{brinc}** adicionada para a luta **{luta}**.")
                st.balloons()

    with c2:
        st.subheader("Registros atuais")
        registros = carregar_dados()
        if not registros:
            st.info("Nenhum registro ainda.")
        else:
            df = pd.DataFrame(registros)
            # mostra contagens agregadas também
            df2 = df.copy()
            for k in ["hab_tecnicas_of","hab_tecnicas_def","hab_taticas_of","hab_taticas_def"]:
                df2[k] = df2[k].apply(lambda xs: ", ".join(xs))
            st.dataframe(
                df2.rename(columns={
                    "luta":"Luta","brincadeira":"Brincadeira",
                    "hab_tecnicas_of":"Téc. Ofensivas",
                    "hab_tecnicas_def":"Téc. Defensivas",
                    "hab_taticas_of":"Tática Ofensiva",
                    "hab_taticas_def":"Tática Defensiva",
                }),
                use_container_width=True, hide_index=True
            )

    st.info("Dica: compartilhe este link com os alunos: `?page=insercao`")

def pagina_visualizacao():
    st.header("Área do Professor – Visualização e Controles")

    with st.sidebar:
        st.subheader("Filtros de Visualização")
        show_LB = st.checkbox("Luta → Brincadeira", value=st.session_state.filters["LB"])
        show_BH = st.checkbox("Brincadeira → Habilidade", value=st.session_state.filters["BH"])
        show_LH = st.checkbox("Luta → Habilidade", value=st.session_state.filters["LH"])
        st.session_state.filters.update({"LB": show_LB, "BH": show_BH, "LH": show_LH})

        st.divider()
        st.subheader("Legenda (Visual A)")
        st.markdown(
            "- 🔵 **Lutas**\n"
            "- 🟢 **Brincadeiras**\n"
            "- 🟠 **Técnicas** (Ofensivas e Defensivas)\n"
            "- 🟣 **Táticas** (Ofensivas e Defensivas)"
        )

        st.divider()
        if st.button("♻️ Limpar Rede", type="secondary"):
            salvar_dados([])
            st.toast("Rede limpa.", icon="✅")

    # Controle de visual
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("🟦 Visualização A – por tipo de nó", use_container_width=True):
            st.session_state.view = "A"
    with c2:
        if st.button("🌈 Visualização B – clusters (Louvain)", use_container_width=True):
            st.session_state.view = "B"
    with c3:
        export_clicked = st.button("💾 Exportar PNG", use_container_width=True)

    # Montar rede
    registros = carregar_dados()
    if not registros:
        st.warning("Sem registros. Peça que os alunos adicionem dados em `?page=insercao`.")
        return

    G_full = build_graph_full(registros)
    f = st.session_state.filters
    G = subgraph_for_relation(G_full, f["LB"], f["BH"], f["LH"])

    colA, colB = st.columns([3, 1])
    with colA:
        if G.number_of_nodes() == 0:
            st.warning("Nenhum nó para exibir com os filtros atuais.")
        else:
            if st.session_state.view == "A":
                html = render_pyvis(G, view="A")
                st.components.v1.html(html, height=720, scrolling=True)
            else:
                part = community_louvain.best_partition(G) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
                html = render_pyvis(G, view="B", partition=part)
                st.components.v1.html(html, height=720, scrolling=True)

            if export_clicked:
                if st.session_state.view == "A":
                    fn = export_png(G, "rede_v4_A.png", view="A")
                    with open(fn, "rb") as fbin:
                        st.download_button("Baixar PNG (Visual A)", data=fbin, file_name="rede_v4_A.png", mime="image/png")
                else:
                    part = community_louvain.best_partition(G) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
                    fn = export_png(G, "rede_v4_B.png", view="B", partition=part)
                    with open(fn, "rb") as fbin:
                        st.download_button("Baixar PNG (Visual B)", data=fbin, file_name="rede_v4_B.png", mime="image/png")

    with colB:
        st.subheader("Resumo")
        st.write(f"**Nós:** {G.number_of_nodes()} | **Arestas:** {G.number_of_edges()}")
        # contagem por tipo
        tipos = {"luta":0,"brincadeira":0,"habilidade_tecnica":0,"habilidade_tatica":0}
        for n, attrs in G.nodes(data=True):
            if attrs.get("tipo") == "luta":
                tipos["luta"] += 1
            elif attrs.get("tipo") == "brincadeira":
                tipos["brincadeira"] += 1
            elif attrs.get("tipo") == "habilidade":
                if attrs.get("sub_tipo") == "tecnica":
                    tipos["habilidade_tecnica"] += 1
                else:
                    tipos["habilidade_tatica"] += 1
        st.markdown(
            f"- Lutas: **{tipos['luta']}**  \n"
            f"- Brincadeiras: **{tipos['brincadeira']}**  \n"
            f"- Habilidades (Técnica): **{tipos['habilidade_tecnica']}**  \n"
            f"- Habilidades (Tática): **{tipos['habilidade_tatica']}**"
        )

# -----------------------------------------------------------
# Roteamento simples por query string
# -----------------------------------------------------------
init_state()
page = st.experimental_get_query_params().get("page", ["visualizacao"])[0].lower()

if page == "insercao":
    pagina_insercao()
else:
    pagina_visualizacao()
