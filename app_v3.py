import json
import base64
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import community.community_louvain as community_louvain
from pathlib import Path

# -----------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# -----------------------------------------------------------
st.set_page_config(
    page_title="Rede das Lutas, Jogos e Habilidades",
    page_icon="🥋",
    layout="wide"
)
st.markdown("""
<style>
    .stButton>button {
        font-weight: 600;
        border-radius: 8px;
    }
    .stSuccess {
        animation: flash 0.8s ease-in-out;
    }
    @keyframes flash {
        0% { background-color: #e6ffe6; }
        50% { background-color: #ccffcc; }
        100% { background-color: #e6ffe6; }
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
ARQ_DADOS = BASE_DIR / "dados.json"
ARQ_TEC_OF  = BASE_DIR / "habilidades_tecnicas_ofensivas.txt"
ARQ_TEC_DEF = BASE_DIR / "habilidades_tecnicas_defensivas.txt"
ARQ_TAC     = BASE_DIR / "habilidades_taticas.txt"

# Cores fixas
COR_LUTA         = "#1f77b4"   # azul
COR_BRINCADEIRA  = "#2ca02c"   # verde
COR_TECNICA      = "#ff7f0e"   # laranja
COR_TATICA       = "#9467bd"   # roxo

# -----------------------------------------------------------
# FUNÇÕES DE SUPORTE
# -----------------------------------------------------------
def _ler_lista(arquivo: Path, fallback: list[str]) -> list[str]:
    if arquivo.exists():
        linhas = [l.strip() for l in arquivo.read_text(encoding="utf-8").splitlines()]
        return [l for l in linhas if l]
    return fallback

def carregar_habilidades_catalogo():
    return {
        "tecnicas_of": _ler_lista(ARQ_TEC_OF,  ["projetar", "chutar", "golpear"]),
        "tecnicas_def": _ler_lista(ARQ_TEC_DEF, ["bloquear", "imobilizar", "defender"]),
        "taticas": _ler_lista(ARQ_TAC, ["feintar", "cobrir", "pressionar"])
    }

def carregar_dados():
    if ARQ_DADOS.exists():
        try:
            return json.loads(ARQ_DADOS.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def salvar_dados(registros):
    ARQ_DADOS.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")

def init_state():
    if "view" not in st.session_state:
        st.session_state.view = "A"
    if "filters" not in st.session_state:
        st.session_state.filters = {"LB": True, "BH": True, "LH": True}

# -----------------------------------------------------------
# REDE
# -----------------------------------------------------------
def build_graph_full(registros):
    G = nx.Graph()
    for r in registros:
        luta = r["luta"].strip()
        brinc = r["brincadeira"].strip()

        G.add_node(luta, tipo="luta", label=luta)
        G.add_node(brinc, tipo="brincadeira", label=brinc)
        G.add_edge(luta, brinc, rel="LB")

        grupos = [
            ("tecnica", r.get("hab_tecnicas_of", [])),
            ("tecnica", r.get("hab_tecnicas_def", [])),
            ("tatica",  r.get("hab_taticas", [])),
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

def subgraph_for_relation(G_full, show_LB, show_BH, show_LH):
    allowed = set()
    if show_LB: allowed.add("LB")
    if show_BH: allowed.add("BH")
    if show_LH: allowed.add("LH")

    H = nx.Graph()
    for u, v, attrs in G_full.edges(data=True):
        if attrs.get("rel") in allowed:
            if u not in H: H.add_node(u, **G_full.nodes[u])
            if v not in H: H.add_node(v, **G_full.nodes[v])
            H.add_edge(u, v, **attrs)
    return H

def color_and_size_for_node(G, n):
    attrs = G.nodes[n]
    t = attrs.get("tipo", "")
    sub = attrs.get("sub_tipo", "")
    deg = G.degree(n)

    if t == "luta":
        size = 25
        color = COR_LUTA
    elif t == "brincadeira":
        size = 20
        color = COR_BRINCADEIRA
    elif t == "habilidade":
        size = max(14, 10 + 6 * deg)
        color = COR_TECNICA if sub == "tecnica" else COR_TATICA
    else:
        size = 15
        color = "#aaaaaa"

    return {"color": color, "size": size, "title": f"{t} • conexões: {deg}"}

# -----------------------------------------------------------
# VISUALIZAÇÃO PYVIS
# -----------------------------------------------------------
def render_pyvis(G, view="A", partition=None, height="740px"):
    net = Network(height=height, width="100%", notebook=False, directed=False, bgcolor="#ffffff")

    # Parâmetros visuais otimizados
    net.set_options('''
    {
      "nodes": {
        "shape": "dot",
        "opacity": 1.0,
        "borderWidth": 1,
        "scaling": { "min": 10, "max": 55 },
        "font": {
          "size": 10,
          "face": "Arial",
          "background": "rgba(255,255,255,0.8)",
          "strokeWidth": 0,
          "vadjust": 0
        }
      },
      "edges": {
        "smooth": false,
        "color": { "inherit": "from" },
        "width": 1.5
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragNodes": true,
        "selectConnectedEdges": true,
        "hoverConnectedEdges": true
      },
      "physics": {
        "enabled": true,
        "solver": "repulsion",
        "repulsion": {
          "centralGravity": 0.08,
          "springLength": 260,
          "springConstant": 0.03,
          "nodeDistance": 380,
          "damping": 0.1
        },
        "minVelocity": 0.75
      },
      "layout": {
        "improvedLayout": true,
        "hierarchical": false
      }
    }
    ''')

    # Nós e arestas
    if view == "A":
        for n in G.nodes():
            sty = color_and_size_for_node(G, n)
            net.add_node(str(n),
                         label=str(G.nodes[n].get("label", n)),
                         color={"background": sty["color"], "border": "#333333"},
                         size=sty["size"], title=sty["title"])
        for u, v, attrs in G.edges(data=True):
            net.add_edge(str(u), str(v))
    else:
        part = partition or {n: 0 for n in G.nodes()}
        palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                   "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
        for n in G.nodes():
            t = G.nodes[n].get("tipo")
            deg = G.degree(n)
            size = max(14, 10 + 6 * deg) if t == "habilidade" else (25 if t=="luta" else 20)
            color = palette[part.get(n, 0) % len(palette)]
            net.add_node(str(n),
                         label=str(G.nodes[n].get("label", n)),
                         color={"background": color, "border": "#333333"},
                         size=size, title=f"grau: {deg}")
        for u, v, attrs in G.edges(data=True):
            net.add_edge(str(u), str(v))

    # JavaScript customizado
    custom_js = """
    <script type="text/javascript">
      // Foco seletivo
      network.on("selectNode", function(params) {
        var selectedNode = params.nodes[0];
        var connectedNodes = network.getConnectedNodes(selectedNode);
        var allNodes = network.body.nodes;
        var allEdges = network.body.edges;

        for (var nodeId in allNodes) {
          if (nodeId == selectedNode || connectedNodes.includes(nodeId)) {
            allNodes[nodeId].setOptions({opacity:1, font:{color:'black'}});
          } else {
            allNodes[nodeId].setOptions({opacity:0.15, font:{color:'#bbbbbb'}});
          }
        }

        for (var edgeId in allEdges) {
          var edge = allEdges[edgeId];
          if (connectedNodes.includes(edge.fromId) || connectedNodes.includes(edge.toId)) {
            edge.setOptions({color:{opacity:1}});
          } else {
            edge.setOptions({color:{opacity:0.05}});
          }
        }
      });

      // Restaurar foco
      network.on("deselectNode", function(params) {
        var allNodes = network.body.nodes;
        var allEdges = network.body.edges;
        for (var nodeId in allNodes) {
          allNodes[nodeId].setOptions({opacity:1, font:{color:'black'}});
        }
        for (var edgeId in allEdges) {
          allEdges[edgeId].setOptions({color:{opacity:1}});
        }
      });

      // Mantém fonte legível mesmo com zoom
      network.on("zoom", function(params) {
        var scale = params.scale;
        var allNodes = network.body.nodes;
        for (var nodeId in allNodes) {
          allNodes[nodeId].setOptions({font:{size: Math.max(18, 24/scale)}});
        }
      });
    </script>
    """
    html = net.generate_html(notebook=False)
    html = html.replace("</body>", custom_js + "\n</body>")
    return html

# -----------------------------------------------------------
# DOWNLOAD HTML
# -----------------------------------------------------------
def download_html_button(html: str, filename="rede_lutas.html"):
    b64 = base64.b64encode(html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">💾 Baixar Rede (HTML)</a>'
    st.markdown(href, unsafe_allow_html=True)

# -----------------------------------------------------------
# PÁGINA DOS ALUNOS
# -----------------------------------------------------------
def pagina_insercao():
    st.header("Área dos Alunos – Inserção")
    st.caption("Preencha os campos e adicione o jogo com suas habilidades.")
    cat = carregar_habilidades_catalogo()
    registros = carregar_dados()

    if "recarregar" in st.session_state and st.session_state.recarregar:
        st.session_state.recarregar = False
        st.rerun()

    if "mensagem_sucesso" in st.session_state:
        st.success(st.session_state["mensagem_sucesso"])
        if st.session_state.get("mostrar_baloes"):
            st.balloons()
        del st.session_state["mensagem_sucesso"]
        del st.session_state["mostrar_baloes"]

    luta = st.text_input("Nome da Luta")
    brinc = st.text_input("Nome do Jogo / Brincadeira")
    tec_of  = st.multiselect("Técnicas Ofensivas",  options=cat["tecnicas_of"])
    tec_def = st.multiselect("Técnicas Defensivas", options=cat["tecnicas_def"])
    taticas = st.multiselect("Habilidades Táticas", options=cat["taticas"])

    if st.button("➕ Adicionar Jogo", type="primary"):
        if not luta.strip() or not brinc.strip():
            st.error("Informe Luta e Jogo.")
        elif len(tec_of + tec_def + taticas) == 0:
            st.error("Selecione pelo menos uma habilidade.")
        else:
            novo = {
                "luta": luta.strip(),
                "brincadeira": brinc.strip(),
                "hab_tecnicas_of": sorted(set(tec_of)),
                "hab_tecnicas_def": sorted(set(tec_def)),
                "hab_taticas": sorted(set(taticas))
            }
            registros.append(novo)
            salvar_dados(registros)
            st.session_state["mensagem_sucesso"] = f"✅ Jogo **{brinc}** adicionado à luta **{luta}** com sucesso!"
            st.session_state["mostrar_baloes"] = True
            st.session_state.recarregar = True
            st.rerun()

    st.subheader("📋 Registros atuais")
    registros = carregar_dados()
    if registros:
        df = pd.DataFrame(registros)
        for k in ["hab_tecnicas_of","hab_tecnicas_def","hab_taticas"]:
            df[k] = df[k].apply(lambda xs: ", ".join(xs))
        st.dataframe(df.rename(columns={
            "luta":"Luta","brincadeira":"Jogo / Brincadeira",
            "hab_tecnicas_of":"Téc. Ofensivas","hab_tecnicas_def":"Téc. Defensivas",
            "hab_taticas":"Habilidades Táticas"
        }), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum registro ainda.")

# -----------------------------------------------------------
# PÁGINA DO PROFESSOR
# -----------------------------------------------------------
def pagina_visualizacao():
    st.header("Área do Professor – Visualização e Controles")
    with st.sidebar:
        st.subheader("Filtros de Visualização")
        f = st.session_state.filters
        f["LB"] = st.checkbox("Luta → Jogo", value=f["LB"])
        f["BH"] = st.checkbox("Jogo → Habilidade", value=f["BH"])
        f["LH"] = st.checkbox("Luta → Habilidade", value=f["LH"])
        st.divider()
        st.subheader("Legenda (Visual A)")
        st.markdown("- 🔵 **Lutas**\n- 🟢 **Jogos**\n- 🟠 **Técnicas (Of/Def)**\n- 🟣 **Táticas**")
        st.divider()
        if st.button("♻️ Limpar Rede", type="secondary"):
            salvar_dados([]); st.toast("Rede limpa.", icon="✅")

    registros = carregar_dados()
    if not registros:
        st.warning("Sem registros ainda. Peça aos alunos para adicionar em `?page=insercao`.")
        return

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("🟦 Visualização A", use_container_width=True):
            st.session_state.view = "A"
    with c2:
        if st.button("🌈 Visualização B", use_container_width=True):
            st.session_state.view = "B"

    G_full = build_graph_full(registros)
    G = subgraph_for_relation(G_full, f["LB"], f["BH"], f["LH"])
    if G.number_of_nodes() == 0:
        st.warning("Nenhum nó com os filtros atuais.")
        return

    if st.session_state.view == "A":
        html = render_pyvis(G, view="A")
    else:
        part = community_louvain.best_partition(G) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
        html = render_pyvis(G, view="B", partition=part)

    st.components.v1.html(html, height=760, scrolling=True)
    download_html_button(html, filename="rede_lutas.html")

# -----------------------------------------------------------
# ROTEAMENTO
# -----------------------------------------------------------
init_state()
params = st.query_params
page = params.get("page", "visualizacao").lower()

if page == "insercao":
    pagina_insercao()
else:
    pagina_visualizacao()
