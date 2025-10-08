import json
import base64
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import community.community_louvain as community_louvain
from pathlib import Path
from typing import List, Dict

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
  .stButton>button{font-weight:600;border-radius:8px}
  .stSuccess{animation:flash .8s ease-in-out}
  @keyframes flash{0%{background:#e6ffe6}50%{background:#ccffcc}100%{background:#e6ffe6}}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
ARQ_DADOS   = BASE_DIR / "dados.json"
ARQ_TEC_OF  = BASE_DIR / "habilidades_tecnicas_ofensivas.txt"
ARQ_TEC_DEF = BASE_DIR / "habilidades_tecnicas_defensivas.txt"
ARQ_TAC_OF  = BASE_DIR / "habilidades_taticas_ofensivas.txt"
ARQ_TAC_DEF = BASE_DIR / "habilidades_taticas_defensivas.txt"

# Cores
COR_LUTA        = "#1f77b4"  # azul
COR_JOGO        = "#2ca02c"  # verde
COR_TEC_OF      = "#ff7f0e"  # laranja
COR_TEC_DEF     = "#d95f02"  # laranja escuro
COR_TAC_OF      = "#9467bd"  # roxo
COR_TAC_DEF     = "#6a51a3"  # roxo escuro

# Tamanhos fixos
SIZE_LUTA   = 35
SIZE_JOGO   = 25
SIZE_TEC    = 18
SIZE_TAC    = 16

# -----------------------------------------------------------
# UTILIDADES
# -----------------------------------------------------------
def _ler_lista(arquivo: Path, fallback: List[str]) -> List[str]:
    if arquivo.exists():
        linhas = [l.strip() for l in arquivo.read_text(encoding="utf-8").splitlines()]
        return [l for l in linhas if l]
    return fallback

def carregar_habilidades_catalogo() -> Dict[str, List[str]]:
    return {
        "tecnicas_of": _ler_lista(ARQ_TEC_OF,  ["projetar", "chutar", "golpear", "derrubar"]),
        "tecnicas_def": _ler_lista(ARQ_TEC_DEF, ["bloquear", "imobilizar", "defender", "segurar"]),
        "taticas_of": _ler_lista(ARQ_TAC_OF,  ["pressionar", "feintar", "avançar", "marcação alta"]),
        "taticas_def": _ler_lista(ARQ_TAC_DEF, ["cobertura", "contenção", "recuo", "posicionamento"])
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
    if "filters_rel" not in st.session_state:
        st.session_state.filters_rel = {"LB": True, "BH": True, "LH": True}

# -----------------------------------------------------------
# CONSTRUÇÃO DA REDE
# -----------------------------------------------------------
def build_graph_full(registros: List[dict]) -> nx.Graph:
    """
    Nós:
      - tipo="luta"
      - tipo="jogo"
      - tipo="habilidade" com sub_tipo em {"tecnica_of","tecnica_def","tatica_of","tatica_def"}
    Arestas:
      - rel="LB" (luta-jogo)
      - rel="BH" (jogo-habilidade)
      - rel="LH" (luta-habilidade)
    """
    G = nx.Graph()
    for r in registros:
        luta = (r.get("luta") or "").strip()
        jogo = (r.get("brincadeira") or "").strip()
        if not luta or not jogo:
            continue

        # Luta e Jogo
        G.add_node(luta, tipo="luta",   label=luta)
        G.add_node(jogo, tipo="jogo",   label=jogo)
        G.add_edge(luta, jogo, rel="LB")

        # Habilidades (todas opcionais)
        grupos = [
            ("tecnica_of", r.get("hab_tecnicas_of", [])),
            ("tecnica_def", r.get("hab_tecnicas_def", [])),
            ("tatica_of",  r.get("hab_taticas_of", [])),
            ("tatica_def", r.get("hab_taticas_def", [])),
        ]
        for sub_tipo, lista in grupos:
            for h in lista or []:
                nome = (h or "").strip()
                if not nome:
                    continue
                G.add_node(nome, tipo="habilidade", sub_tipo=sub_tipo, label=nome)
                G.add_edge(jogo, nome, rel="BH")
                G.add_edge(luta, nome, rel="LH")
    return G

def filter_edges_by_relation(G_full: nx.Graph, show_LB: bool, show_BH: bool, show_LH: bool) -> nx.Graph:
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

def filter_nodes_by_type(G: nx.Graph,
                         show_luta: bool, show_jogo: bool,
                         show_to: bool, show_td: bool,
                         show_ta_of: bool, show_ta_def: bool) -> nx.Graph:
    H = nx.Graph()
    for n, data in G.nodes(data=True):
        tipo = data.get("tipo", "")
        sub  = data.get("sub_tipo", "")
        keep = (
            (tipo == "luta" and show_luta) or
            (tipo == "jogo" and show_jogo) or
            (tipo == "habilidade" and sub == "tecnica_of" and show_to) or
            (tipo == "habilidade" and sub == "tecnica_def" and show_td) or
            (tipo == "habilidade" and sub == "tatica_of"  and show_ta_of) or
            (tipo == "habilidade" and sub == "tatica_def" and show_ta_def)
        )
        if keep:
            H.add_node(n, **data)
    for u, v, attrs in G.edges(data=True):
        if u in H and v in H:
            H.add_edge(u, v, **attrs)
    return H

def color_and_size_for_node(data: dict) -> Dict[str, str]:
    tipo = data.get("tipo", "")
    sub  = data.get("sub_tipo", "")
    if tipo == "luta":
        return {"color": COR_LUTA, "size": SIZE_LUTA}
    if tipo == "jogo":
        return {"color": COR_JOGO, "size": SIZE_JOGO}
    if tipo == "habilidade":
        if sub == "tecnica_of":
            return {"color": COR_TEC_OF, "size": SIZE_TEC}
        if sub == "tecnica_def":
            return {"color": COR_TEC_DEF, "size": SIZE_TEC}
        if sub == "tatica_of":
            return {"color": COR_TAC_OF, "size": SIZE_TAC}
        if sub == "tatica_def":
            return {"color": COR_TAC_DEF, "size": SIZE_TAC}
    return {"color": "#888888", "size": 16}

def render_pyvis(G: nx.Graph, partition=None, height="760px", highlight_cluster=None) -> str:
    net = Network(height=height, width="100%", notebook=False, directed=False, bgcolor="#ffffff")

    # Física estável e sem sobreposição; nós opacos por padrão
    net.set_options('''
    {
      "nodes": {
        "shape": "dot",
        "opacity": 0.85,
        "font": { "size": 18 }
      },
      "edges": {
        "smooth": false,
        "color": { "inherit": "from" },
        "width": 1.6
      },
      "physics": {
        "enabled": true,
        "solver": "barnesHut",
        "barnesHut": {
          "gravitationalConstant": -2800,
          "centralGravity": 0.08,
          "springLength": 240,
          "springConstant": 0.03,
          "avoidOverlap": 1.0
        }
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragNodes": true,
        "selectConnectedEdges": true,
        "hoverConnectedEdges": true,
        "multiselect": false,
        "navigationButtons": true
      },
      "manipulation": { "enabled": false },
      "configure": { "enabled": false }
    }
    ''')

    # Adiciona nós/arestas
    for n, data in G.nodes(data=True):
        style = color_and_size_for_node(data)
        label = str(data.get("label", n))
        # título = nome do nó (fundo amarelo será via CSS default do vis.js)
        net.add_node(str(n), label=label, title=label, color=style["color"], size=style["size"])

    for u, v, attrs in G.edges(data=True):
        net.add_edge(str(u), str(v))

    # JS: foco ao clicar (nós/arestas do ego destacados) + destaque de cluster
    cluster_json = "{}"
    if partition:
        # Mapa id -> cluster
        mapping = {str(n): int(partition.get(n, 0)) for n in G.nodes()}
        cluster_json = json.dumps(mapping)

    selected_cluster = "None" if highlight_cluster is None else str(highlight_cluster)

    custom_js = f"""
    <script type="text/javascript">
      const clusterMap = {cluster_json};
      const selectedCluster = "{selected_cluster}";

      function applyClusterHighlight() {{
        if (selectedCluster === "None") return;
        const target = parseInt(selectedCluster);
        const allNodes = network.body.nodes;
        for (const nodeId in allNodes) {{
          const cid = clusterMap[nodeId];
          if (cid === target) {{
            allNodes[nodeId].setOptions({{ color: {{ opacity: 1 }}, borderWidth: 3 }});
          }} else {{
            allNodes[nodeId].setOptions({{ color: {{ opacity: 0.15 }} }});
          }}
        }}
      }}

      network.on("afterDrawing", function() {{
        applyClusterHighlight();
      }});

      network.on("selectNode", function(params) {{
        var selected = params.nodes[0];
        var connected = network.getConnectedNodes(selected);
        var allNodes = network.body.nodes;
        var allEdges = network.body.edges;

        // nós
        for (var nodeId in allNodes) {{
          if (nodeId == selected || connected.includes(nodeId)) {{
            allNodes[nodeId].setOptions({{color:{{opacity:1}}, font:{{color:'black'}}, borderWidth: 3}});
          }} else {{
            allNodes[nodeId].setOptions({{color:{{opacity:0.2}}, font:{{color:'#cccccc'}}, borderWidth: 1}});
          }}
        }}
        // arestas - "brilho" nas conectadas diretamente ao nó selecionado
        for (var edgeId in allEdges) {{
          var e = allEdges[edgeId];
          if (e.fromId == selected || e.toId == selected) {{
            e.setOptions({{color:{{color:'#ffffff', opacity:1}}, width:3, shadow: true}});
          }} else {{
            e.setOptions({{color:{{opacity:0.1}}, width:1, shadow:false}});
          }}
        }}
      }});

      network.on("deselectNode", function(params) {{
        var allNodes = network.body.nodes;
        var allEdges = network.body.edges;
        for (var nodeId in allNodes) {{
          allNodes[nodeId].setOptions({{color:{{opacity:0.85}}, font:{{color:'black'}}, borderWidth: 1}});
        }}
        for (var edgeId in allEdges) {{
          allEdges[edgeId].setOptions({{color:{{opacity:1}}, width:1.6, shadow:false}});
        }}
        applyClusterHighlight(); // mantém destaque de cluster se estiver ativo
      }});
    </script>
    """

    html = net.generate_html(notebook=False)
    html = html.replace("</body>", custom_js + "\n</body>")
    return html

def download_html_button(html: str, filename="rede_lutas.html"):
    b64 = base64.b64encode(html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">💾 Baixar Rede (HTML)</a>'
    st.markdown(href, unsafe_allow_html=True)

# -----------------------------------------------------------
# PÁGINA DOS ALUNOS
# -----------------------------------------------------------
def pagina_insercao():
    st.header("Área dos Alunos – Inserção")
    st.caption("Preencha os campos e adicione o jogo com suas habilidades (tudo opcional, exceto Luta e Jogo).")
    cat = carregar_habilidades_catalogo()
    registros = carregar_dados()

    # Mensagem pós-inserção
    if "mensagem_sucesso" in st.session_state:
        st.success(st.session_state["mensagem_sucesso"])
        if st.session_state.get("mostrar_baloes"):
            st.balloons()
        del st.session_state["mensagem_sucesso"]
        del st.session_state["mostrar_baloes"]

    luta = st.text_input("Nome da Luta *")
    jogo = st.text_input("Nome do Jogo / Brincadeira *")

    col1, col2 = st.columns(2)
    with col1:
        tec_of  = st.multiselect("Técnicas Ofensivas (opcional)",  options=cat["tecnicas_of"])
        tac_of  = st.multiselect("Táticas Ofensivas (opcional)",    options=cat["taticas_of"])
    with col2:
        tec_def = st.multiselect("Técnicas Defensivas (opcional)", options=cat["tecnicas_def"])
        tac_def = st.multiselect("Táticas Defensivas (opcional)",   options=cat["taticas_def"])

    if st.button("➕ Adicionar Jogo", type="primary"):
        if not luta.strip() or not jogo.strip():
            st.error("Informe Luta e Jogo.")
        else:
            novo = {
                "luta": luta.strip(),
                "brincadeira": jogo.strip(),
                "hab_tecnicas_of": sorted(set(tec_of or [])),
                "hab_tecnicas_def": sorted(set(tec_def or [])),
                "hab_taticas_of": sorted(set(tac_of or [])),
                "hab_taticas_def": sorted(set(tac_def or [])),
            }
            registros.append(novo)
            salvar_dados(registros)
            st.session_state["mensagem_sucesso"] = f"✅ Jogo **{jogo}** adicionado à luta **{luta}** com sucesso!"
            st.session_state["mostrar_baloes"] = True
            st.rerun()

    st.subheader("📋 Registros atuais")
    registros = carregar_dados()
    if registros:
        df = pd.DataFrame(registros)
        for k in ["hab_tecnicas_of","hab_tecnicas_def","hab_taticas_of","hab_taticas_def"]:
            df[k] = df[k].apply(lambda xs: ", ".join(xs))
        st.dataframe(df.rename(columns={
            "luta":"Luta","brincadeira":"Jogo / Brincadeira",
            "hab_tecnicas_of":"Téc. Ofensivas","hab_tecnicas_def":"Téc. Defensivas",
            "hab_taticas_of":"Tát. Ofensivas","hab_taticas_def":"Tát. Defensivas"
        }), use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum registro ainda.")

# -----------------------------------------------------------
# PÁGINA DO PROFESSOR
# -----------------------------------------------------------
def pagina_visualizacao():
    st.header("Área do Professor – Visualização e Controles")

    with st.sidebar:
        # Filtros de arestas (relações)
        st.subheader("Filtros de Relação")
        fr = st.session_state.filters_rel
        fr["LB"] = st.checkbox("Luta ↔ Jogo", value=fr["LB"])
        fr["BH"] = st.checkbox("Jogo ↔ Habilidade", value=fr["BH"])
        fr["LH"] = st.checkbox("Luta ↔ Habilidade", value=fr["LH"])

        st.divider()
        # Filtros de tipos de nó
        st.subheader("Filtros de Tipos de Nó")
        show_luta  = st.checkbox("Luta", True)
        show_jogo  = st.checkbox("Jogo", True)
        show_to    = st.checkbox("Técnica Ofensiva", True)
        show_td    = st.checkbox("Técnica Defensiva", True)
        show_ta_of = st.checkbox("Tática Ofensiva", True)
        show_ta_def= st.checkbox("Tática Defensiva", True)

        st.divider()
        st.subheader("Clusters")
        st.caption("Selecione um cluster para destacá-lo (método Louvain).")
        # (lista será atualizada após construir o grafo)
        cluster_placeholder = st.empty()

        st.divider()
        st.subheader("Legenda (Visual A)")
        st.markdown(
            "- 🔵 **Lutas** (maiores)\n"
            "- 🟢 **Jogos** (médios)\n"
            "- 🟠 **Técnica Ofensiva** (menor)\n"
            "- 🟧 **Técnica Defensiva** (menor)\n"
            "- 🟣 **Tática Ofensiva** (menor)\n"
            "- 🟪 **Tática Defensiva** (menor)"
        )

        st.divider()
        if st.button("♻️ Limpar Rede", type="secondary"):
            salvar_dados([]); st.toast("Rede limpa.", icon="✅")

    registros = carregar_dados()
    if not registros:
        st.warning("Sem registros ainda. Peça aos alunos para adicionar em `?page=insercao`.")
        return

    # Controles de visualização
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("🟦 Visualização A (por tipo)", use_container_width=True):
            st.session_state.view = "A"
    with c2:
        if st.button("🌈 Visualização B (clusters)", use_container_width=True):
            st.session_state.view = "B"

    # Constrói grafo e aplica filtros
    G_full = build_graph_full(registros)
    G_rel  = filter_edges_by_relation(G_full, fr["LB"], fr["BH"], fr["LH"])
    G      = filter_nodes_by_type(G_rel, show_luta, show_jogo, show_to, show_td, show_ta_of, show_ta_def)

    if G.number_of_nodes() == 0:
        st.warning("Nenhum nó com os filtros atuais.")
        return

    # Clusters (para B e para destaque)
    partition = community_louvain.best_partition(G) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
    unique_clusters = sorted(set(partition.values())) if partition else []
    with cluster_placeholder:
        cluster_choice = st.selectbox(
            "Destacar Cluster",
            options=["Nenhum"] + [f"Cluster {c}" for c in unique_clusters],
            index=0
        )
    highlight_cluster = None if cluster_choice == "Nenhum" else int(cluster_choice.split()[-1])

    # Render
    if st.session_state.view == "A":
        html = render_pyvis(G, partition=None, highlight_cluster=highlight_cluster)
    else:
        html = render_pyvis(G, partition=partition, highlight_cluster=highlight_cluster)

    st.components.v1.html(html, height=760, scrolling=True)
    download_html_button(html, filename="rede_lutas.html")

# -----------------------------------------------------------
# ROTEAMENTO
# -----------------------------------------------------------
def main():
    init_state()
    params = st.query_params
    page = params.get("page", "visualizacao").lower()
    if page == "insercao":
        pagina_insercao()
    else:
        pagina_visualizacao()

if __name__ == "__main__":
    main()
