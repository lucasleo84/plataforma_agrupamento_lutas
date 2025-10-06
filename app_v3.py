# Versão 3 do app Rede das Lutas, Brincadeiras e Habilidades (conteúdo omitido para ZIP)

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import community.community_louvain as community_louvain  # compatível com Streamlit Cloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

# ----------------------
# Config page
# ----------------------
st.set_page_config(
    page_title="Rede das Lutas, Brincadeiras e Habilidades",
    page_icon="🥋",
    layout="wide"
)

st.title("🥋 Rede das Lutas, Brincadeiras e Habilidades")
st.caption("Modele relações **Luta → Brincadeira → Habilidades** e explore diferentes camadas de conexão.")

DATA_DIR = Path(__file__).parent
HABILIDADES_ARQ = DATA_DIR / "habilidades.txt"

# ----------------------
# Helpers
# ----------------------
def carregar_habilidades():
    if HABILIDADES_ARQ.exists():
        linhas = [l.strip() for l in HABILIDADES_ARQ.read_text(encoding="utf-8").splitlines()]
        return [l for l in linhas if l]
    return ["projetar","empurrar","desequilibrar","imobilizar","derrubar",
            "golpear","esquivar","chutar","deslocar","defender"]

def init_state():
    if "registros" not in st.session_state:
        st.session_state.registros = []
    if "habilidades" not in st.session_state:
        st.session_state.habilidades = carregar_habilidades()
    if "view" not in st.session_state:
        st.session_state.view = "A"
    if "filters" not in st.session_state:
        st.session_state.filters = {"LB": True, "BH": True, "LH": True}

def add_or_update_registro(luta, brincadeira, habilidades):
    for r in st.session_state.registros:
        if r["luta"].lower().strip() == luta.lower().strip() and r["brincadeira"].lower().strip() == brincadeira.lower().strip():
            r["habilidades"] = sorted(set(r["habilidades"]).union(set(habilidades)))
            return
    st.session_state.registros.append({
        "luta": luta.strip(),
        "brincadeira": brincadeira.strip(),
        "habilidades": sorted(set(habilidades))
    })

def build_full_graph(registros, habilidades_validas):
    G = nx.Graph()
    for r in registros:
        luta = r["luta"]
        brinc = r["brincadeira"]
        habs = [h for h in r["habilidades"] if h in habilidades_validas]

        G.add_node(luta, tipo="luta", label=luta)
        G.add_node(brinc, tipo="brincadeira", label=brinc)
        for h in habs:
            G.add_node(h, tipo="habilidade", label=h)

        G.add_edge(luta, brinc, rel="LB")
        for h in habs:
            G.add_edge(brinc, h, rel="BH")
            G.add_edge(luta, h, rel="LH")
    return G

def subgraph_with_filters(G_full, show_LB, show_BH, show_LH):
    H = nx.Graph()
    for n, attrs in G_full.nodes(data=True):
        H.add_node(n, **attrs)
    allowed = set()
    if show_LB: allowed.add("LB")
    if show_BH: allowed.add("BH")
    if show_LH: allowed.add("LH")
    for u, v, attrs in G_full.edges(data=True):
        if attrs.get("rel") in allowed:
            H.add_edge(u, v, **attrs)
    return H

def render_pyvis(G, view="A", partition=None, height="700px"):
    net = Network(height=height, width="100%", notebook=False, directed=False, bgcolor="#ffffff")
    net.barnes_hut()
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
        for n, attrs in G.nodes(data=True):
            tipo = attrs.get("tipo","")
            deg = G.degree(n)
            size = max(10, 10 + 4*deg)
            color = "#1f77b4" if tipo=="luta" else "#2ca02c" if tipo=="brincadeira" else "#ff7f0e"
            net.add_node(str(n), label=str(n), color=color, size=size, title=f"{tipo} • grau: {deg}")
        for u,v,attrs in G.edges(data=True):
            net.add_edge(str(u), str(v))
    else:
        palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
        for n, attrs in G.nodes(data=True):
            deg = G.degree(n)
            size = max(10, 10 + 4*deg)
            g = partition.get(n) if partition else 0
            color = palette[g % len(palette)]
            net.add_node(str(n), label=str(n), color=color, size=size, title=f"grau: {deg}")
        for u,v,attrs in G.edges(data=True):
            net.add_edge(str(u), str(v))
    return net.generate_html(notebook=False)

def export_png(G, filename="rede_tripartida.png", view="A", partition=None):
    plt.figure(figsize=(13,9))
    pos = nx.spring_layout(G, seed=42, k=0.75)
    if view == "A":
        colors, sizes, labels = [], [], {n:str(n) for n in G.nodes()}
        for n,attrs in G.nodes(data=True):
            tipo = attrs.get("tipo","")
            color = {"luta":"#1f77b4","brincadeira":"#2ca02c","habilidade":"#ff7f0e"}.get(tipo,"#999999")
            colors.append(color)
            sizes.append(300+60*G.degree(n))
        nx.draw_networkx_nodes(G,pos,node_color=colors,node_size=sizes,alpha=0.9)
        nx.draw_networkx_edges(G,pos,alpha=0.3)
        nx.draw_networkx_labels(G,pos,labels=labels,font_size=10)
        plt.title("Visualização A – Por tipo de nó")
        plt.figtext(0.01,0.01,"Azul=Lutas | Verde=Brincadeiras | Laranja=Habilidades",ha="left",va="bottom")
    else:
        if not partition:
            partition={n:0 for n in G.nodes()}
        palette=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
        labels={n:str(n) for n in G.nodes()}
        for g in sorted(set(partition.values())):
            ns=[n for n in G.nodes() if partition[n]==g]
            sizes=[300+60*G.degree(n) for n in ns]
            nx.draw_networkx_nodes(G,pos,nodelist=ns,node_color=palette[g%len(palette)],node_size=sizes,alpha=0.9,label=f"Cluster {g}")
        nx.draw_networkx_edges(G,pos,alpha=0.3)
        nx.draw_networkx_labels(G,pos,labels=labels,font_size=10)
        plt.title("Visualização B – Clusters de Similaridade (Louvain)")
        plt.legend(scatterpoints=1,fontsize=9,frameon=False,loc="lower right")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename,dpi=300,bbox_inches="tight")
    plt.close()
    return filename

# ----------------------
# Init
# ----------------------
init_state()

# Sidebar (filters only)
with st.sidebar:
    st.header("⚙️ Filtros de Visualização (Saída)")
    show_LB = st.checkbox("Luta → Brincadeira", value=st.session_state.filters["LB"])
    show_BH = st.checkbox("Brincadeira → Habilidade", value=st.session_state.filters["BH"])
    show_LH = st.checkbox("Luta → Habilidade", value=st.session_state.filters["LH"])
    st.session_state.filters.update({"LB":show_LB,"BH":show_BH,"LH":show_LH})
    st.divider()
    st.subheader("Legenda (Visual A)")
    st.markdown("- 🔵 **Lutas**\n- 🟢 **Brincadeiras**\n- 🟠 **Habilidades**")
    st.divider()
    if st.button("♻️ Limpar rede", type="secondary"):
        st.session_state.registros=[]
        st.toast("Rede limpa.", icon="✅")

# Entrada de dados
st.subheader("Adicionar registro: Luta → Brincadeira → Habilidades")
c1,c2 = st.columns([1.2,1])
with c1:
    luta = st.text_input("Nome da Luta")
    brinc = st.text_input("Nome da Brincadeira")
    habs = st.multiselect("Habilidades", options=st.session_state.habilidades)
    if st.button("➕ Adicionar brincadeira", type="primary"):
        if not luta or not brinc or not habs:
            st.error("Preencha todos os campos.")
        else:
            add_or_update_registro(luta, brinc, habs)
            st.success(f"Brincadeira **{brinc}** adicionada à luta **{luta}**.")
            st.balloons()
with c2:
    if len(st.session_state.registros)==0:
        st.info("Nenhum registro adicionado.")
    else:
        df=pd.DataFrame(st.session_state.registros)
        df["habilidades"]=df["habilidades"].apply(lambda x:", ".join(x))
        st.dataframe(df.rename(columns={"luta":"Luta","brincadeira":"Brincadeira","habilidades":"Habilidades"}),use_container_width=True,hide_index=True)

st.divider()

b1,b2,b3=st.columns([1,1,1])
with b1:
    if st.button("🟦 Visualização A – por tipo de nó",use_container_width=True):
        st.session_state.view="A"
with b2:
    if st.button("🌈 Visualização B – por clusters (Louvain)",use_container_width=True):
        st.session_state.view="B"
with b3:
    export_clicked=st.button("💾 Exportar PNG",use_container_width=True)

if len(st.session_state.registros)==0:
    st.warning("Adicione registros para visualizar a rede.")
else:
    G_full=build_full_graph(st.session_state.registros,st.session_state.habilidades)
    f=st.session_state.filters
    G=subgraph_with_filters(G_full,f["LB"],f["BH"],f["LH"])
    if st.session_state.view=="A":
        html=render_pyvis(G,view="A")
        st.components.v1.html(html,height=720,scrolling=True)
        if export_clicked:
            fn=export_png(G,"rede_tripartida_A.png",view="A")
            with open(fn,"rb") as f: st.download_button("Baixar PNG (Visual A)",data=f,file_name="rede_tripartida_A.png",mime="image/png")
    else:
        part=community_louvain.best_partition(G) if G.number_of_edges()>0 else {n:0 for n in G.nodes()}
        html=render_pyvis(G,view="B",partition=part)
        st.components.v1.html(html,height=720,scrolling=True)
        if export_clicked:
            fn=export_png(G,"rede_tripartida_B.png",view="B",partition=part)
            with open(fn,"rb") as f: st.download_button("Baixar PNG (Visual B)",data=f,file_name="rede_tripartida_B.png",mime="image/png")

st.divider()
with st.expander("ℹ️ Dicas de uso"):
    st.markdown("""
1) Cadastre as **Brincadeiras** associadas a uma **Luta** e selecione as **Habilidades**.
2) Use os **filtros** para alternar entre camadas:  
   - Luta → Brincadeira (LB)  
   - Brincadeira → Habilidade (BH)  
   - Luta → Habilidade (LH)
3) Alterne entre **Visualização A** (por tipo) e **B** (por clusters).
4) Clique e arraste nós, exporte imagens e explore a rede em tempo real.
""")
