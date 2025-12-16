import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

from milestone3 import run_graph_rag

st.set_page_config(page_title="Graph-RAG Travel Assistant", layout="wide")

st.title("Graph-RAG Travel Assistant (Milestone 3)")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")

llm_model = st.sidebar.selectbox(
    "Model Selection (LLM)",
    ["deepseek", "llama", "gemma"],
    index=0
)

retrieval_mode = st.sidebar.selectbox(
    "Retrieval Method",
    ["baseline", "embeddings", "both"],
    index=0
)

embeddings_override = ""

show_graph = st.sidebar.checkbox("Show graph visualization", value=True)

# -----------------------------
# Main input
# -----------------------------
user_text = st.text_input(
    "Ask a question",
    placeholder="e.g., Recommend 5-star hotels in Dubai, or do Egyptians need a visa to France?"
)

run_btn = st.button("Run")

if run_btn and user_text.strip():
    with st.spinner("Running Graph-RAG pipeline..."):
        out = run_graph_rag(
            user_text=user_text.strip(),
            llm_model=llm_model,
            retrieval_mode=retrieval_mode,
        )

    st.subheader("LLM answer")
    st.markdown(out["final_answer"])

    # -----------------------------
    # Top summary
    # -----------------------------
    st.subheader("Pipeline Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Intent", out["intent"])
    c2.metric("LLM Model", out["llm_model"])
    c3.metric("Retrieval Mode", out["retrieval_mode"])

    # -----------------------------
    # Required: show KG context
    # -----------------------------
    st.subheader("KG Retrieved Context (Raw)")

    # raw rows
    with st.expander("Raw Neo4j rows (as returned)", expanded=False):
        st.json(out["raw_rows"])

    # formatted text given to LLM
    st.subheader("KG Context (Formatted for LLM)")
    st.code(out["cypher_answer"] if out["cypher_answer"] else "(empty)", language="text")

    if retrieval_mode in ("embeddings", "both"):
        st.subheader("Embeddings Context (Model 1)")
        st.code(out.get("embeddings_model1") or "(empty)", language="text")

        st.subheader("Embeddings Context (Model 2)")
        st.code(out.get("embeddings_model2") or "(empty)", language="text")


    # -----------------------------
    # Optional: Cypher query executed
    # -----------------------------
    st.subheader("Cypher Query Executed")
    st.code(out["cypher_query"] if out["cypher_query"] else "(none)", language="cypher")
    st.write("**Params:**")
    st.json(out["cypher_params"])

   

    # -----------------------------
    # Optional: graph visualization
    # -----------------------------
    if show_graph:
        st.subheader("Graph Visualization (Snippet)")

        rows = out.get("raw_rows", [])
        if isinstance(rows, list) and rows:
            G = nx.Graph()

            def add_node_safe(label: str):
                if label and label not in G:
                    G.add_node(label)

            def add_edge_safe(a: str, b: str, rel: str):
                if a and b:
                    G.add_edge(a, b, rel=rel)

            # Try to infer node/edge structure
            for r in rows:
                if not isinstance(r, dict):
                    continue

                h = r.get("h") or {}
                city = r.get("city") or {}
                country = r.get("country") or {}
                to_c = r.get("to") or {}

                hname = h.get("name")
                cname = city.get("name")
                countryname = country.get("name")
                toname = to_c.get("name")

                # Hotels
                if hname:
                    add_node_safe(hname)
                    if cname:
                        add_node_safe(cname)
                        add_edge_safe(hname, cname, "IN_CITY")
                    if countryname:
                        add_node_safe(countryname)
                        add_edge_safe(cname or hname, countryname, "IN_COUNTRY")

                # Visa-free destinations (from -> to)
                # We may not have the "from" node in row, so label it from params if possible
                if toname:
                    add_node_safe(toname)

            if len(G.nodes) < 2:
                st.info("Not enough connected nodes in returned rows to draw a graph.")
            else:
                pos = nx.spring_layout(G, seed=7)

                edge_x, edge_y = [], []
                for u, v in G.edges():
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

                node_x, node_y, node_text = [], [], []
                for n in G.nodes():
                    x, y = pos[n]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(n)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none"))
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y, mode="markers+text",
                    text=node_text, textposition="top center",
                    hoverinfo="text"
                ))
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=520
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No raw rows returned. Nothing to visualize.")
