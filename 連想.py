import os
import urllib.request
import zipfile
import numpy as np
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# ============================================================
# 1. 軽量 word2vec モデルを自動ダウンロード（初回のみ）
# ============================================================

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ja.vec"
MODEL_PATH = "wiki.ja.vec"
TOP_N = 10

@st.cache_resource(show_spinner="モデルを読み込んでいます（初回数秒）...")
def download_and_load_model():

    # モデルがなければダウンロード
    if not os.path.exists(MODEL_PATH):
        st.info("日本語軽量 word2vec モデルをダウンロード中...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    words = []
    vectors = []

    with open(MODEL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            if len(parts) < 300:
                continue
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
            except:
                continue
            words.append(word)
            vectors.append(vec)

    vectors = np.array(vectors, dtype=np.float32)
    word2idx = {w: i for i, w in enumerate(words)}
    return words, vectors, word2idx


# ============================================================
# 2. 類似語計算
# ============================================================

def most_similar(word, words, vectors, word2idx, topn=10):
    if word not in word2idx:
        return []

    idx = word2idx[word]
    vec = vectors[idx]

    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1e-10
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        vec_norm = 1e-10

    sims = (vectors @ vec) / (norms * vec_norm)
    sims[idx] = -1

    top_indices = np.argpartition(sims, -topn)[-topn:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    return [(words[i], float(sims[i])) for i in top_indices]


# ============================================================
# 3. PyVis ネットワーク生成
# ============================================================

def build_network_html(start_word, words, vectors, word2idx, branch_counts):
    G = nx.Graph()
    queue = [(start_word, 0)]
    visited = {start_word}

    G.add_node(start_word, size=45, color="#FF4500", title="起点", label=start_word)

    while queue:
        current_word, depth = queue.pop(0)
        if depth >= len(branch_counts):
            continue

        n_branch = branch_counts[depth]
        similar_words = most_similar(current_word, words, vectors, word2idx, topn=n_branch + 5)

        count = 0
        for word, score in similar_words:
            if count >= n_branch:
                break
            if word in visited or len(word) < 2:
                continue

            node_colors = ["#FFD700", "#87CEEB", "#32CD32"]
            node_sizes = [35, 25, 15]

            G.add_node(word,
                       size=node_sizes[depth],
                       color=node_colors[depth],
                       title=f"関連度: {score:.3f}",
                       label=word)
            G.add_edge(current_word, word, value=score)

            visited.add(word)
            queue.append((word, depth + 1))
            count += 1

    # PyVis に変換
    net = Network(height="750px", width="100%", bgcolor="#1a1a1a", font_color="white")
    net.from_nx(G)
    net.toggle_physics(True)

    html_path = "wordmap.html"
    net.write_html(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# 4. Streamlit UI
# ============================================================

st.set_page_config(page_title="連想ワードマップ", layout="wide")
st.title("🔍 連想ワードマップ（軽量・高速版）")

words, vectors, word2idx = download_and_load_model()

with st.sidebar:
    st.header("⚙️ 設定")
    start_word = st.text_input("起点ワード", value="半導体")

    st.markdown("---")
    b1 = st.slider("1階層目の表示数", 1, 10, 5)
    b2 = st.slider("2階層目の表示数", 1, 10, 3)
    b3 = st.slider("3階層目の表示数", 1, 5, 1)

    run = st.button("🚀 マップ生成", use_container_width=True)

if run:
    if start_word not in word2idx:
        st.error("単語が語彙にありません。別のワードで試してください。")
    else:
        with st.spinner("ネットワーク構築中..."):
            html = build_network_html(start_word, words, vectors, word2idx, [b1, b2, b3])
        st.success("完了！")
        components.html(html, height=770, scrolling=False)
else:
    st.info("左の起点ワードを入力して「マップ生成」を押してください。")
