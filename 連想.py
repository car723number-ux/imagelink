import os
import subprocess
import sys

# =================================================================
# 1. 必要なライブラリの自動インストール
# =================================================================
def manage_libraries():
    required = ["streamlit", "networkx", "pyvis", "sentence-transformers", "torch"]
    for lib in required:
        try:
            __import__(lib)
        except ImportError:
            print(f"ライブラリ '{lib}' をインストール中...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

manage_libraries()

import numpy as np
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from sentence_transformers import SentenceTransformer, util

# =================================================================
# 2. Sentence-BERT（MiniLM）モデルをロード
# =================================================================
@st.cache_resource(show_spinner="軽量日本語モデルを読み込んでいます…")
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

model = load_model()

# 保存しておく単語リスト（jawiki が無い代わりに語彙を自由に定義）
# → 必要ならここを外部ファイルにしてもOK
VOCAB = [
    "半導体", "AI", "自動車", "銀行", "経済", "株式", "金融", "エネルギー", "資源",
    "ネットワーク", "ソフトウェア", "製造業", "精密機器", "ロボット", "宇宙",
    "物流", "電子部品", "電気", "スマホ", "インフラ", "データセンター"
]

# 事前に全単語をベクトル化
@st.cache_resource(show_spinner="語彙のベクトル化中…")
def encode_vocab(vocab):
    vectors = model.encode(vocab, convert_to_tensor=True)
    return vocab, vectors

words, vectors = encode_vocab(VOCAB)


# =================================================================
# 3. 類似単語検索（Sentence-BERT版）
# =================================================================
def most_similar(word, words, vectors, topn=10):
    if word not in words:
        raise KeyError(f"'{word}' は語彙リストにありません。")

    idx = words.index(word)
    vec = vectors[idx]

    sims = util.cos_sim(vec, vectors)[0].cpu().numpy()
    sims[idx] = -1  # 自分自身を除外

    top_indices = sims.argsort()[::-1][:topn]

    return [(words[i], float(sims[i])) for i in top_indices]


# =================================================================
# 4. ネットワーク作成
# =================================================================
def build_network_html(start_word, words, vectors, branch_counts):
    G = nx.Graph()
    queue = [(start_word, 0)]
    G.add_node(start_word, size=45, title="起点", color="#FF4500", label=start_word)
    visited = {start_word}

    while queue:
        current_word, depth = queue.pop(0)
        if depth >= len(branch_counts):
            continue

        n_branch = branch_counts[depth]
        similar_words = most_similar(current_word, words, vectors, topn=n_branch + 5)

        count = 0
        for w, score in similar_words:
            if count >= n_branch:
                break
            if w in visited or len(w) < 1:
                continue

            node_colors = ["#FFD700", "#87CEEB", "#32CD32"]
            node_sizes = [35, 25, 15]

            G.add_node(w, size=node_sizes[depth], color=node_colors[depth],
                       title=f"関連度: {score:.3f}", label=w)
            G.add_edge(current_word, w, value=score)

            visited.add(w)
            queue.append((w, depth + 1))
            count += 1

    net = Network(height="750px", width="100%", bgcolor="#1a1a1a", font_color="white")
    net.from_nx(G)
    net.toggle_physics(True)

    html_path = "investment_map.html"
    net.write_html(html_path)
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()


# =================================================================
# 5. Streamlit UI
# =================================================================
st.set_page_config(page_title="連想マップ", layout="wide")
st.title("📈 連想ワードマップ（軽量版）")

with st.sidebar:
    st.header("⚙️ 設定")
    start_word = st.text_input("起点ワード", value="半導体")

    st.markdown("---")
    st.markdown("**階層ごとの表示数**")
    b1 = st.slider("1階層目", 1, 10, 5)
    b2 = st.slider("2階層目", 1, 10, 3)
    b3 = st.slider("3階層目", 1, 5, 1)

    run = st.button("🔍 マップを生成", use_container_width=True)

# 入力語が語彙リストにあるか判定
if run:
    if start_word not in words:
        st.error("この単語は語彙リストにありません。\nVOCAB に追加してください。")
    else:
        with st.spinner(f"「{start_word}」のネットワークを構築中..."):
            html = build_network_html(start_word, words, vectors, [b1, b2, b3])
        st.success("完成！")
        components.html(html, height=770, scrolling=False)
else:
    st.info("左のサイドバーに起点ワードを入力して「マップを生成」を押してください。")
