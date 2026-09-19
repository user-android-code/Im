import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline

# ページの設定
st.title("2D画像 3D奥行き変換アプリ")
st.write("画像をアップロードすると、AIが奥行き（デプスマップ）を推定します。")


# AIモデルの読み込み
@st.cache_resource
def load_model():
    return pipeline(
        task="depth-estimation",
        model="LiheYoung/depth-anything-small-hf",
    )


pipe = load_model()

# 画像のアップロードボタン
uploaded_file = st.file_uploader(
    "画像を選択してください...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("元の画像")
        st.image(image, use_container_width=True)

    with st.spinner("奥行きを計算中..."):
        depth_result = pipe(image)["depth"]

    with col2:
        st.subheader("生成されたデプスマップ")
        st.image(depth_result, use_container_width=True)

    st.success("計算完了！")
