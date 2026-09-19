import numpy as np
import streamlit as st
import timm
import torch
import torch.nn.functional as F
from PIL import Image

st.title("2D画像 3D奥行き変換アプリ")
st.write("画像をアップロードすると、AIが奥行き（デプスマップ）を推定します。")


@st.cache_resource
def load_model():
    # 超軽量なMiDaSモデルを読み込み
    model = timm.create_model("midas_small", pretrained=True)
    model.eval()
    return model


model = load_model()

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
        # 画像の前処理
        img = np.array(image)
        img_input = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        img_input = F.interpolate(
            img_input, size=(256, 256), mode="bilinear", align_corners=False
        )
        img_input = img_input / 255.0

        # 推定
        with torch.no_grad():
            prediction = model(img_input)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.numpy()
        depth_map = (depth_map - depth_map.min()) / (
            depth_map.max() - depth_map.min()
        )

    with col2:
        st.subheader("生成されたデプスマップ")
        st.image(depth_map, use_container_width=True)

    st.success("計算完了！")
