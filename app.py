import streamlit as st
from PIL import Image, ImageChops
import numpy as np
import torch
from rembg import remove

st.set_page_config(page_title="3Dパララックス生成器", layout="centered")

st.title("3Dパララックス生成器")
st.write("画像をアップロードして、手前と背景をずらした立体的なプレビューを作成します。")

# MiDaSモデルのロード
@st.cache_resource
def load_midas():
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    return midas, transform, device

uploaded_file = st.file_uploader("画像をアップロード（JPG / PNG）", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="元画像", use_column_width=True)

    if st.button("3D空間を生成"):
        with st.spinner("AIで被写体の切抜きと奥行き解析を行っています..."):
            # 1. 被写体切り抜き
            fg_image = remove(input_image)
            
            # 2. 深度マップ生成
            midas, transform, device = load_midas()
            img_np = np.array(input_image)
            input_batch = transform(img_np).to(device)
            
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_np.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            # 正規化
            depth_map = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            depth_pil = Image.fromarray(depth_map)

            st.session_state["fg_image"] = fg_image
            st.session_state["input_image"] = input_image
            st.session_state["depth_map"] = depth_pil
            st.success("解析完了！")

if "fg_image" in st.session_state:
    st.subheader("立体プレビュー操作")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state["fg_image"], caption="抽出した前景", use_column_width=True)
    with col2:
        st.image(st.session_state["depth_map"], caption="深度マップ", use_column_width=True)

    offset_x = st.slider("左右のシフト (X軸)", -30, 30, 0)
    offset_y = st.slider("上下のシフト (Y軸)", -30, 30, 0)

    # Pillowを使った合成処理 (OpenCVに依存しない)
    bg_img = st.session_state["input_image"].copy()
    fg_img = st.session_state["fg_image"].copy()

    # 背景をずらす
    bg_shifted = ImageChops.offset(bg_img, offset_x, offset_y)

    # 前景を上に重ねる
    composite = bg_shifted.copy()
    composite.paste(fg_img, (0, 0), fg_img)

    st.image(composite, caption="合成結果", use_column_width=True)
