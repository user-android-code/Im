import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from rembg import remove

st.set_page_config(page_title="3Dパララックス生成器", layout="centered")

st.title("3Dパララックス生成器")
st.write("画像をアップロードして、手前と背景をずらした立体的なプレビューを作成します。")

# MiDaSモデルのロード（軽量モデルを使用）
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
        with st.spinner("AIで被写体の切抜きと奥行き解析を行っています...（数十秒かかる場合があります）"):
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
            depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # セッション状態へ保存
            st.session_state["fg_image"] = fg_image
            st.session_state["img_np"] = img_np
            st.session_state["depth_map"] = depth_map
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

    # 前景と背景をずらして合成
    img_np = st.session_state["img_np"]
    fg_np = np.array(st.session_state["fg_image"])

    # 背景をシフト
    bg_shifted = np.roll(img_np, (offset_y, offset_x), axis=(0, 1))

    # アルファチャンネルを使って合成
    alpha = fg_np[:, :, 3] / 255.0
    composite = np.zeros_like(img_np)
    for c in range(3):
        composite[:, :, c] = fg_np[:, :, c] * alpha + bg_shifted[:, :, c] * (1 - alpha)

    st.image(composite, caption="合成結果", use_column_width=True)
