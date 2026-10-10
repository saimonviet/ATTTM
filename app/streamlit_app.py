"""Demo Streamlit: tấn công đối kháng FGSM/PGD và phòng thủ trên model CIFAR-10.

Chạy:  streamlit run app/streamlit_app.py
Giao diện chỉ gọi adversarial_lab.demo; không huấn luyện lại và không giả lập kết quả khi chưa có checkpoint.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from adversarial_lab.demo import list_checkpoints, preprocess_upload, run_demo
from adversarial_lab.device import get_device
from adversarial_lab.utils.checkpoint import load_checkpoint

CKPT_DIR = Path(os.environ.get("ADV_LAB_CKPT_DIR", ROOT / "checkpoints"))
DATA_DIR = Path(os.environ.get("ADV_LAB_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("ADV_LAB_RESULTS_DIR", ROOT / "results"))
DEVICE = get_device("auto")

st.set_page_config(page_title="Adversarial Attack & Defense - CIFAR-10", layout="wide")


@st.cache_resource(show_spinner="Đang tải checkpoint...")
def get_model(path: str, mtime: float):
    return load_checkpoint(path, DEVICE)


@st.cache_resource(show_spinner=False)
def get_testset():
    try:
        return CIFAR10(str(DATA_DIR), train=False, download=False, transform=ToTensor())
    except RuntimeError:
        return None


def show(t: torch.Tensor, caption: str, size: int = 192):
    img = Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8"))
    st.image(img.resize((size, size), Image.NEAREST), caption=caption)


def perturbation_view(x: torch.Tensor, x_adv: torch.Tensor, eps: float) -> torch.Tensor:
    return (0.5 + (x_adv - x) / (2 * max(eps, 1e-8))).clamp(0, 1)


def prob_chart(res: dict, names: list[str]):
    df = pd.DataFrame({"Trước attack": res["probs_clean"].numpy(), "Sau attack": res["probs_adv"].numpy()}, index=names)
    top = df.assign(m=df.max(axis=1)).sort_values("m", ascending=False).head(5).drop(columns="m")
    st.bar_chart(top, height=220)


def result_block(title: str, res: dict, x: torch.Tensor, eps: float, names: list[str], label: int | None):
    st.subheader(title)
    c1, c2, c3 = st.columns(3)
    with c1:
        show(x, "Ảnh đầu vào")
    with c2:
        show(perturbation_view(x, res["x_adv"], eps), "Perturbation (phóng đại: 0.5 + δ/(2ε), xám = không đổi)")
    with c3:
        show(res["x_adv"], "Ảnh sau attack")
    m = st.columns(4)
    m[0].metric("‖δ‖∞ (pixel [0,1])", f"{res['linf']:.4f}")
    m[1].metric("‖δ‖₂", f"{res['l2']:.3f}")
    m[2].metric("Dự đoán trước attack", names[res["pred_clean"]], f"{100 * res['conf_clean']:.1f}%", delta_color="off")
    m[3].metric("Dự đoán sau attack", names[res["pred_adv"]], f"{100 * res['conf_adv']:.1f}%", delta_color="off")
    if label is not None:
        ok0, ok1 = res["pred_clean"] == label, res["pred_adv"] == label
        st.write(f"Nhãn thật: **{names[label]}** | trước attack: {'đúng' if ok0 else 'sai'} | sau attack: {'đúng' if ok1 else 'sai'}"
                 + (" | **attack thành công trên ảnh này**" if ok0 and not ok1 else ""))
    elif res["changed"]:
        st.write("Dự đoán đã bị thay đổi bởi attack (ảnh tải lên không có nhãn thật).")
    prob_chart(res, names)


st.title("Adversarial Attack & Defense trên CIFAR-10")
st.caption("Demo học tập (Đề tài 2.28, lớp Nh22.15). Kết quả trên một ảnh chỉ mang tính minh họa; đánh giá định lượng nằm trong results/.")

cks = list_checkpoints(CKPT_DIR, st.sidebar.checkbox("Hiện checkpoint smoke test", value=False))
if not cks:
    st.error(f"Chưa có checkpoint trong `{CKPT_DIR}`. Hãy huấn luyện trước (demo không giả lập kết quả):")
    st.code("python scripts/train_baseline.py --config configs/config.yaml\n"
            "python scripts/train_adversarial.py --config configs/config.yaml", language="bash")
    st.stop()

# ---- Sidebar -------------------------------------------------------------------------------
st.sidebar.header("Model")
default_ck = next((i for i, c in enumerate(cks) if "baseline" in c.name), 0)
ck_name = st.sidebar.selectbox("Checkpoint (model bị tấn công)", [c.name for c in cks], index=default_ck)
ck = CKPT_DIR / ck_name
model, meta = get_model(str(ck), ck.stat().st_mtime)
names = meta["class_names"]
st.sidebar.caption(f"{meta.get('training', 'unknown')} | best epoch: {meta.get('best_epoch', '?')} | device: {DEVICE}")

st.sidebar.header("Ảnh đầu vào")
source = st.sidebar.radio("Nguồn ảnh", ["CIFAR-10 test set", "Tải ảnh lên"])
x, label, out_of_dist = None, None, False
if source == "CIFAR-10 test set":
    ds = get_testset()
    if ds is None:
        st.sidebar.warning(f"Chưa có dữ liệu CIFAR-10 trong `{DATA_DIR}`. Chạy một script train để tự tải, hoặc dùng ảnh tải lên.")
    else:
        idx = st.sidebar.number_input("Chỉ số ảnh (0-9999)", 0, len(ds) - 1, 0)
        img, label = ds[int(idx)]
        x = img.unsqueeze(0)
else:
    up = st.sidebar.file_uploader("Ảnh (png/jpg)", type=["png", "jpg", "jpeg"])
    if up is not None:
        x, out_of_dist = preprocess_upload(Image.open(up)), True
if x is None:
    st.info("Chọn ảnh CIFAR-10 hoặc tải ảnh lên ở thanh bên.")
    st.stop()
if out_of_dist:
    st.warning("Ảnh tải lên nằm NGOÀI phân phối CIFAR-10 (đã cắt vuông và resize 32×32, không có nhãn thật). "
               "Kết quả không đại diện cho benchmark CIFAR-10.")

st.sidebar.header("Attack")
attack = {"None": "none", "FGSM": "fgsm", "PGD": "pgd"}[st.sidebar.selectbox("Attack", ["None", "FGSM", "PGD"], index=2)]
eps = st.sidebar.slider("Epsilon (L∞, pixel [0,1])", 0.0, 0.1, 0.03, 0.005, disabled=attack == "none")
steps = st.sidebar.slider("Số iteration PGD", 1, 50, 10, disabled=attack != "pgd")
rstart = st.sidebar.checkbox("Random start (PGD)", value=True, disabled=attack != "pgd")

st.sidebar.header("Defense")
defense_ui = st.sidebar.selectbox("Defense", ["None", "Gaussian Blur", "Model adversarially trained"])
kw = {"defense": "none"}
if defense_ui == "Gaussian Blur":
    kw.update(defense="gaussian_blur", blur_kernel=st.sidebar.selectbox("Kernel size", [3, 5, 7]),
              blur_sigma=st.sidebar.slider("Sigma", 0.3, 2.0, 1.0, 0.1),
              adaptive=st.sidebar.checkbox("Attack thích nghi (tạo attack qua blur)", value=False))
elif defense_ui == "Model adversarially trained":
    adv_default = next((i for i, c in enumerate(cks) if "adv" in c.name), 0)
    d_name = st.sidebar.selectbox("Checkpoint phòng thủ", [c.name for c in cks], index=adv_default)
    d_path = CKPT_DIR / d_name
    d_model, d_meta = get_model(str(d_path), d_path.stat().st_mtime)
    kw.update(defense="adversarial_model", defense_model=d_model, defense_name=d_name.replace(".pth", ""),
              adaptive=st.sidebar.checkbox("Attack tạo trực tiếp trên model phòng thủ (white-box)", value=True))
    if "adversarial" not in str(d_meta.get("training", "")):
        st.sidebar.warning("Checkpoint này không được huấn luyện đối kháng (theo metadata).")

# ---- Chạy và hiển thị --------------------------------------------------------------------------
res = run_demo(model, x, label, attack, eps, steps, rstart, model_name=ck_name.replace(".pth", ""), **kw)
if attack == "none":
    st.info("Chưa chọn attack: ảnh sau attack giống ảnh gốc.")
result_block(f"Model: {res['main']['model']} (attack tạo trên model này)", res["main"], x, eps, names, label)
if res["defended"] is not None:
    note = {"gaussian_blur": "adaptive: attack tạo qua blur" if kw.get("adaptive") else "không thích nghi: dùng ảnh đối kháng của model gốc, blur trước khi phân loại",
            "adversarial_model": "white-box trên model phòng thủ" if kw.get("adaptive") else "transfer: ảnh đối kháng từ model gốc"}[kw["defense"]]
    result_block(f"Với defense: {res['defended']['model']} ({note})", res["defended"], x, eps, names, label)

with st.expander("Kết quả đánh giá định lượng (toàn bộ test set, từ results/tables)"):
    p = RESULTS_DIR / "tables" / "defense_table.csv"
    if p.exists():
        t = pd.read_csv(p).drop(columns=["Label", "Group", "Short"], errors="ignore")
        st.dataframe(t, use_container_width=True)
    else:
        st.write("Chưa có `defense_table.csv`. Chạy `python scripts/evaluate_all.py --config configs/config.yaml`.")
