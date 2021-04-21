from pathlib import Path

import hydra
import streamlit as st
import torch

from src.common.utils import render_images
from src.pl_modules.model import MyModel
from src.ui.ui_utils import get_hydra_cfg
from src.ui.ui_utils import select_checkpoint


@st.cache(allow_output_mutation=True)
def get_model(checkpoint_path: Path):
    return MyModel.load_from_checkpoint(checkpoint_path=str(checkpoint_path))


cfg = get_hydra_cfg()

checkpoint_path = select_checkpoint()
model: MyModel = get_model(checkpoint_path=checkpoint_path)
dataset = hydra.utils.instantiate(
    cfg.data.datamodule.datasets.test[0], _recursive_=False
)

sample_idx = st.number_input(
    "Insert the sample index", min_value=0, max_value=len(dataset), value=0
)

image, y = dataset[sample_idx]

noise = torch.randn_like(image)

alpha = st.number_input("Noise level:", min_value=0.0, max_value=10.0, value=0.0)

image = image + alpha * noise

image = image[None, ...]
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

st.header(f"Ground truth class: `{classes[y]}`")
st.image(render_images(image, autoshow=False), use_column_width=True)

logits = model(image)
y_pred = logits.argmax(-1)

st.header(f"Predicted class: `{classes[y_pred]}`")
