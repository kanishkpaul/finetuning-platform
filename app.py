# app.py

import streamlit as st
import tempfile
from finetune_backend import fine_tune_model
import torch

st.set_page_config(page_title="Fine-tune Your Model", layout="centered")
st.title("🧠 Fine-Tune Your Model on Excel Data")

model_file = st.file_uploader("Upload Model Weights (.safetensors)", type=["safetensors"])
config_file = st.file_uploader("Upload Model Config (config.json)", type=["json"])
tokenizer_file = st.file_uploader("Upload Tokenizer (tokenizer.json)", type=["json"])
excel_file = st.file_uploader("Upload Dataset (.xlsx)", type=["xlsx"])
use_gpu = st.checkbox("Use GPU (WebGPU/DirectML)", value=True)

if st.button("🚀 Start Fine-tuning"):
    if model_file and config_file and tokenizer_file and excel_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as model_temp:
            model_temp.write(model_file.read())
            model_path = model_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as config_temp:
            config_temp.write(config_file.read())
            config_path = config_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tokenizer_temp:
            tokenizer_temp.write(tokenizer_file.read())
            tokenizer_path = tokenizer_temp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as excel_temp:
            excel_temp.write(excel_file.read())
            excel_path = excel_temp.name

        device = torch.device("dml") if use_gpu else torch.device("cpu")

        with st.spinner("Fine-tuning in progress..."):
            out_dir = fine_tune_model(
                config_path=config_path,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                excel_path=excel_path,
                device=device
            )

        st.success(f"✅ Fine-tuning complete! Model saved to: {out_dir}")
    else:
        st.warning("Please upload all required files: model, config, tokenizer, and dataset.")
