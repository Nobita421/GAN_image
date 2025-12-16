import streamlit as st
import yaml
import base64
from inference import generate, load_config
from PIL import Image

st.set_page_config(page_title='Synthetic Image Generator', layout='wide')

cfg = load_config()

st.title('Synthetic Image Generator — Vanilla GAN (Keras)')

cols = st.columns([1,2])
with cols[0]:
    n = st.slider('Number of images', min_value=1, max_value=64, value=16)
    seed = st.number_input('Seed (0 for random)', min_value=0, value=0)
    size = st.selectbox('Image Size', options=[32,64,128], index=1)
    if st.button('Generate'):
        seed_val = None if seed == 0 else int(seed)
        with st.spinner('Generating...'):
            paths = generate(n=n, seed=seed_val, out_dir='./streamlit_out')
        st.success('Generated')
        imgs = [Image.open(p) for p in paths]
        cols2 = st.columns(4)
        for i, im in enumerate(imgs):
            cols2[i%4].image(im, use_column_width=True)
        # zip download
        import zipfile, io
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            for p in paths:
                zf.write(p)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        href = f"data:application/zip;base64,{b64}"
        st.markdown(f"[Download images]({href})")

with cols[1]:
    st.markdown('''
    **Instructions**
    - Put your dataset (images) under the `data/` folder and set `dataset_path` in `config.yaml`.
    - Run training: `python train.py`
    - Start Streamlit: `streamlit run app.py`
    ''')
