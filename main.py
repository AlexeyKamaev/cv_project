import streamlit as st




st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)



st.title('📝 &  ⚡💨🍃🪫💡')

import streamlit as st
from streamlit_image_comparison import image_comparison

# set page config
st.set_page_config(page_title="Image-Comparison Example", layout="centered")

# render image-comparison
image_comparison(
    img1="image1.jpg",
    img2="image2.jpg",
)

st.write('choose your option')



st.page_link("pages/autoenc.py", label="Denoiser DOCS", icon='📝')
st.page_link("pages/myapp.py", label="Wind stations", icon='⚡')

st.header(f'''made by: Alexey Kamaev & Marina Kochetova''')
