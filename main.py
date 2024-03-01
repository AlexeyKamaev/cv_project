import streamlit as st
from streamlit_image_comparison import image_comparison
from streamlit_lottie import st_lottie

st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)



st.title('📝 &  ⚡💨🍃🪫💡')

st.write('choose your option')


with st.echo():
    st_lottie("https://lottie.host/d50d44da-8c5f-48da-af8b-57ad93b6e14c/h139wgXhFw.json")
st.markdown(st.page_link("pages/autoenc.py", label="Denoiser DOCS", icon='📝'), unsafe_allow_html=True)


with st.echo():
    st_lottie("https://lottie.host/embed/65695457-cee0-4a43-bad6-33ff5ab81798/n77e59fnDc.json")
    
st.page_link("pages/myapp.py", label="Wind stations", icon='⚡')

st.header(f'''made by: Alexey Kamaev & Marina Kochetova''')
