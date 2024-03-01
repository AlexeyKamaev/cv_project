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

st.markdown(st.page_link("pages/autoenc.py", label="Denoiser DOCS", icon='📝'), unsafe_allow_html=True)
with st.echo():
    st_lottie("https://lottie.host/d50d44da-8c5f-48da-af8b-57ad93b6e14c/h139wgXhFw.json")

st.page_link("pages/myapp.py", label="Wind stations", icon='⚡')
with st.echo():
    st.lottie("https://lottie.host/a7d94a5a-41ff-428d-8760-1d6445f6a4dc/N7tAKQoF92.json")
    


st.subheader(f'''made by: Alexey Kamaev & Marina Kochetova''')
