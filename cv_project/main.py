import streamlit as st


st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)



st.title('📝 &  ⚡💨🍃🪫💡')



st.write('choose your option')



st.page_link("pages/autoenc.py", label="Denoiser DOCS", icon='📝')
st.page_link("pages/myapp.py", label="Wind stations", icon='⚡')

st.header(f'''made by: Alexey Kamaev & Marina Kochetova''')
