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
# st.page_link("pages/streamlit_sport_model.py", label="Sport detector", icon='⚡')

st.header(f'''made by: Alexey Kamaev & Marina Kochetova''')