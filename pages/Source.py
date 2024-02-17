import streamlit as st
st.set_page_config(page_title="Sources")
st.title("Sources")
st.write(st.session_state['MetaData_source'][0][0:-1])
# st.sidebar.text(str(source_vect[-1]))