import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Ahmed Bendrioua | Vis", layout="wide",page_icon="favicon.png")
# st.session_state.theme="light"
st.markdown("""
        <style>
                .ea3mdgi8{background-color: #181818;}
                .e1b2p2ww15{background-color: rgb(81 81 87);}
                .st-emotion-cache-19rxjzo{background-color: rgb(40 40 40);}
                .ezrtsby2{display:none;}
                summary{display:none;}
                .e1nzilvr3{display:none}
                .stAlert{background-color: #ababab;border-radius: 10px;}
                .stAlert .e1nzilvr5{color:black;}
                .e115fcil1 img{scale:0.6;}
        </style>
""",unsafe_allow_html=True)
st.title('Vis - visulaize and Discover the story behind your data with one click')
st.write('<h6>Made by Ahmed Bendrioua</h6>',unsafe_allow_html=True)

uploaded_file = st.file_uploader("Load your data",accept_multiple_files=False)

if st.button('Submit'):
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.write(df)
            object_columns=[column for column in df.columns if df[column].dtype=='object']
            df_num = df.drop(columns=object_columns)
            st.write("<h2>DESCRIBE</h2>",unsafe_allow_html=True)
            st.write(df.describe())
            st.write("<h2>HeatMap<i> to show the correlation between the variables<i></h2>",unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.heatmap(df_num.corr(), ax=ax,annot=True)
            st.write(fig)
            st.write("<h2>SCATTER PLOT</h2>",unsafe_allow_html=True)
            st.scatter_chart(df_num,x='Nb_moyen_pieces',y='Population',color="#ff2828")
        else:
            error_msg = "files of type "+ uploaded_file.type +" are not supported"
            st.error(error_msg)

    else:
        st.error('Load your data first !!')

else:
    st.info('Load your data')
    st.write("""
            <footer class="frame frame--footer" style="display:none;">
                <p class="frame__author"><span>Made by <a target="_blank" href="https://www.linkedin.com/in/ahmedbendrioua/">@ahmedbendrioua</a></span> <span><a target="_blank" href="mailto:ahmedbendriouaa@gmail.com">Hire Me</a></span></p>
            </footer>          
             """,unsafe_allow_html=True)
