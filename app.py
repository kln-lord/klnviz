import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.config
from itertools import combinations

st.set_page_config(page_title="Ahmed Bendrioua | Vis", layout="wide",page_icon="favicon.png")
# st.session_state.theme="light"
streamlit.config.set_option("theme.base","light")
streamlit.config.set_option('server.enableXsrfProtection', False)
streamlit.config.set_option('server.enableCORS', False)
# .ea3mdgi8{background-color: #181818;}
#                 .e1b2p2ww15{background-color: rgb(81 81 87);}
#                 .st-emotion-cache-19rxjzo{background-color: rgb(40 40 40);}
st.markdown("""
        <style>
                .ezrtsby2{display:none;}
                summary{display:none;}
                .e1nzilvr3{display:none}
                .stAlert{background-color: #ababab;border-radius: 10px;}
                .stAlert .e1nzilvr5{color:black;}
        </style>
""",unsafe_allow_html=True)
st.title('Vis - visualize and discover the story behind your data with one click')
st.write('<h6>Made by Ahmed Bendrioua</h6>',unsafe_allow_html=True)

uploaded_file = st.file_uploader("Load your data",accept_multiple_files=False)

if st.button('Submit'):
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.write("<h2> your data : "+uploaded_file.name+"</h2>",unsafe_allow_html=True)
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
            var = list()
            colors = ['#FAEBD7', '#00FFFF', '#7FFFD4', '#F5F5DC', '#FFE4C4', '#000000', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887', '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#DC143C', '#00008B', '#008B8B', '#B8860B', '#A9A9A9', '#006400', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B', '#2F4F4F', '#00CED1', '#9400D3', '#FF1493', '#00BFFF', '#696969', '#1E90FF', '#B22222', '#228B22', '#FF00FF', '#DCDCDC', '#FFD700', '#DAA520', '#808080', '#008000', '#ADFF2F', '#FF69B4', '#CD5C5C', '#4B0082', '#F0E68C', '#E6E6FA', '#FFF0F5', '#7CFC00', '#FFFACD', '#ADD8E6', '#F08080', '#FAFAD2', '#90EE90', '#D3D3D3', '#FFB6C1', '#FFA07A', '#20B2AA', '#87CEFA', '#778899', '#B0C4DE', '#FFFFE0', '#00FF00', '#32CD32', '#FAF0E6', '#800000', '#66CDAA', '#0000CD', '#BA55D3', '#9370DB', '#3CB371', '#7B68EE', '#00FA9A', '#48D1CC', '#C71585', '#191970', '#F5FFFA', '#FFE4E1', '#FFE4B5', '#FFDEAD', '#000080', '#FDF5E6', '#808000', '#6B8E23', '#FFA500', '#FF4500', '#DA70D6', '#EEE8AA', '#98FB98', '#AFEEEE', '#DB7093', '#FFEFD5', '#FFDAB9', '#CD853F', '#FFC0CB', '#DDA0DD', '#B0E0E6', '#800080', '#663399', '#FF0000', '#BC8F8F', '#4169E1', '#8B4513', '#FA8072', '#F4A460', '#2E8B57', '#FFF5EE', '#A0522D', '#C0C0C0', '#87CEEB', '#6A5ACD', '#708090', '#FFFAFA', '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', '#FF6347', '#40E0D0', '#EE82EE', '#F5DEB3']

            i=0
            for column_1 in df_num.columns:
                var.append(column_1)
                for column_2 in df_num.columns:
                    if column_2 not in var:
                        st.scatter_chart(df_num,x=column_1,y=column_2,color=colors[i])
                        i+=1
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
