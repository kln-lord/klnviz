import io
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.config
from itertools import combinations
import random

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

if "Submit" not in st.session_state:
    st.session_state["Submit"] = False

if "Send" not in st.session_state:
    st.session_state["Send"] = False

target_variable=""
if st.button('Submit'):
    st.session_state["Submit"] = not st.session_state["Submit"]

if st.session_state['Submit']:
    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.write("<h2> your data : "+uploaded_file.name+"</h2>",unsafe_allow_html=True)
            st.write(df)
            object_columns=[column for column in df.columns if df[column].dtype=='object']
            df_num = df.drop(columns=object_columns)
            st.write("<h2>Description of the data</h2>",unsafe_allow_html=True)
            st.write(df.describe())
            st.write("<h2>informations about the data</h2>",unsafe_allow_html=True)
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            st.write("<h2>HeatMap<i> to show the correlation between the variables<i></h2>",unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.heatmap(df_num.corr(), ax=ax,annot=True)
            st.write(fig)
            st.write("<h2>Computing pairwise correlation of columns</h2>",unsafe_allow_html=True)
            st.write(df_num.corr())
            st.write("<h2>SCATTER PLOT</h2>",unsafe_allow_html=True)
            var = list()
            colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                "#b3e1ff", "#ffdb58", "#aaf0d1", "#f88379", "#9fb6cd",
                "#ffb3ff", "#ff6347", "#8b3a3a", "#4876ff", "#7b68ee",
                "#b0c4de", "#c71585", "#00ff00", "#ffd700", "#ff7f50",
                "#ff69b4", "#ffc0cb", "#800000", "#3cb371", "#4682b4",
                "#afeeee", "#db7093", "#ff00ff", "#ba55d3", "#9370db",
                "#ffdab9", "#ffa07a", "#dda0dd", "#da70d6", "#ff4500",
                "#ff6347", "#ff8c00", "#ff69b4", "#ff1493", "#ff00ff",
                "#ff0000", "#ee82ee", "#e9967a", "#e0ffff", "#e0e0e0",
                "#deb887", "#d2b48c", "#d8bfd8", "#d8bfd8", "#cd5c5c",
                "#c71585", "#b8860b", "#b0e0e6", "#b0c4de", "#a52a2a",
                "#a9a9a9", "#8fbc8f", "#7fffd4", "#7fff00", "#7cfc00",
                "#708090", "#6b8e23", "#6495ed", "#5f9ea0", "#556b2f",
                "#4682b4", "#2e8b57", "#228b22", "#20b2aa", "#191970",
                "#00ffff", "#00ff7f", "#00ff00", "#00ced1", "#008080",
                "#008000", "#006400", "#0000ff", "#0000cd", "#000080",
                "#fffff0", "#ffffe0", "#ffff00", "#ffdead", "#ffd700",
                "#ff4500", "#ff1493", "#ee82ee", "#eedd82", "#e9967a",
                "#e6e6fa", "#e0ffff", "#e0e0e0", "#db7093", "#d8bfd8"
            ]
            i=0
            for column_1 in df_num.columns:
                var.append(column_1)
                for column_2 in df_num.columns:
                    if column_2 not in var:
                        st.scatter_chart(df_num,x=column_1,y=column_2,color=colors[random.randint(0,len(colors))-1])
                        i+=1
            st.header("Let us filter your data!")
            target_variable = st.text_input("Enter your target variable","")
            if st.button("Send"):
                st.session_state["Send"] = not st.session_state["Send"]
                if target_variable not in df.columns:
                    error_msg = "the variable " +target_variable +" doesn't exist in the data, try again"
                    st.error(error_msg)
                else:
                    st.header(f"Compute pairwise correlation of {target_variable} and the other columns")
                    st.write(df_num.corr()[target_variable])
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

# st.write(
#     f"""
#     ## Session state:
#     {st.session_state["Submit"]=}

#     {st.session_state["Send"]=}

#     """
# )