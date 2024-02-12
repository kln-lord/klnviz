import io
import operator
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.config
from itertools import combinations
import random
import plotly.express as px

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
                .plotlyjsicon{display:none;}
        </style>
""",unsafe_allow_html=True)
st.title('Vis - visualize and discover the story behind your data with one click')
st.write('<h6>Made by Ahmed Bendrioua</h6>',unsafe_allow_html=True)

uploaded_file = st.file_uploader("Load your data",accept_multiple_files=False,type=["csv", "json"])

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

if "Submit" not in st.session_state:
    st.session_state["Submit"] = False

if "Send" not in st.session_state:
    st.session_state["Send"] = False

target_variable=""
if st.button('Submit'):
    st.session_state["Submit"] = not st.session_state["Submit"]

if st.session_state['Submit']:
    if uploaded_file is not None:
        if uploaded_file.type in  ["text/csv","application/json"]:
            # success upload message
            st.success("data uploaded succefully")
            # reading data
            if uploaded_file.type=="text/csv": df = pd.read_csv(uploaded_file)
            else: df = pd.read_json(uploaded_file)
            # displaying data
            st.write("<h2> your data : "+uploaded_file.name+"</h2>",unsafe_allow_html=True)
            st.write(df)
            # droping non numirecal columns
            object_columns=[column for column in df.columns if df[column].dtype in ['object','datetime64[ns, UTC]']]
            df_num = df.drop(columns=object_columns)
            #desc
            st.write("<h2>Description of the data</h2>",unsafe_allow_html=True)
            st.write(df.describe())
            #infos
            st.write("<h2>informations about the data</h2>",unsafe_allow_html=True)
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            if operator.countOf(df_num.isna().sum().values,0) != len(df_num.isna().sum().values):
                st.header(f"deleting rows with missing values..")
                st.write(pd.DataFrame(df_num.isna().sum()).rename(columns={0:"Number of missing values"}))
                df_num = df_num.dropna(how="any")
                st.header("the data after modification : ")
                st.write(df_num)
                if len(df_num)==0:
                    exit(0)
            #dup
            st.write("<h2>Duplicates in the data</h2>",unsafe_allow_html=True)
            if df_num.duplicated().sum()==0: st.write(f"there are no Duplicates in the data") 
            else: 
                st.write(f"{df_num.duplicated().sum()}")
                df_num = df_num.drop_duplicates()
                st.header("the data after modification : ")
                st.write(df_num)
            # st.header("Bar chart of the data")
            # st.bar_chart(df_num)
            # for column in df_num.columns: 
            #     st.header(f"{column} Bar chart")
            #     st.bar_chart(df_num[column],y=column,color=colors[random.randint(0,len(colors))-1])
            # fig, ax = plt.subplots()
            # ax.hist(df_num['Population'],bins=500)
            # st.pyplot(fig)
            # st.data_editor(df,hide_index=True)
            # //
            # st.write("<h2>HeatMap<i> to show the correlation between the variables<i></h2>",unsafe_allow_html=True)
            # fig, ax = plt.subplots()
            # sns.heatmap(df_num.corr(), ax=ax,annot=True)
            # st.write(fig)
            st.write("<h2>Computing pairwise correlation of columns</h2>",unsafe_allow_html=True)
            st.write(df_num.corr())
            # st.write("<h2>SCATTER PLOT</h2>",unsafe_allow_html=True)
            # var = list()
            
            # i=0
            # for column_1 in df_num.columns:
            #     var.append(column_1)
            #     for column_2 in df_num.columns:
            #         if column_2 not in var:
            #             # fig, ax = plt.subplots()
            #             # sns.scatterplot(df_num,x=str(column_1),y=str(column_2), ax=ax,annot=True)
            #             # st.write(fig)
            #             st.scatter_chart(df_num,x=column_1,y=column_2,color=colors[random.randint(0,len(colors))-1])
            #             i+=1
            st.header("choose how you want to plot your Data")
            col1, col2, col3, col4, col5 = st.columns(5,gap="small")

            with col1:
                Heatmap_btn = st.button('Heatmap')

            with col2:
                barchChart_btn = st.button('Bar chart')
            with col3:
                scatterPlot_btn = st.button('Scatter plot')
            with col4:
                boxPlot_btn = st.button('Box plot')
            with col5:
                lineChart_btn = st.button('Line chart')

            if Heatmap_btn:
                st.write("<h2>HeatMap<i> to show the correlation between the variables<i></h2>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                sns.heatmap(df_num.corr(), ax=ax,annot=True)
                st.write(fig)
                

            if barchChart_btn:
                st.header("Bar chart of the data")
                var = list()
                i=0
                for column in df_num.columns:
                    st.subheader(f"Bar chart of {column}")
                    st.bar_chart(df_num[column],color=colors[random.randint(0,len(colors))-1])
                st.subheader(f"Bar chart of all variables")
                st.bar_chart(df_num)
               # Do something...

            if scatterPlot_btn:
                st.write("<h2>Scatter Plot of each pair variables</h2>",unsafe_allow_html=True)
                var = list()
                i=0
                for column_1 in df_num.columns:
                    var.append(column_1)
                    for column_2 in df_num.columns:
                        if column_2 not in var:
                            st.subheader(f"Scatter plot of {column_2} and {column_1}")
                            st.scatter_chart(df_num,x=column_1,y=column_2,color=colors[random.randint(0,len(colors))-1])
                            i+=1

            if boxPlot_btn:
                # var = list()
                # for column_1 in df_num.columns:
                #     var.append(column_1)
                #     for column_2 in df_num.columns:
                #             if column_2 not in var:
                #                 st.subheader(f"Box plot of {column_2} and {column_1}")
                #                 fig, ax = plt.subplots()
                #                 st.write(fig)
                st.write("<h2>Box Plots of the Data</h2>",unsafe_allow_html=True)
                for column in df_num.columns:
                    st.subheader(f"Box plot of {column}")
                    fig = px.box(df_num,y=column,points="all")
                    st.plotly_chart(fig,theme="streamlit",use_container_width=True)
                    
            if lineChart_btn:
                st.write("<h2>Line chart of each pair variables</h2>",unsafe_allow_html=True)
                var = list()
                i=0
                for column_1 in df_num.columns:
                    var.append(column_1)
                    for column_2 in df_num.columns:
                        if column_2 not in var:
                            st.subheader(f"Line plot of {column_2} and {column_1}")
                            st.line_chart(df_num,x=column_1,y=column_2,color=colors[random.randint(0,len(colors))-1])
                st.subheader(f"Line plot of all variables")
                st.line_chart(df_num)
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
                    stat,stat1=False,False
                    val = {df_num.corr()[target_variable].index[i]: df_num.corr()[target_variable].values[i] for i in range(len(df_num.corr()[target_variable].values))}
                    # st.write(val)
                    for key,value in val.items():
                        if abs(value)<=0.2:
                            stat=True
                            st.info(f"we notice that there isn't a significant correlation between {key} and {target_variable} : {value} therefor the entry variable {key} doesn't explain the target variable")
                            val = {df_num.corr()[key].index[i]: df_num.corr()[key].values[i] for i in range(len(df_num.corr()[key].values))}
                            for key_sub,value1 in val.items():
                                if key!=key_sub and abs(value1)>=0.4:
                                    stat1=True
                            if stat1:
                                st.header(f"Compute pairwise correlation of {key} and the other columns")
                                st.write(df_num.corr()[key])
                                st.info(f"there is significant correlation between {key} and {key_sub} : {value1} which means that the two entry variables contain same information so it's perferable to remove {key}")


                    if not stat:
                        st.write("all variables explain the target value")

        else:
            error_msg = "files of type "+ uploaded_file.type +" are not supported"
            st.error(error_msg)

    else:
        st.error('Load your data first !!')
else:
    st.info('Load your data')

# if st.session_state["Send"]:

    
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