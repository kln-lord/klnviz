import io
import math
import operator
import openai
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.config
import random
import plotly.express as px
import statsmodels.formula.api as smf

openai.api_key="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"
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
                .st-emotion-cache-7ym5gk:hover {
                    border-color: rgb(0, 0, 0);
                    color: rgb(0, 0, 0);
                }
                .st-emotion-cache-7ym5gk:focus:not(:active) {
                    border-color: rgb(0, 0, 0);
                    color: rgb(0, 0, 0);
                }
                .st-emotion-cache-7ym5gk:active {
                    color: rgb(255, 255, 255);
                    border-color: rgb(0, 0, 0);
                    background-color: rgb(0, 0, 0);
                }
                .ezrtsby2{display:none;}
                summary{display:none;}
                .e1nzilvr3{display:none}
                .stAlert{background-color: #ababab;border-radius: 10px;}
                .stAlert .e1nzilvr5{color:black;}
                .plotlyjsicon{display:none;}
                .simpletable{margin-bottom:50px}
        </style>
""",unsafe_allow_html=True)
st.title('Vis - visualize and discover the story behind your data')
st.write('<h6>Made by Ahmed Bendrioua</h6>',unsafe_allow_html=True)

uploaded_file = st.file_uploader("Load your data",accept_multiple_files=False,type=["csv", "json","xlsx"])

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

if "scatterPlot" not in st.session_state:
    st.session_state["scatterPlot"] = False

if "lineChart" not in st.session_state:
    st.session_state["lineChart"] = False

if "barChart" not in st.session_state:
    st.session_state["barChart"] = False

if "boxPlot" not in st.session_state:
    st.session_state["boxPlot"] = False
if "Edit" not in st.session_state:
    st.session_state['Edit'] = False


target_variable=""
if st.button('Submit'):
    st.session_state["Submit"] = not st.session_state["Submit"]

if st.session_state['Submit']:
    if uploaded_file is not None:
        if uploaded_file.type in  ["text/csv","application/json","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            # success upload message
            st.success("data uploaded succefully")
            # reading data
            if uploaded_file.type=="text/csv": df = pd.read_csv(uploaded_file)
            elif uploaded_file.type=="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": 
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            for column in df.columns:
                df = df.rename(columns={column:column.replace(" ","_")})
            for column in df.columns:
                if "unnamed" in column.lower(): df=df.drop(columns=[column])

            # displaying data
            st.write("<h2> your data : "+uploaded_file.name+"</h2>",unsafe_allow_html=True)
            st.write(df)
            st.header("Edit your data")
            st.write("Hit Enter to save changes")
            st.session_state['Edit'] = not st.session_state['Edit']
            df = st.data_editor(data=df,hide_index=True,num_rows='dynamic')
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
                df_num = df_num.drop_duplicates(ignore_index=True)
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
                st.session_state["lineChart"] = False
                st.session_state["scatterPlot"] = False
                st.session_state["barChart"] = False
                st.session_state["boxPlot"] = False
                st.write("<h2>HeatMap<i> to show the correlation between the variables<i></h2>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                sns.heatmap(df_num.corr(), ax=ax,annot=True)
                st.write(fig)
                

            if barchChart_btn:
                st.session_state["lineChart"] = False
                st.session_state["scatterPlot"] = False
                st.session_state["barChart"] = not st.session_state["barChart"]
                st.session_state["boxPlot"] = False
                # st.header("Bar chart of the data")
                # var = list()
                # i=0
                # for column in df_num.columns:
                #     st.subheader(f"Bar chart of {column}")
                #     st.bar_chart(df_num[column],color=colors[random.randint(0,len(colors))-1])
                # st.subheader(f"Bar chart of all variables")
                # st.bar_chart(df_num)
               # Do something...

            if scatterPlot_btn:
                st.session_state["lineChart"] = False
                st.session_state["scatterPlot"] = not st.session_state["scatterPlot"] 
                st.session_state["barChart"] = False
                st.session_state["boxPlot"] = False
                
                # st.write("<h2>Scatter Plot of each pair variables</h2>",unsafe_allow_html=True)
                # var = list()
                # i=0
                # for column_1 in df_num.columns:
                #     var.append(column_1)
                #     for column_2 in df_num.columns:
                #         if column_2 not in var:
                #             st.subheader(f"Scatter plot of {column_2} and {column_1}")
                #             st.scatter_chart(df_num,x=column_1,y=column_2,color=colors[random.randint(0,len(colors))-1])
                #             i+=1

            if boxPlot_btn:
                st.session_state["lineChart"] = False
                st.session_state["scatterPlot"] = False
                st.session_state["boxPlot"] = not st.session_state["boxPlot"]
                st.session_state["barChart"] = False
                # var = list()
                # for column_1 in df_num.columns:
                #     var.append(column_1)
                #     for column_2 in df_num.columns:
                #             if column_2 not in var:
                #                 st.subheader(f"Box plot of {column_2} and {column_1}")
                #                 fig, ax = plt.subplots()
                #                 st.write(fig)
                # st.write("<h2>Box Plots of the Data</h2>",unsafe_allow_html=True)
                # for column in df_num.columns:
                #     st.subheader(f"Box plot of {column}")
                #     fig = px.box(df_num,y=column,points="all")
                #     st.plotly_chart(fig,theme="streamlit",use_container_width=True)
                    
            if lineChart_btn:
                st.session_state["scatterPlot"] = False
                st.session_state["lineChart"] = not st.session_state["lineChart"]
                st.session_state["barChart"] = False
                st.session_state["boxPlot"] = False
                # st.write("<h2>Line chart of each pair variables</h2>",unsafe_allow_html=True)
                # var = list()
                # i=0
                # for column_1 in df_num.columns:
                #     var.append(column_1)
                #     for column_2 in df_num.columns:
                #         if column_2 not in var:
                #             st.subheader(f"Line plot of {column_2} and {column_1}")
                #             st.line_chart(df_num,x=column_1,y=column_2,color=colors[random.randint(0,len(colors))-1])
                # st.subheader(f"Line plot of all variables")
                # st.line_chart(df_num)

            if st.session_state["scatterPlot"]:
                st.write("<h2>Scatter plot </h2>",unsafe_allow_html=True)
                column_1 = st.selectbox('choose the first variable for the x axis?',df_num.columns)
                if column_1 is not None: 
                    column_2 = st.selectbox('choose the first variable for the y axis?',df_num.columns)
                    st.subheader(f"Scatter plot of {column_1} and {column_2}")
                    st.scatter_chart(df_num,x=column_1,y=column_2,color=colors[random.randint(0,len(colors))-1])

            if st.session_state["lineChart"]:
                st.write("<h2>Line chart </h2>",unsafe_allow_html=True)
                column_1 = st.selectbox('choose the first variable for the x axis?',df_num.columns)
                if column_1 is not None: 
                    column_2 = st.selectbox('choose the first variable for the y axis?',df_num.columns)
                    st.subheader(f"Line chart of {column_1} and {column_2}")
                    st.line_chart(df_num,x=column_1,y=column_2,color=colors[random.randint(0,len(colors))-1])

            if st.session_state["barChart"]:
                st.header("Bar chart of the data")
                column = st.selectbox('choose the first variable you want to plot?',df_num.columns)
                st.subheader(f"Bar chart of {column}")
                st.bar_chart(df_num[column],y=column,color=colors[random.randint(0,len(colors))-1])
            
            if st.session_state["boxPlot"]:
                st.write("<h2>Box Plot of the Data</h2>",unsafe_allow_html=True)
                column = st.selectbox('choose the first variable you want to plot?',df_num.columns)
                st.subheader(f"Box plot of {column}")
                fig = px.box(df_num,y=column,points="all")
                st.plotly_chart(fig,theme="streamlit",use_container_width=True)

            st.header("Let us study your data!")
            st.subheader("At the moment, it is only applicable to linear regression.")
            target_variable = st.text_input("Enter your target variable","")

            if st.button("Send"):
                st.session_state["Send"] = not st.session_state["Send"]
                if target_variable not in df.columns:
                    error_msg = "the variable " +target_variable +" doesn't exist in the data, try again"
                    st.error(error_msg)
                else:
                    st.header(f"Computing pairwise correlation of {target_variable} and the other columns")
                    st.write(df_num.corr()[target_variable])
                    stat,stat1=False,False
                    val = {df_num.corr()[target_variable].index[i]: df_num.corr()[target_variable].values[i] for i in range(len(df_num.corr()[target_variable].values))}
                    # st.write(val)
                    i=0
                    for key,value in val.items():
                        if abs(value)<=0.2:
                            stat=True
                            st.info(f"we notice that there isn't a significant correlation between {key} and {target_variable} : {value} therefor the entry variable {key} doesn't explain the target variable")
                            val = {df_num.corr()[key].index[i]: df_num.corr()[key].values[i] for i in range(len(df_num.corr()[key].values))}
                            for key_sub,value1 in val.items():
                                if key!=key_sub and abs(value1)>=0.4:
                                    if i==0 : st.header(f"Computing pairwise correlation of {key} and the other columns")
                                    i+=0
                                    st.write(df_num.corr()[key])
                                    st.info(f"there is significant correlation between {key} and {key_sub} : {value1} which means that the two entry variables contain same information so it's perferable to remove {key}")
                    if not stat:
                        st.write("all variables explain the target value")


                    columns=[column for column in df_num.columns if column!=target_variable and not math.isnan(df_num[column].corr(df_num[target_variable]))]
                    var = target_variable + " ~ " + ' + '.join(columns)
                    st.header(f"identifying the more significant features on {target_variable} using Ordinary Least Squares(OLS)")
                    linear_reg = smf.ols(var,data=df_num)
                    res_reg = linear_reg.fit()
                    st.write(res_reg.summary())
                    st.header("Interpretation of the ols resuls")
                    insignificant_variables = []
                    for i in range(len(res_reg.pvalues)):
                        if res_reg.pvalues[i]>0.05:
                            insignificant_variables.append(res_reg.params.index.values[i])
                            st.info(f"we failed to reject the H₀ (null hypothesis) : β{i} = 0 because the p-value = {res_reg.pvalues[i]} > 0.05, the {res_reg.params.index.values[i]} coef is likely to equal 0 or the data doesn't give statistically significant evidence to conclude that β{i} ≠ 0")
                    if len(insignificant_variables)>0:
                        st.subheader(f"The results after removing {', '.join(insignificant_variables)}")
                        var = target_variable + " ~ " + ' + '.join([column for column in columns if column not in insignificant_variables])
                        linear_reg = smf.ols(var,data=df_num)
                        res_reg = linear_reg.fit()
                        st.write(res_reg.summary())

                    st.subheader("R-squared")
                    st.write(f"the Coefficient of determination r-squared: {res_reg.rsquared} shows how well the data fit the regression model")
                    st.subheader("Log-Likelihood")
                    st.write(f"the log-likelihood value of the model is a measure of how well the model predicts the observed data. A lower log-likelihood value indicates a better fit.")
                    st.write(f"in our case the log-likelihood equal {res_reg.llf}")
                    st.subheader("F-statistic and Prob (F-statistic)")
                    if(res_reg.f_pvalue<0.05): st.write(f"This is a measure of the overall significance of the regression model. It assesses whether the regression model as a whole is statistically significant in explaining the variance in the dependent variable. The F-statistic value is {res_reg.fvalue}, and the associated p-value (Prob (F-statistic)) is {res_reg.f_pvalue}, indicating that the regression model is statistically significant.")
                    else : st.write(f"This is a measure of the overall significance of the regression model. It assesses whether the regression model as a whole is statistically significant in explaining the variance in the dependent variable. The F-statistic value is {res_reg.fvalue}, and the associated p-value (Prob (F-statistic)) is {res_reg.f_pvalue}, indicating that the regression model is statistically insignificant.")
                    st.subheader("Akaike Information Criterion (AIC)")
                    st.write(f"AIC is a measure of the relative quality of a statistical model for a given set of data. It penalizes the model for including additional parameters and aims to balance model complexity with goodness of fit. A lower AIC value indicates a better-fitting model.")
                    st.write(f"AIC : {res_reg.aic}")
                    st.subheader("Bayesian Information Criterion (BIC)")
                    st.write(f"Similar to AIC, BIC : {res_reg.bic} is also a measure of the relative quality of a statistical model. It also penalizes the model for including additional parameters but uses a different penalty term than AIC. As with AIC, a lower BIC value indicates a better-fitting model.")
                    st.header("Linear regression equation")
                    latext = r'''
                        ## 
                        ###  
                        $$ 
                        Y = \Beta_0 + X^t\Beta + \epsilon
                        $$ 
                        '''
                    st.write(latext)
                    st.subheader("The estimated Betas values")
                    for i in range(len(res_reg.params)):
                        latext = r'''
                        ## 
                        ###  
                        $$ 
                        \Beta_{i} = {var}
                        $$ 
                        '''.replace("var",('%.6f' % res_reg.params[i])).replace("i",str(i))
                        st.write(latext)

                    res = res_reg.summary()
                    def interpretate_res(res):
                        model_engine = "gpt-3.5-turbo-instruct"
                        prompt = (
                            f"Interpretate the results of the fitted data\n"
                            f"{res}\n"
                            f"Interpretation:"
                        )
                        response = openai.Completion.create(
                            engine=model_engine,
                            prompt=prompt,
                            temperature=0,
                            max_tokens=300,
                            top_p=1.0,
                            frequency_penalty=0.0,
                            presence_penalty=0.0,
                            stop=["#", ";"]
                        )
                        return response.choices[0].text.strip()
                    st.subheader("Summary")
                    st.write(interpretate_res(res))

                    st.header("Predict the target variable based on input features")
                    X = df_num[[column for column in df_num.columns if column not in insignificant_variables and column!=target_variable]]
                    Y = df_num[target_variable]

                    from sklearn.model_selection import train_test_split
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random.randint(0,10000))

                    from sklearn.linear_model import LinearRegression
                    from sklearn import metrics
                    import numpy as np
                    lm = LinearRegression()
                    lm.fit(X_train,Y_train)
                    predictions = lm.predict(X_test)
                    df_pred = pd.DataFrame({'predictions':predictions, 'y test':Y_test})
                    st.scatter_chart(data=df_pred,x='predictions',y='y test',color=colors[random.randint(0,len(colors))-1])
                    latex = r'''
                        ## Evaluation criteria for the regression model
                        There are three evaluation criteria for regression problems:

                        **Mean Absolute Error** (MAE) :

                        $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$

                        **Mean Squared Error** (MSE) :

                        $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$

                        **Root Mean Squared Error** (RMSE) :

                        $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$

                        Comparaison de ces trois critères :

                        - **MAE** is the easiest to undesrtand because it's simply Mean Absolute Error.
                        - **MSE** is more popular than MAE because MSE is more affected by the highest errors, which tends to be useful in practice.
                        - **RMSE** is even more popular than MSE, because RMSE is interpretable by comparing it to the 'Y' values as they are in the same units.

                        All these criteria are **loss functions** that we aim to minimize.
                    '''
                    st.write(latex)
                    st.write(f'MAE: {metrics.mean_absolute_error(Y_test, predictions)}')
                    st.write(f'MSE: {metrics.mean_squared_error(Y_test, predictions)}')
                    st.write(f'RMSE: {np.sqrt(metrics.mean_squared_error(Y_test, predictions))}')

        else:
            error_msg = "files of type "+ uploaded_file.type +" are not supported"
            st.error(error_msg)

    else:
        st.error('Load your data first !!')
else:
    st.info('Load your data')

# if st.session_state["Send"]:

    
st.write("""
        <script src=".\iframeResizer.contentWindow.js"></script>
                  
            """,unsafe_allow_html=True)

# st.write(
#     f"""
#     ## Session state:
#     {st.session_state["Submit"]=}

#     {st.session_state["Send"]=}

#     """
# )