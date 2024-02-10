import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.config

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
                .e115fcil1 img{scale:0.6;}
        </style>
        <script>
        /*

        MIT License

        Copyright (c) 2019 Jacob Filipp

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        */


        // Add this script into the page that will appear <b>inside an iFrame</b>
        // This code monitors the page for changes in size. When change is detected, it sends send the latest size to the parent page using postMessage


        // determine height of content on this page
        function getMyHeight()
        {
            //https://stackoverflow.com/a/11864824
            return Math.max( document.body.scrollHeight, document.documentElement.scrollHeight)
        }


        // send the latest page dimensions to the parent page on which this iframe is embedded
        function sendDimensionsToParent()
        {
            var iframeDimensions_New = {
                        'width': window.innerWidth, //supported from IE9 onwards
                        'height': getMyHeight()
                    };
                    
            if( (iframeDimensions_New.width != iframeDimensions_Old.width) || (iframeDimensions_New.height != iframeDimensions_Old.height) )  // if old width is not equal new width, or old height is not equal new height, then...
            {
                
                window.parent.postMessage(iframeDimensions_New, "*");
                iframeDimensions_Old = iframeDimensions_New;
                
            }
                
        }


        // on load - send the page dimensions. (we do this on load because then all images have loaded...)
        window.addEventListener( 'load', function(){
            
                
            iframeDimensions_Old = {
                                'width': window.innerWidth, //supported from IE9 onwards
                                'height': getMyHeight()
                            };

            window.parent.postMessage(iframeDimensions_Old, "*"); //send our dimensions once, initially - so the iFrame is initialized to the correct size


            if( window.MutationObserver ) // if mutationobserver is supported by this browser
            {
                //https://developer.mozilla.org/en-US/docs/Web/API/MutationObserver

                var observer = new MutationObserver(sendDimensionsToParent);
                config = {
                    attributes: true,
                    attributeOldValue: false,
                    characterData: true,
                    characterDataOldValue: false,
                    childList: true,
                    subtree: true
                };

                observer.observe(document.body, config);
                
            }
            else // if mutationobserver is NOT supported
            {
                //check for changes on a timed interval, every 1/3 of a second
                window.setInterval(sendDimensionsToParent, 300);
            }


        });   // end of window.onload

        </script>
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
