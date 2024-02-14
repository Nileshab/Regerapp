import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.express as px
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
import statsmodels.formula.api as smf
import statsmodels.tools.tools as smt
import plotly.graph_objects as go
from scipy.stats import shapiro
import seaborn as sns
from statsmodels.stats.stattools import durbin_watson
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.style.use('dark_background')
import matplotlib
from streamlit_option_menu import option_menu
try : 
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None :
        data = pd.read_csv(uploaded_file)
        try :

            numeric_cols = data.select_dtypes(include=['int', 'float']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            boolean_cols = data.select_dtypes(include=['bool']).columns.tolist()
        except : 
           st.write("We Need More Resources , Try with Cleaned Data ")
    if 'selected_option' not in st.session_state:
     st.session_state.selected_option = "Info"  # Default option
     



    st.sidebar.title('Exploratory Data Analysis')
    option = st.sidebar.selectbox('Select Option', ('Info', 'Interactive Plots'))
    st.session_state.option = option

    if option == "Info":
        st.subheader("Glance To Data ")
        st.dataframe(data)
        st.divider()
        cola , colb = st.columns(2)
        with cola:
         st.subheader("Summary Statistics")
         st.write(data.describe())
        with colb:
         st.subheader("Column Config")
         st.write("Numerics",numeric_cols ,"Categories", categorical_cols  )
         st.divider()

    if option == "Interactive Plots":
        st.subheader("Choose Options for Scatter Plot")
        col1 , col2 = st.columns(2)
        with col1 :
         option_1 = st.selectbox("Select Column 1", numeric_cols)
        with col2 :
         option_2 = st.selectbox("Select Column 2", numeric_cols)
        # if st.button("Create Chart"):
         fig = px.scatter(data,x =data[option_1] ,y =data[option_2])
        st.plotly_chart(fig)
        st.divider()


        st.subheader("Choose Options for Bar Plot")
        col3 , col4 = st.columns(2)
        with col3 :
         Bar_num = st.selectbox("Select Numeric Column For Bar Chart", numeric_cols)
        with col4 :
         Bar_cat = st.selectbox("Select Categorical Column For Bar Chart", categorical_cols)
        Bar_chart = data.groupby(Bar_cat)[Bar_num].sum()
        if st.button("Create Bar Chart"):
         st.bar_chart(Bar_chart)

        values = data[Bar_num]
        names = data[Bar_cat]
        fig_pie = px.pie(data,values = values, names = names, title='Pie Chart')
        if st.button("Create Pie Chart"):
         st.write(fig_pie)
        st.divider()

        st.subheader("Histogram")
        Hist_num = st.selectbox("Select Column For Histogram", numeric_cols)
        fig2 = px.histogram(data_frame=data, x=data[Hist_num])
        st.write(fig2)
        # fig_ = data[Hist_num].plot(kind = "kde")
        fig_ = px.density_contour(x=data[Hist_num])
        if st.button("Show Kde :"):
         st.write(fig_)
        st.divider()


        st.sidebar.title('Model Fitting (OLS)')
    option_ = st.sidebar.selectbox('Select Option', ( 'Linearity Plot' , 'Homoscedasticity','Normality','Multicollinearity','Auto- Correlation'))
    st.session_state.option_ = option

    df_num_col = pd.DataFrame(data[numeric_cols])
    # global target_variable ,target_col ,independent_variables,independent_col ,model ,result ,residuals
    target_variable = st.selectbox("Choose Target Variable (Y)", numeric_cols)
    target_col = df_num_col[target_variable]
    independent_variables = [col for col in df_num_col.columns if col != target_variable]
    independent_col = df_num_col[independent_variables]
        
        # Fit the model
    X = sm.add_constant(independent_col)
    model = sm.OLS(target_col, sm.add_constant(independent_col))
    result = model.fit()
    residuals = result.resid
    predicted_values = result.predict(X)
    residual_df = pd.DataFrame({ "Indexx":residuals.index,
                                "Residual":residuals 
    })
        
        # Display model summary
    st.subheader("Model Summary")
    st.write(result.summary())
    st.divider()

    if option_ == "Linearity Plot":
    #  option04 = st.sidebar.selectbox("Please Choose ",("Graph","Theory"))
     option04 = option_menu(menu_title=None,
        options=["Graph","Theory"],
                 orientation="horizontal")
     try:# Plot linearity using Plotly Express
        if option04 == "Graph":
         predvstarget = px.scatter(data, x= predicted_values , y= target_col, trendline='ols', trendline_color_override='red')
         predvstarget.update_layout(
         title=" Scatter Plot of Predicted vs. Actual Values",
         xaxis_title="Predicted Values",
         yaxis_title="Actual Values"
                    )

         st.plotly_chart(predvstarget)
         st.divider()
         residvstarget = px.scatter(data, x = predicted_values , y= residuals
         , trendline='ols', trendline_color_override='red')
         residvstarget.update_layout(
         title=" Scatter Plot of Predicted vs. Residual",
         xaxis_title="Predicted Values",
         yaxis_title="Residual"
                    )
         st.plotly_chart(residvstarget)
         st.divider()
        if option04 == "Theory":
           st.markdown("""
                     ### Linearity 
                    * Linearity is one of the Most important Assumption of Linear Regression
                    * When Ploting Data if Linearity is Missing this shows using Linear fit Line is Not Appropriate in this Situation.
                       i.e we need Transformation 
                        Boxcox transformation : Does not Guarantees to provide Linearity but can provide help to achieve closer to our Goal
                        It Uses the Maximum Likelihood Estimate to find out which λ satisfies the lowest Error, 
                        * λ value here is fix , ranges generally between -2 < λ < 2
                        *             (y^λ -1)/λ : λ != 0
                                        log(y)   : λ = 0 
                        * for λ = 0       ; {log Transformation}
                         *    λ = 1       ; {No Transformation}
                         *    λ = 0.5     ; {Root Transformation}
                         *    λ = 0.75    ; {Cube Transformation}
                         *    λ = 2       ; {Square Transformation}
                         *    λ < 1       ; {Reciprocal Transformation}
                            
                       """)
     except NameError:
            st.error("Please run 'Model Building' first to define the target variable.")


    if option_ == "Normality":
    #  option03 = st.sidebar.selectbox("Please Choose ",("Graph","Theory"))
     option03 = option_menu(menu_title=None,
        options=["Graph","Theory"],
                 orientation="horizontal")
     try:# Plot linearity using Plotly Express
        if option03 == "Graph":
         sm.qqplot(residual_df["Residual"])
         st.pyplot();
    # To plot Graph
         sns.distplot(residual_df["Residual"])
         st.pyplot();
        
        if option03 == "Theory":   
           
            statistic, p_value = shapiro(data[numeric_cols])
            st.markdown("""
                        # Shapiro Test is used to Test , Test of Association using Chi Square Test 
                        # Does Data is Normally Distributed  
                        # Value for Statistics""") 
            st.write("Statistic",statistic)
                        
                        # Probability Value 
            st.write("Probability ",p_value)
                        
            st.markdown("""
                        H0 : The errors are normally distributed.
                        H1 : Not H0

                        Reject H0 iff  Probability Value < 0.05 
                        To overcome this Problem :
                        Boxcox transformation : Does not Guarantees to provide Linearity but can provide help to achieve closer to our Goal
                        It Uses the Maximum Likelihood Estimate to find out which λ satisfies the lowest Error, 
                        * λ value here is fix , ranges generally between -2 < λ < 2
                        *             (y^λ -1)/λ : λ != 0
                                        log(y)   : λ = 0 
                        * for λ = 0       ; {log Transformation}
                         *    λ = 1       ; {No Transformation}
                         *    λ = 0.5     ; {Root Transformation}
                         *    λ = 0.75    ; {Cube Transformation}
                         *    λ = 2       ; {Square Transformation}
                         *    λ < 1       ; {Reciprocal Transformation}
                            
                        """)



     except NameError:
            st.error("Sometihng Strange Happened")


    if option_ == "Homoscedasticity":
    #  option02 = st.sidebar.selectbox("Please Choose ",("Graph","Theory"))
     option02 = option_menu(menu_title=None,
        options=["Graph","Theory"],
                 orientation="horizontal")
     try:# Plot linearity using Plotly Express
        if option02 == "Graph":
         resdiduals = result.resid
         fg = px.scatter(target_col,resdiduals)
         px.scatter(target_col,[0]*target_col)
         st.plotly_chart(fg)

        if option02 == "Theory":
            lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(result.resid, X)
            st.markdown("### Bruesch Pagan Test")
            st.markdown(""" BP Test Uses Chisquare to check Assumption , Variance should be constant along predicted Values 
                        to overcome this Problem we use Boxcox transfomation criteria  
                        Boxcox transformation : Does not Guarantees to provide Linearity but can provide help to achieve closer to our Goal
                        It Uses the Maximum Likelihood Estimate to find out which λ satisfies the lowest Error, 
                        * λ value here is fix , ranges generally between -2 < λ < 2
                        *             (y^λ -1)/λ : λ != 0
                                        log(y)   : λ = 0 
                        * for λ = 0       ; {log Transformation}
                         *    λ = 1       ; {No Transformation}
                         *    λ = 0.5     ; {Root Transformation}
                         *    λ = 0.75    ; {Cube Transformation}
                         *    λ = 2       ; {Square Transformation}
                         *    λ < 1       ; {Reciprocal Transformation}
                        

                        Also we use Generalised Least Square Method 
                            
                        """)
            st.divider()
            st.write("Null Hypothesis (Ho) : Independent variables coefficient are Equal to Zero(Constant Variance)")
            st.write("ALternate Hypothesis (H1) : Atleast One coefficient is Not Equal to Zero")
            st.divider()
            st.write("Lagrange Multiplier :",round(lm,2),"    ","P-value : ",round(lm_p_value,4))
            st.write("Reject Ho  :  Iff P-value < 0.05 ")
            st.divider()

     except NameError:
            st.error("Something Strange Happened")

    if option_ == "Multicollinearity":
    #  option01 = st.sidebar.selectbox("Please Choose ",("Graph","Theory"))
     option01 = option_menu(menu_title=None,
        options=["Graph","Theory"],
                 orientation="horizontal")
     try:# Plot linearity using Plotly Express
        if option01 == "Graph":
         correlation_matrix = data[numeric_cols].corr()
         mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
         f, ax = plt.subplots(figsize=(4, 3))
         cmap = sns.diverging_palette(230, 20, n=256, as_cmap=True)
         sns.heatmap(correlation_matrix, 
                mask=mask, 
                cmap=cmap, 
                vmax=1, 
                vmin = -.25,
                center=0,
                square=True, 
                linewidths=.5,
                annot = True,
                fmt='.2f', 
                annot_kws={'size': 10},
                cbar_kws={"shrink": .75})
         st.pyplot()
        if option01 == "Theory":
          from statsmodels.stats.outliers_influence import variance_inflation_factor
        st.markdown("""
                   #  For VIF greater than 10 we interpret problem of Multicolinearity
                    """)
        st.dataframe({X.columns[i] : variance_inflation_factor(X.values, i) for i in range(X.shape[1])})
        st.markdown("""
                    Multicollinearity can be Challenging Problem to Interpret the Result  : 
                    Multicollinearity is the Problem which is Related to independent Variables , if this Problem Occurs then we will not able to clarify the individual effect 
                    of varible in reusltant we will get combined effect of both collinear variables .
                    to Deal with such problem we use PCA (Principal Component Analysis) , Fcator Analysis , Ridge regression , Lasso Regression 
                    Weighted Least Square can be used to deal with it as it normalises variance by dividing it by itself
                    """)
     except NameError:
            st.error("Sometihng Strange Happened")


    if option_ == "Auto- Correlation":
    #  option00 = st.sidebar.selectbox("Please Choose ",("Graph","Theory"))
     option00 = option_menu(menu_title=None,
        options=["Graph","Theory"],
                 orientation="horizontal")
     try:# Plot linearity using Plotly Express
        residual_df = pd.DataFrame({ "Indexx":residuals.index,
                                "Residual":residuals 
                                })
        if option00 == "Graph":
         st.bar_chart(residual_df["Residual"])

        if   option00 == "Theory":
         durbin_watson_statistic = durbin_watson(residuals)
         st.markdown(""" # We Assume that the error Terms are Independent , and if this Violets then we say theres Presence of Autocorrelation.
                     This Can Lead to Biased Coefficients , and in such cases using Arima Time Series Model can help us 
                     """)
         st.write("durbin_watson_statistic",durbin_watson_statistic)
         st.markdown(""" Durbin Watson Statistic Helps us to Check Autocorrelation presence in data.
                                          D = 2(1-rho) where rho is correlation between errors...  D is Distance
                     
                     interpretation : value closer to 2 represents no autocorrelation , else shows presence.
                     so , range becomes 0 < rho < 4

                    """)
         
           
     except NameError:
            st.error("Sometihng Strange Happened")
except NameError :
   if uploaded_file == None :
    img = Image.open("C://Users//niles//Downloads//Streamlit_project//sir.jpeg")
    image_path = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Sir_Francis_Galton_by_Gustav_Graef.jpg/1200px-Sir_Francis_Galton_by_Gustav_Graef.jpg"
   
    st.image(image_path,caption= " Sir Francis Galton",width = 250)


    st.markdown(""" 
                ###
     Sir Francis Galton, 
                a pioneer in statistics, introduced the concept of correlation and developed the method of least squares for regression analysis. 
                His work laid the foundation for linear regression,
                enabling the estimation of relationships between variables by minimizing the sum of squared differences. 
                Galton's contributions, including the notion of Galton's Line, continue to shape modern statistical techniques.
     """)

    st.markdown("""
            ### Prerequisites of using this App.
            *  Uploaded File Should be Preprocessed. 
            *  Should Not Contain Null Values. 
            *  Using Linear Regression Model Fits all the Numerical Columns except Target Column. 

            ### Brief : 
            *  App Totally Focuses on Teaching the Basics of Linear Regression.
            *  App Provides Interactive Plots , Understanding of Theory Concept.
            *  Naming Tests and their Interpretation. 
            *  Some Recommendations Based on Specific Test.


            ### To whom This Belongs : 
            *  Any Personality who is Beginner and Wants to Play with Hot Topic Regression.
            *  To Get General Unnderstanding of Data.
            *  To make Regression Easier , Affordable to Learn.
            
            
            """)

# tabs = st.sidebar.radio("Navigation", ["Tab 1", "Tab 2", "Tab 3"])
# if tabs == "Tab 1":
#     st.write("You are in Tab 1")
# elif tabs == "Tab 2":
#     st.write("You are in Tab 2")
# elif tabs == "Tab 3":
#     st.write("You are in Tab 3")
   