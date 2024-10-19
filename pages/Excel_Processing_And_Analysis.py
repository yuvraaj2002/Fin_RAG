import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from ydata_profiling import ProfileReport

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
                .top-margin{
                    margin-top: 4rem;
                    margin-bottom:2rem;
                }
                .block-button{
                    padding: 10px; 
                    width: 100%;
                    background-color: #c4fcce;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def analyze_file(dataframe, output_path):
    """
    Analyzes the given dataframe and generates a report using pandas profiling.
    
    Parameters:
    - dataframe (pd.DataFrame): The pandas dataframe to be analyzed.
    - output_path (str): The directory path where the report will be saved.
    
    Returns:
    - str: The file path of the generated report.
    """
    # Performing automatic EDA 
    profile = ProfileReport(dataframe, title="Pandas Profiling Report")
    final_path = os.path.join(output_path,"report.html")
    profile.to_file(final_path)
    
    return final_path

def excel_processing_and_analysis():
    """
    This function enables the user to upload an Excel file and perform various data processing and analysis tasks. 
    It provides an interface to select a sheet from the uploaded file, display a preview of the sheet, and initiate an analysis process.
    The analysis process generates a report using pandas profiling, which is then made available for download.
    
    This function is designed to facilitate data analysis and visualization for Excel files, particularly for financial data such as sales, expenses, COGS, and balance sheet data.
    It aims to provide insights into the data, helping users make informed decisions and understand trends within their data.
    """

    upload_tab, display_tab = st.columns(spec=(3,1), gap="large")
    with upload_tab:
        st.title("Excel Processing and Analysis")
        st.write("This page allows you to upload an Excel file and perform various data processing and analysis tasks. With power of large language models and statistical techniques, this module will help you to perform a complete analysis of your Excel file, providing insights into your data and helping you make informed decisions. You will also get interactive graphs to visualize your data and trends.")

        # File upload
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        with display_tab:
            st.write("")
            st.write("")
            st.write("")
            st.success(f"File {uploaded_file.name} uploaded successfully!")

    if uploaded_file is not None:
        file_path = uploaded_file
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names

        # Display sheet names as markdown
        st.markdown("#### Available Sheets📂")
        selected_sheet = st.selectbox("Select a sheet to analyze", sheet_names)
      
        start_analysis = False
        if selected_sheet:
            df = pd.read_excel(file_path, sheet_name=selected_sheet)
            sheet_display, description = st.columns(spec=(1,1), gap="small")
            with sheet_display:
                st.dataframe(df.head(5))
            with description:
                if selected_sheet == 'Sales Data':
                    st.write("The Sales Data sheet is crucial for understanding the revenue streams of the business. It helps in analyzing total revenue, identifying key customers, understanding sales trends over different time periods, and evaluating the performance of different products. Insights from this sheet can be used to determine which products are high-performing, which customer segments are most profitable, and to monitor revenue growth. It also provides foundational data for the Profit & Loss (PnL) analysis.")
                    if st.button("Start Analysis", key="start_analysis_sales", use_container_width=True):
                        start_analysis = True
                        report_path = analyze_file(df,"artifacts")
                        
                elif selected_sheet == 'Expenses':
                    st.write("The Expenses sheet is essential for monitoring and controlling the costs of the business. Tracking expenses helps in understanding where money is being spent and identifying opportunities to reduce costs or improve efficiency. This sheet also provides input for calculating net profit, as it details all costs that need to be subtracted from gross revenue. Analyzing expenses also supports budgeting and helps identify areas that might need cost-cutting measures.")
                    if st.button("Start Analysis", key="start_analysis_expenses", use_container_width=True):
                        start_analysis = True
                        report_path = analyze_file(df,"artifacts")
                        
                elif selected_sheet == 'COGS':
                    st.write("The COGS sheet is vital for calculating the gross profit of the business. By understanding how much it costs to produce each product, a company can determine profitability at a product level. It helps in identifying the cost efficiency of production processes and in setting appropriate product pricing. Gross profit, calculated as revenue minus COGS, is a key metric to evaluate whether the business is covering its direct costs and how much margin is available to cover operating expenses.")
                    if st.button("Start Analysis", key="start_analysis_cogs", use_container_width=True):
                        start_analysis = True
                        report_path = analyze_file(df,"artifacts")
                        
                elif selected_sheet == 'Balance Sheet Data':
                    st.write("The Balance Sheet Data sheet is used to evaluate the overall financial health of the company. It shows what the company owns (assets), what it owes (liabilities), and the net worth (equity). This sheet is crucial for understanding liquidity, solvency, and the financial stability of the business. It helps in assessing whether the company has enough assets to cover its liabilities, understanding the capital structure, and calculating important financial ratios such as the current ratio and debt-to-equity ratio. The balance sheet data also helps stakeholders understand the long-term sustainability of the company.")
                    if st.button("Start Analysis", key="start_analysis_balance", use_container_width=True):
                        start_analysis = True
                        report_path = analyze_file(df,"artifacts")
                else:
                    if st.button("Start Analysis", key="start_analysis_sales", use_container_width=True):
                        start_analysis = True
                        report_path = analyze_file(df,"artifacts")


            if start_analysis:
                st.success("Analysis Completed Successfully!")
                st.download_button(
                    label="Download Analysis Report",
                    data=open(report_path, 'rb').read(),
                    file_name="analysis_report.html",
                    mime='text/html'
                )
        

excel_processing_and_analysis()