import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import scipy.stats as stats
import seaborn as sns
import plotly.figure_factory as ff

def market_basket_analysis(df):
    # Convert 'Sub-Category' column into dummies
    transaction_df = pd.get_dummies(df['Sub-Category'])
    transaction_df = pd.concat([df['Order ID'], transaction_df], axis=1)
    transaction_df = transaction_df.groupby('Order ID').sum()
    transaction_df = transaction_df.applymap(lambda x: 1 if x > 0 else 0)

    # Market Basket Analysis
    frequent_itemsets = apriori(transaction_df, min_support=0.001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)

    # Calculate standard error of confidence
    std_error_confidence = stats.sem(rules['confidence'])
    # Calculate critical value
    critical_value = stats.t.ppf(0.975, len(rules) - 1)
    # Calculate margin of error
    margin_of_error = critical_value * std_error_confidence
    # Calculate confidence interval
    confidence_interval_lower = rules['confidence'] - margin_of_error
    confidence_interval_upper = rules['confidence'] + margin_of_error

    # Add confidence interval to the DataFrame
    rules['confidence_interval_lower'] = confidence_interval_lower
    rules['confidence_interval_upper'] = confidence_interval_upper

    return rules, transaction_df

def main():
    st.set_page_config(page_title="Minger Sales Dashboard", page_icon=":bar_chart:", layout="wide")
    st.title(":bar_chart: Minger Sales Dashboard")
    st.markdown('<style>div.block-container{padding-top:1.5%;}</style>', unsafe_allow_html=True)

    # File uploading or browsing the available file
    fl = st.file_uploader(":file_folder: Upload a file", type=(["xlsx", "xls"]))
    if fl is not None:
        filename = fl.name
        st.write(filename)
        df = pd.read_excel(filename)
    else:
        df = pd.read_excel("Global Superstore lite.xlsx")

    col1, col2 = st.columns((2))
    df["Order Date"] = pd.to_datetime(df["Order Date"])

    # Minimum and maximum dates
    startDate = pd.to_datetime(df["Order Date"].min())
    endDate = pd.to_datetime(df["Order Date"].max())

    with col1:
        date1 = pd.to_datetime(st.date_input("Start Date", startDate))

    with col2:
        date2 = pd.to_datetime(st.date_input("End Date", endDate))

    df = df[(df["Order Date"] >= date1) & (df["Order Date"] <= date2)].copy()

    # Add a 'month' column to the DataFrame
    df['month'] = df['Order Date'].dt.month

    # filters
    st.sidebar.header("Choose your filters :")
    # Filter for region
    region = st.sidebar.multiselect("Pick your Region", df["Region"].unique())
    if not region:
        df2 = df.copy()
    else:
        df2 = df[df["Region"].isin(region)]

    # Filter for state based on region
    state = st.sidebar.multiselect("Pick your State", df2["State"].unique())
    if not state:
        df3 = df2.copy()
    else:
        df3 = df2[df2["State"].isin(state)]

    # Filter for city, based on state and region
    city = st.sidebar.multiselect("Pick your City", df3["City"].unique())

    if not region and not state and not city:
        filtered_df = df
    elif not state and not city:
        filtered_df = df[df["Region"].isin(region)]
    elif not region and not city:
        filtered_df = df[df["State"].isin(state)]
    elif state and city:
        filtered_df = df3[df["State"].isin(state) & df3["City"].isin(city)]
    elif region and city:
        filtered_df = df3[df["Region"].isin(region) & df3["City"].isin(city)]
    elif region and state:
        filtered_df = df3[df["Region"].isin(region) & df3["State"].isin(state)]
    elif city:
        filtered_df = df3[df3["City"].isin(city)]
    else:
        filtered_df = df3[df3["Region"].isin(region) & df3["State"].isin(state) & df3["City"].isin(city)]

    category_df = filtered_df.groupby(by=["Category"], as_index=False)["Sales"].sum()

    # Column chart
    with col1:
        st.subheader(":shopping_trolley: Category wise Sales")
        fig = px.bar(category_df, x="Category", y="Sales", text=['${:,.2f}'.format(x) for x in category_df["Sales"]],
                     template="seaborn")
        st.plotly_chart(fig, use_container_width=True, height=200)

    # Pie chart
    with col2:
        st.subheader(":earth_asia: Region wise Sales")
        fig = px.pie(filtered_df, values="Sales", names="Region", hole=0.5)
        fig.update_traces(text=filtered_df["Region"], textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    #Downloading Category wise and Region wise data
    cl1,cl2 = st.columns(2)
    with cl1:
        with st.expander("Category Data"):
            st.write(category_df.style.background_gradient(cmap="Blues"))
            excel = category_df.to_csv(index = False).encode('utf-8')
            st.download_button("Dowload Data", data = excel, file_name = "Category.csv", mime = "text/csv",
                               help = 'Click here to download the data as an excel file')

    with cl2:
        with st.expander("Region Data"):
            region = filtered_df.groupby(by = "Region", as_index = False)["Sales"].sum()
            st.write(region.style.background_gradient(cmap="Blues"))
            csv = region.to_csv(index = False).encode('utf-8')
            st.download_button("Dowload Data", data = csv, file_name = "Region.csv", mime = "text/csv",
                               help = 'Click here to download the data as a CSV file')

    #Time Series Analysis
    filtered_df["month/year"] = filtered_df["Order Date"].dt.to_period("M")
    st.subheader(':chart_with_downwards_trend: Time Series Analysis')

    linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month/year"].dt.strftime("%Y : %b"))["Sales"].sum()).reset_index()
    fig2 = px.line(linechart, x = "month/year", y = "Sales", labels = {"Sales":"Amount"}, height = 500, width = 100, template = "gridon")
    st.plotly_chart(fig2,use_container_width=True)

    #Downloading Time series Analysis data
    with st.expander(" Time Series Analysis Data"):
            st.write(linechart.T.style.background_gradient(cmap="Greens"))
            csv = linechart.to_csv(index = False).encode('utf-8')
            st.download_button("Dowload Data", data = csv, file_name = "Time_Series_Analysis.csv", mime = "text/csv",
                               help = 'Click here to download the data as a CSV file')

    # Segment wise and Category wise Sales charts
    chart1, chart2 = st.columns((2))
    with chart1 :
        st.subheader(':chart: Segment wise Sales')
        fig = px.pie(filtered_df,values = "Sales", names = "Segment", template = "plotly_dark")
        fig.update_traces(text = filtered_df["Segment"], textposition = "inside")
        st.plotly_chart(fig,use_container_width=True)

    with chart2:
        st.subheader(':chart: Category wise Sales')
        fig = px.pie(filtered_df,values = "Sales", names = "Category", template = "gridon")
        fig.update_traces(text = filtered_df["Category"], textposition = "inside")
        st.plotly_chart(fig,use_container_width=True)
    
    # Summary tables
    st.subheader(":calendar: Month wise Sub-Category Sales Summary")
    st.markdown("Summary Table")
    df_sample = df[0:5][["Region", "State", "City", "Category", "Sales", "Profit", "Quantity"]]
    fig = ff.create_table(df_sample, colorscale="Cividis")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Month wise sub-category Table")
    sub_category_Year = pd.pivot_table(data=filtered_df, values="Sales", index=["Sub-Category"], columns="month")
    st.write(sub_category_Year.style.background_gradient(cmap="Oranges"))

    # Scatter plot
    data1 = px.scatter(filtered_df, x="Sales", y="Profit", size="Quantity")
    data1["layout"].update(title="Relationship between Sales and Profits using Scatter Plot",
                           titlefont=dict(size=20),
                           xaxis=dict(title="Sales", titlefont=dict(size=19)),
                           yaxis=dict(title="Profit", titlefont=dict(size=19)))
    st.plotly_chart(data1, use_container_width=True)

    # Market Basket Analysis
    st.subheader(':clipboard: Market Basket Analysis')
    rules, transaction_df = market_basket_analysis(filtered_df)

    # Display association rules
    st.write("Association Rules:")
    
    st.dataframe(rules)

    # Confidence, lift, and support charts
    st.subheader(':ledger: Association Rules Metrics')
    with st.expander("Bar Graph of Confidence Values"):
        st.subheader("Bar Graph of Confidence Values")
        confidence = rules['confidence']
        plt.figure(figsize=(8, 3))
        plt.bar(range(len(confidence)), confidence, color='orange')
        plt.xlabel('Association Rules', fontsize=8)
        plt.ylabel('Confidence Values', fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        st.pyplot()

    with st.expander("Bar Graph of Lift Values"):
        st.subheader("Bar Graph of Lift Values")
        lift = rules['lift']
        plt.figure(figsize=(8, 3))
        plt.bar(range(len(lift)), lift, color='blue')
        plt.axhline(y=1, color='black', linestyle='--')
        plt.xlabel('Association Rules', fontsize=8)
        plt.ylabel('Lift Values', fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        st.pyplot()

    with st.expander("Bar Graph of Support Values"):
        st.subheader("Bar Graph of Support Values")
        support = rules['support']
        plt.figure(figsize=(8, 3))
        plt.bar(range(len(support)), support, color='green')
        plt.xlabel('Association Rules', fontsize=8)
        plt.ylabel('Support Values', fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        st.pyplot()


    with st.expander("Co-occurrence Matrix of Sub-Categories"):
        st.subheader('Co-occurrence Matrix of Sub-Categories (Excluding Same Product Combinations)')
        # Co-occurrence Matrix of Sub-Categories (Excluding Same Product Combinations) (Heatmap)
        co_occurrence_matrix = transaction_df.T.dot(transaction_df)
        np.fill_diagonal(co_occurrence_matrix.values, 0)
        plt.figure(figsize=(6, 4))
        heatmap = sns.heatmap(co_occurrence_matrix, annot=True, cmap="YlGnBu", fmt="d", linewidths=1, annot_kws={"fontsize": 5})
        heatmap.set_xlabel('Sub-Category', fontsize=6)
        heatmap.set_ylabel('Sub-Category', fontsize=6)
        heatmap.tick_params(axis='both', which='major', labelsize=5)
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6)  # Set font size for color bar
        st.pyplot()


    # Displaying all the data
    st.subheader(":shopping_trolley: Entire Sales Table")
    with st.expander("View Entire data"):
        st.write(filtered_df.iloc[:, 1:20:2].style.background_gradient(cmap="Greens"))

    # Downloading the entire data
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Dowload Data", data=csv, file_name="Minge_SalesData_Set.csv", mime="text/csv",
                       help='Click here to download the data as a CSV file')


if __name__ == '__main__':
    main()
