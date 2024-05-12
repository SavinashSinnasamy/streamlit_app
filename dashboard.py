import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
# Title of the app
st.set_page_config(page_title="Minger Sales Dashboard", page_icon=":bar_chart:",layout="wide")
st.title(" :bar_chart: Minger Sales Dashboard")
st.markdown('<style>div.block-container{padding-top:1.5%;}</style>',unsafe_allow_html=True)

# File uploading or browsing the available file
fl = st.file_uploader(":file_folder: Upload a file",type=(["xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_excel(filename)
else:
   # os.chdir("C:\Users\savinash\Desktop\Lubnacw")
    df = pd.read_excel("Global Superstore lite.xlsx")

col1,col2 = st.columns((2))
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Minimum and maximum dates
startDate = pd.to_datetime(df["Order Date"].min())
endDate = pd.to_datetime(df["Order Date"].max())

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["Order Date"] >= date1) & (df["Order Date"] <= date2)].copy()

# filters
st.sidebar.header("Choose your filters :")
# Filter for region
region = st.sidebar.multiselect("Pick your Region",df ["Region"].unique())
if not region:
    df2 = df.copy()
else:
    df2 = df[df["Region"].isin(region)]

# Filter for state based on region
state = st.sidebar.multiselect("Pick your State",df2 ["State"].unique())
if not state:
    df3 = df2.copy()
else:
    df3 = df2[df2["State"].isin(state)]

# Filter for city, based on state and region
city = st.sidebar.multiselect("Pick your City",df3 ["City"].unique())

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

category_df = filtered_df.groupby(by = ["Category"], as_index = False)["Sales"].sum()

#Column chart
with col1:
    st.subheader("Catergory wise Sales")
    fig = px.bar(category_df,x = "Category", y = "Sales", text = ['${:,.2f}'.format(x) for x in category_df["Sales"]],
                 template = "seaborn")
    st.plotly_chart(fig,use_container_width=True,height=200)

#Pie chart 
with col2:
    st.subheader("Region wise Sales")
    fig = px.pie(filtered_df,values = "Sales", names = "Region", hole= 0.5)
    fig.update_traces(text= filtered_df["Region"], textposition = "outside")
    st.plotly_chart(fig,use_container_width=True)

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
        st.write(region.style.background_gradient(cmap="Oranges"))
        csv = region.to_csv(index = False).encode('utf-8')
        st.download_button("Dowload Data", data = csv, file_name = "Region.csv", mime = "text/csv",
                           help = 'Click here to download the data as a CSV file')

#Time Series Analysis
filtered_df["month/year"] = filtered_df["Order Date"].dt.to_period("M")
st.subheader('Time Series Analysis')

linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month/year"].dt.strftime("%Y : %b"))["Sales"].sum()).reset_index()
fig2 = px.line(linechart, x = "month/year", y = "Sales", labels = {"Sales":"Amount"}, height = 500, width = 100, template = "gridon")
st.plotly_chart(fig2,use_container_width=True)

#Downloading Time series Analysis data
with st.expander("Time Series Analysis Data"):
        st.write(linechart.T.style.background_gradient(cmap="Greens"))
        csv = linechart.to_csv(index = False).encode('utf-8')
        st.download_button("Dowload Data", data = csv, file_name = "Time_Series_Analysis.csv", mime = "text/csv",
                           help = 'Click here to download the data as a CSV file')

# Tree map based on Region, Category, Sub-Category
st.subheader("Hierarchical View of Sales using Tree Map")
fig3 = px.treemap(filtered_df,path = ["Region", "Category", "Sub-Category"], values = "Sales", hover_data = "Sales",
    color = "Sub-Category")
fig3.update_layout(width = 800, height = 650)
st.plotly_chart(fig3,use_container_width=True)

# Segment wise and Category wise Sales charts
chart1, chart2 = st.columns((2))
with chart1 :
    st.subheader('Segment wise Sales')
    fig = px.pie(filtered_df,values = "Sales", names = "Segment", template = "plotly_dark")
    fig.update_traces(text = filtered_df["Segment"], textposition = "inside")
    st.plotly_chart(fig,use_container_width=True)

with chart2:
    st.subheader('Category wise Sales')
    fig = px.pie(filtered_df,values = "Sales", names = "Category", template = "gridon")
    fig.update_traces(text = filtered_df["Category"], textposition = "inside")
    st.plotly_chart(fig,use_container_width=True)

# Summary tables
import plotly.figure_factory as ff
st.subheader(":point_right: Month wise Sub-Category Sales Summary")
with st.expander("Summary_Table"):
    df_sample = df[0:5][["Region","State","City","Category","Sales","Profit","Quantity"]]
    fig = ff.create_table(df_sample,colorscale = "Cividis")
    st.plotly_chart(fig,use_container_width = True)

    st.markdown("Month wise sub-category Table")
    filtered_df["month"] = filtered_df["Order Date"].dt.month_name()
    sub_category_Year = pd.pivot_table(data = filtered_df, values = "Sales", index = ["Sub-Category"],columns = "month")
    st.write(sub_category_Year.style.background_gradient(cmap = "Oranges"))

#Scatter plot
data1 = px.scatter(filtered_df, x = "Sales", y = "Profit", size = "Quantity")
data1["layout"].update(title= "Relationship between Sales and Profits using Scatter Plot",
                       titlefont = dict(size = 20),
                       xaxis = dict(title = "Sales", titlefont = dict(size = 19)),
                       yaxis = dict(title = "Profit", titlefont = dict(size = 19)))
st.plotly_chart(fig,use_container_width=True)

#Dislaying all the data
st.subheader(":point_right: Entire Sales Table")
with st.expander("View Entire data"):
    st.write(filtered_df.iloc[:,1:20:2].style.background_gradient(cmap="Greens"))

#Downloading the entire data
csv = df.to_csv(index = False).encode('utf-8')
st.download_button("Dowload Data", data = csv, file_name = "Minge_SalesData_Set.csv", mime = "text/csv",
                           help = 'Click here to download the data as a CSV file')
