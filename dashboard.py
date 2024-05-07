import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import scipy.stats as stats
import seaborn as sns

# Load your data
data = pd.ExcelFile("Global Superstore lite.xlsx")

# Reading each sheet into a DataFrame
orders_df = data.parse('Orders')

# Convert 'Sub-Category' column into dummies i.e. Each unique value in the sub-category variable is converted into a column. 
transaction_df = pd.get_dummies(orders_df['Sub-Category'])

# Concatenate 'Order ID' column to transaction_df
transaction_df = pd.concat([orders_df['Order ID'], transaction_df], axis=1)

# Group by 'Order ID' and sum the occurrence of each sub-category for each order 
transaction_df = transaction_df.groupby('Order ID').sum()

# Replace occurrence greater than 1 with 1  - presence = 1 & absence = 0
transaction_df = transaction_df.applymap(lambda x: 1 if x > 0 else 0)

# Market Basket Analysis - Utilizing the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(transaction_df, min_support=0.001, use_colnames=True)

# Generate association rules with a confidence threshold of 5%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)

# Calculate standard error of confidence
std_error_confidence = stats.sem(rules['confidence'])

# Calculate critical value (e.g., for 95% confidence interval)
critical_value = stats.t.ppf(0.975, len(rules)-1)

# Calculate margin of error
margin_of_error = critical_value * std_error_confidence

# Calculate confidence interval
confidence_interval_lower = rules['confidence'] - margin_of_error
confidence_interval_upper = rules['confidence'] + margin_of_error

# Add confidence interval to the DataFrame
rules['confidence_interval_lower'] = confidence_interval_lower
rules['confidence_interval_upper'] = confidence_interval_upper

# Define functions to create bar charts
def create_support_bar_chart():
    # Plotting support
    support = rules['support']
    plt.bar(range(len(support)), support, color='green')

    # Adding labels and title
    plt.xlabel('Association Rules')
    plt.ylabel('Support Values')
    plt.title('Bar Graph of Support Values')

    st.pyplot()

def create_confidence_bar_chart():
    confidence = rules['confidence']
    plt.bar(range(len(confidence)), confidence, color='orange')

    # Adding labels and title
    plt.xlabel('Association Rules')
    plt.ylabel('Confidence Values')
    plt.title('Bar Graph of Confidence Values')

    st.pyplot()

def create_lift_bar_chart():
    lift = rules['lift']
    plt.bar(range(len(lift)), lift, color='blue')

    # Adding a horizontal line for values above 1
    plt.axhline(y=1, color='black', linestyle='--')

    # Adding labels and title
    plt.xlabel('Association Rules')
    plt.ylabel('Lift Values')
    plt.title('Bar Graph of Lift Values')

    st.pyplot()
    
def create_heatmap():
    # Co-occurrence matrix
    co_occurrence_matrix = transaction_df.T.dot(transaction_df)

    # Set diagonal elements to 0
    np.fill_diagonal(co_occurrence_matrix.values, 0)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(co_occurrence_matrix, annot=True, cmap="YlGnBu", fmt="d", linewidths=2)
    plt.title('Co-occurrence Matrix of Sub-Categories (Excluding Same Product Combinations)')
    plt.xlabel('Sub-Category')
    plt.ylabel('Sub-Category')
    st.pyplot()


# Main function to run the Streamlit app
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Market Basket Analysis of Minger Sales')
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("""
### Association Rules:""")
    st.dataframe(rules)
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("""
## And here are some visualizations:""")
    st.write("""
### Bar chart for support values:""")
    create_support_bar_chart()
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("""
### Bar chart for confidence values:""")
    create_confidence_bar_chart()
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("""
### Bar chart for lift values:""")
    create_lift_bar_chart()
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write("""
### Co-occurrence Matrix of Sub-Categories (Excluding Same Product Combinations) (Heatmap):""")
    create_heatmap()
if __name__ == '__main__':
    main()
