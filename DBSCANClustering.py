import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Function to handle DBSCAN and visualizations
def run_dbscan(df, eps, min_samples):
    try:
        # Preprocessing: Fill missing values
        df['Year'] = df['Year'].fillna(df['Year'].median())
        df['Publisher'] = df['Publisher'].fillna('Unknown')

        # Scale the numerical features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year']])

        # Apply DBSCAN clustering with user-defined parameters
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_features)

        # Add the cluster labels to the original DataFrame
        df['Cluster'] = clusters

        # Cluster analysis: Count of clusters
        st.write("Cluster count:")
        st.write(df['Cluster'].value_counts())

        # Group by cluster and calculate mean sales per region
        cluster_summary = df.groupby('Cluster')[['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].mean()
        st.write("Cluster summary (Average Sales per Region):")
        st.write(cluster_summary)

        # Visualize average sales across regions for each cluster
        st.write("Average Sales per Region for Each Cluster:")
        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_summary.plot(kind='bar', ax=ax)
        plt.title('Average Sales per Region for Each Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Average Sales')
        st.pyplot(fig)

        # PCA for 2D visualization
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(scaled_features)
        st.write("PCA - DBSCAN Clustering Results:")
        fig, ax = plt.subplots()
        scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=df['Cluster'], cmap='plasma', s=50, alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title('DBSCAN Clustering Results (PCA-reduced Data)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        st.pyplot(fig)

        # t-SNE for 2D visualization
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features_tsne = tsne.fit_transform(scaled_features)
        st.write("t-SNE - DBSCAN Clustering Results:")
        fig, ax = plt.subplots()
        scatter_tsne = ax.scatter(reduced_features_tsne[:, 0], reduced_features_tsne[:, 1], c=df['Cluster'], cmap='plasma', s=50, alpha=0.6)
        plt.colorbar(scatter_tsne, label='Cluster')
        plt.title('DBSCAN Clustering Results (t-SNE-reduced Data)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        st.pyplot(fig)

        # Filter the dataframe to get only Cluster -1 (outliers)
        outliers = df[df['Cluster'] == -1]
        st.write("Outliers (Cluster -1):")
        st.write(outliers[['Name', 'Platform', 'Genre', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Global_Sales']])

        # Silhouette score (excluding outliers)
        filtered_df = df[df['Cluster'] != -1]
        filtered_features = scaled_features[df['Cluster'] != -1]
        if len(np.unique(filtered_df['Cluster'])) > 1:  # Silhouette score requires at least 2 clusters
            score = silhouette_score(filtered_features, filtered_df['Cluster'])
            st.write(f'Silhouette Score (excluding outliers): {score}')
        else:
            st.write("Silhouette Score: Not applicable (only one cluster found excluding outliers)")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check the data or try different DBSCAN parameters.")

# Streamlit app layout
st.title('DBSCAN Clustering Analysis of Video Games Sales Data')

# Add a file uploader to allow users to upload CSV files
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the first few rows of the DataFrame
        st.write("Here are the first few rows of your file:")
        st.write(df.head())

        # User inputs for DBSCAN parameters
        eps = st.slider('Select eps (Neighborhood size):', min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        min_samples = st.slider('Select min_samples (Minimum samples per cluster):', min_value=1, max_value=20, value=8)

        # Run DBSCAN clustering and analysis with user-selected parameters
        run_dbscan(df, eps, min_samples)

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
else:
    st.write("Please upload a CSV file to proceed.")
