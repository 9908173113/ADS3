#!/usr/bin/env python
# coding: utf-8



#importing the important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import norm




#loading the population density data
population_density_data = pd.read_csv("population_density.csv", skiprows=4)
population_density_data = population_density_data[["Country Name", "2019"]] 
population_density_data = population_density_data.rename(columns={"Country Name": "Country", "2019": "Population density"})
population_density_data.head()




#loading the clean fuels data
clean_fuels_data = pd.read_csv("clean_fuel.csv", skiprows=4)
clean_fuels_data = clean_fuels_data[["Country Name", "2019"]] 
clean_fuels_data = clean_fuels_data.rename(columns={"Country Name": "Country", "2019": "Access to Clean Fuels"})
clean_fuels_data.head()




# Merge the two dataframes based on the 'Country' column
merged_data = pd.merge(population_density_data, clean_fuels_data, on='Country')
merged_data.head()




#checking for missing values
merged_data.isnull().sum()




# Drop rows with missing values
merged_data.dropna(inplace=True)




#checking for missing values after dropping them
merged_data.isnull().sum()









# Extract the numerical values for clustering
X = merged_data.iloc[:, 1:].values
X




# Normalize the data
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
X_normalized




# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_normalized)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_


# To visualize the clustering results, we can create a scatter plot with the two variables, where each point is colored based on its cluster membership:



# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', c='red', s=100, label='Cluster Centers')
plt.xlabel('Population density (people per sq. km of land area)')
plt.ylabel('Access to Clean Fuels and Technologies for Cooking (% of population)')
plt.title('Clustering Analysis')
plt.legend()
plt.show()


# This code will generate a scatter plot where each point represents a country, and the color indicates the cluster it belongs to. The cluster centers are denoted by red crosses.



# Define the function for the model you want to fit
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c

# Extract the relevant columns from the dataset
population_density = merged_data['Population density'].astype(float)
clean_fuels_access = merged_data['Access to Clean Fuels'].astype(float)

# Define the function for estimating confidence ranges
def err_ranges(x, popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    z_score = np.abs(norm.ppf(alpha / 2))
    lower = polynomial_function(x, *(popt - z_score * perr))
    upper = polynomial_function(x, *(popt + z_score * perr))
    return lower, upper

# Fit the model to the data
popt, pcov = curve_fit(polynomial_function, population_density, clean_fuels_access)

# Make predictions for future values
population_density_future = np.linspace(np.min(population_density), np.max(population_density), 100)
clean_fuels_access_pred = polynomial_function(population_density_future, *popt)

# Calculate confidence ranges
lower, upper = err_ranges(population_density_future, popt, pcov)

# Plot the data, best fitting function, and confidence range
plt.figure(figsize=(10, 6))
plt.scatter(population_density, clean_fuels_access, label='Data')
plt.plot(population_density_future, clean_fuels_access_pred, color='red', label='Best Fit')
plt.fill_between(population_density_future, lower, upper, color='gray', alpha=0.3, label='Confidence Range')
plt.xlabel('Population density (people per sq. km of land area)')
plt.ylabel('Access to Clean Fuels and Technologies for Cooking (% of population)')
plt.title('Polynomial Fit')
plt.legend()
plt.show()


# Based on the provided variables, "Population density (people per sq. km of land area)" and "Access to Clean Fuels and Technologies for Cooking (% of population)," we can explore the relationship between population density and access to clean fuels.
# 
# The plot illustrates the scatter plot of the data points, the best-fitting polynomial function (red line), and the confidence range (gray shaded area).
# 
# The polynomial fit suggests that there is a non-linear relationship between population density and access to clean fuels. As population density increases, the percentage of the population with access to clean fuels initially rises rapidly and then levels off. This pattern suggests that as population density increases, there is an initial effort to provide clean fuels and technologies for cooking to a larger portion of the population. However, beyond a certain threshold, the rate of improvement in clean fuel access slows down.
# 
# The confidence range represents the uncertainty in the predictions. It provides a range within which future values are likely to fall given the fitted model and the associated uncertainties.
# 
# Based on the polynomial fit, we can make predictions for future values of access to clean fuels based on different levels of population density. However, it's important to note that as we move further away from the observed data, the uncertainty in the predictions increases.
# 
# This analysis highlights the importance of considering population density when addressing access to clean fuels and technologies for cooking. It suggests that policies and interventions should take into account the specific challenges associated with population density and tailor solutions accordingly. Additionally, the confidence range serves as a reminder of the uncertainty inherent in making predictions and underscores the need for ongoing monitoring and evaluation to refine our understanding of the relationship between population density and access to clean fuels.



for l, u in zip(lower, upper):
    print(f"Lower limit: {l:.2f}, Upper limit: {u:.2f}")


# By printing the lower and upper limits, we can get a sense of the variability and range of possible values for the access to clean fuels at different population density levels. This information is valuable for understanding the confidence and uncertainty in the predictions and can help inform decision-making processes when considering policies or interventions related to access to clean fuels.




# Define the function for the model you want to fit
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c

# Extract the relevant columns from the dataset
population_density = merged_data['Population density'].astype(float)
clean_fuels_access = merged_data['Access to Clean Fuels'].astype(float)


# Define the function for estimating confidence ranges
def err_ranges(x, popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    z_score = np.abs(stats.norm.ppf(alpha / 2))
    lower = polynomial_function(x, *(popt - z_score * perr))
    upper = polynomial_function(x, *(popt + z_score * perr))
    return lower, upper


# Perform clustering on the dataset
X = np.column_stack((population_density, clean_fuels_access))
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)

# Add cluster labels to the merged dataset
merged_data['Cluster'] = labels

# Initialize a list to store the fitting results for each cluster
fit_results = []

# Iterate over each cluster
for cluster_id in range(3):
    # Get the data points belonging to the current cluster
    cluster_data = merged_data[merged_data['Cluster'] == cluster_id]
    
    # Check if the cluster has sufficient data points for fitting a curve with three parameters
    if len(cluster_data) < 3:
        print(f"Cluster {cluster_id} does not have enough data points for fitting a curve.")
        continue
    
    # Fit the model to the data
    popt, pcov = curve_fit(polynomial_function, cluster_data['Population density'], cluster_data['Access to Clean Fuels'])
    
    # Store the fitting results
    fit_results.append((popt, pcov))

# Plot the data, best fitting function, and confidence range for each cluster
plt.figure(figsize=(10, 6))
for cluster_id, fit_result in enumerate(fit_results):
    popt, pcov = fit_result
    cluster_data = merged_data[merged_data['Cluster'] == cluster_id]
    population_density_future = np.linspace(np.min(population_density), np.max(population_density), 100)
    clean_fuel_access_pred = polynomial_function(population_density_future, *popt)
    lower, upper = err_ranges(population_density_future, popt, pcov)
    plt.scatter(cluster_data['Population density'], cluster_data['Access to Clean Fuels'], label=f'Cluster {cluster_id}')
    plt.plot(population_density_future, clean_fuel_access_pred, label=f'Cluster {cluster_id} - Best Fit')
    plt.fill_between(population_density_future, lower, upper, color='gray', alpha=0.3, label=f'Cluster {cluster_id} - Confidence Range')

plt.xlabel('Population density (people per sq. km of land area)')
plt.ylabel('Access to Clean Fuels and Technologies for Cooking (% of population)')
plt.title('Polynomial Fit by Cluster')
plt.legend()
plt.show()


# Define color properties
boxprops = dict(linestyle='-', linewidth=2, color='black')
medianprops = dict(linestyle='-', linewidth=2, color='red')
whiskerprops = dict(linestyle='-', linewidth=2, color='blue')
capprops = dict(linestyle='-', linewidth=2, color='green')

# Create boxplots for population density for each cluster
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Population Density Distribution by Cluster')
for cluster_id in range(3):
    cluster_data = merged_data[merged_data['Cluster'] == cluster_id]
    plt.boxplot(cluster_data['Population density'], positions=[cluster_id], widths=0.6, 
                boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
plt.xticks([0, 1, 2], ['Cluster 0', 'Cluster 1', 'Cluster 2'])
plt.ylabel('Population Density (people per sq. km of land area)')

# Create boxplots for access to clean fuels for each cluster
plt.subplot(1, 2, 2)
plt.title('Access to Clean Fuels Distribution by Cluster')
for cluster_id in range(3):
    cluster_data = merged_data[merged_data['Cluster'] == cluster_id]
    plt.boxplot(cluster_data['Access to Clean Fuels'], positions=[cluster_id], widths=0.6, 
                boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
plt.xticks([0, 1, 2], ['Cluster 0', 'Cluster 1', 'Cluster 2'])
plt.ylabel('Access to Clean Fuels and Technologies for Cooking (% of population)')

plt.tight_layout()
plt.show()


# The code performs a polynomial curve fitting on the relationship between population density and access to clean fuels and technologies for cooking. It uses K-means clustering to identify three distinct clusters in the data and fits a polynomial function to each cluster.
# 
# The plot generated shows the data points, the best-fitting polynomial curve for each cluster, and the confidence range around the curve.
# 
# The x-axis represents the population density, measured in people per square kilometer of land area. The y-axis represents the percentage of the population with access to clean fuels and technologies for cooking.
# 
# By visually examining the plot, we can observe the following:
# 
# #### Cluster 0: 
# 
# This cluster is represented by the blue data points and the corresponding curve. It shows a positive correlation between population density and access to clean fuels. As the population density increases, the percentage of the population with access to clean fuels also tends to increase. The confidence range (shaded gray area) around the curve provides an estimate of the uncertainty in the fit.






