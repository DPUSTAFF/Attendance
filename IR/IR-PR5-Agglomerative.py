# Agglomerative Hierarchical Clustering on Retail Sales Data (Fixed Version)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Load dataset
data = pd.read_excel("Online Retail.xlsx")  # change sep="," if CSV uses commas

# Step 2: Data preprocessing
data = data.dropna(subset=['CustomerID'])
data["TotalAmount"] = data["Quantity"] * data["UnitPrice"]

customer_data = data.groupby("CustomerID").agg({
    "Quantity": "sum",
    "TotalAmount": "sum"
}).reset_index()

print("Sample Data:\n", customer_data.head())

# Step 3: Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[["Quantity", "TotalAmount"]])

# Step 4: Dendrogram
linked = linkage(scaled_data, method='ward')

plt.figure(figsize=(8, 5))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title("Customer Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Step 5: Agglomerative Clustering (use 'metric' instead of 'affinity')
cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
customer_data["Cluster"] = cluster.fit_predict(scaled_data)

# Step 6: Visualization
plt.figure(figsize=(7, 5))
sns.scatterplot(
    x="Quantity", y="TotalAmount",
    hue="Cluster", data=customer_data,
    palette="rainbow", s=60
)
plt.title("Customer Segmentation using Agglomerative Clustering")
plt.xlabel("Total Quantity Purchased")
plt.ylabel("Total Amount Spent")
plt.show()

# Step 7: Cluster Summary
print("\nCluster Summary:")
print(customer_data.groupby("Cluster")[["Quantity", "TotalAmount"]].mean())