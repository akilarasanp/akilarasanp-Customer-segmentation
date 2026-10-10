"""Customer segmentation on Mall_Customers.csv using K-Means (scikit-learn)."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv").rename(columns={
    "Genre": "Gender", "Annual Income (k$)": "Income", "Spending Score (1-100)": "Spending"})
print(df.shape, "missing:", df.isna().sum().sum(), "duplicates:", df.duplicated().sum())

X = StandardScaler().fit_transform(df[["Income", "Spending"]])

# Choose K: elbow (inertia) + silhouette score
ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(m.inertia_); sil.append(silhouette_score(X, m.labels_))
best_k = list(ks)[sil.index(max(sil))]
print("silhouette by K:", {k: round(s, 3) for k, s in zip(ks, sil)}, "-> best K =", best_k)

df["Cluster"] = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(X)
prof = df.groupby("Cluster").agg(Customers=("CustomerID", "count"), Age=("Age", "mean"),
    Income=("Income", "mean"), Spending=("Spending", "mean"),
    Female_pct=("Gender", lambda s: (s == "Female").mean() * 100)).round(1)
print(prof)

# Name segments from their income / spending levels
def name(r):
    hi_i, hi_s = r.Income >= 70, r.Spending >= 60
    if hi_i and hi_s: return "Premium Spenders"
    if hi_i: return "Careful Wealthy"
    if r.Income < 40 and hi_s: return "Young Enthusiasts"
    if r.Income < 40: return "Budget Savers"
    return "Mainstream"
prof["Segment"] = prof.apply(name, axis=1)
df["Segment"] = df["Cluster"].map(prof["Segment"])
df.to_csv("Mall_Customers_Segmented.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].plot(ks, inertia, "o-"); ax[0].set(title="Elbow method", xlabel="K", ylabel="Inertia")
ax[1].plot(ks, sil, "o-", color="orange"); ax[1].set(title="Silhouette score", xlabel="K")
for s, g in df.groupby("Segment"):
    ax[2].scatter(g.Income, g.Spending, label=f"{s} ({len(g)})", s=40)
ax[2].set(title="Customer segments", xlabel="Annual income (k$)", ylabel="Spending score"); ax[2].legend()
plt.tight_layout(); plt.savefig("segmentation_charts.png", dpi=130)
