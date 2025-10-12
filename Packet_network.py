import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import warnings; warnings.filterwarnings("ignore")


# Load 
df = pd.read_csv(r'C:\Users\hp\Saclay-ai\ML\train_data.csv')
X = df.drop(columns=["packet_id", "target"])
y = df["target"]

# Encode categorical columns
for col in ["second_frame", "src_ip", "dest_ip", "protocol", "info"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Split 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# PCA 
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

# Algos
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)      # Perceptron(max_iter=1000, random_state=42)
model.fit(X_train_pca, y_train)


# Predict
y_val_prob = model.predict(X_val_pca)                                               # using log_loss model.predict_proba(X_val_pca)[:, 1] 
accuracy = accuracy_score(y_val, y_val_prob) 
print(f"Validation Accuracy: {accuracy * 100:.2f} %")


# Visualize 
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)                           # model.predict(grid).reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, probs, 25, cmap="RdBu", alpha=0.8)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap="RdBu", edgecolor="white", s=40, label="Train")
plt.scatter(X_val_pca[:, 0], X_val_pca[:, 1], c=y_val, cmap="coolwarm", edgecolor="black", s=40, marker='x', label="Val")
plt.title("Logistic Regression Decision Boundary (PCA 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Predicted Probability")
plt.legend()
plt.show()