import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

INDEX_FAISS_PATH = BACKEND_DIR / "faiss_data" / "index.faiss"
INDEX_NPY_PATH = BACKEND_DIR / "faiss_data" / "index.npy"

vectors = None

# Try loading from index.npy fallback or via FAISS
if INDEX_NPY_PATH.exists():
    vectors = np.load(str(INDEX_NPY_PATH))
elif INDEX_FAISS_PATH.exists():
    try:
        import faiss
        index = faiss.read_index(str(INDEX_FAISS_PATH))
        vectors = np.array([
            index.reconstruct(i)
            for i in range(index.ntotal)
        ])
    except Exception as error:
        print(f"Could not load via FAISS: {error}")

if vectors is None or len(vectors) == 0:
    print("No vectors found in faiss_data to visualize.")
    sys.exit(1)

print("Number of vectors:", vectors.shape[0])
print("Vector dimension:", vectors.shape[1])

# Reduce 768 dimensions → 2 dimensions
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(10, 7))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6, edgecolors="none", color="#4A90E2")

plt.title("Dense Vector Space Visualization (PCA 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True, linestyle="--", alpha=0.5)

output_plot = BACKEND_DIR / "vector_space_pca.png"
plt.savefig(str(output_plot), dpi=150, bbox_inches="tight")
print(f"Vector plot saved to: {output_plot}")