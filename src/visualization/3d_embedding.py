"""
By:             Gemini 3
Last updated:   24th Feb 2026

Visualize Distributed Representation & Semantic Similarity
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import ollama

def get_gemma_embedding(text):
    # Ensure you have 'embeddinggemma' pulled in Ollama
    response = ollama.embed(model='embeddinggemma', input=text)
    vector = np.array(response['embeddings'][0])
    return vector[:256] 

def create_smooth_matrix(embedding, grid_size=(64, 64), sigma=1.5):
    target_count = grid_size[0] * grid_size[1]
    x_old = np.linspace(0, 1, len(embedding))
    x_new = np.linspace(0, 1, target_count)
    
    interpolator = interp1d(x_old, embedding, kind='cubic')
    grid_flat = interpolator(x_new)
    matrix = grid_flat.reshape(grid_size)
    
    smoothed = gaussian_filter(matrix, sigma=sigma)
    return smoothed

def cosine_similarity(a, b):
    # Standard formula for vector similarity
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Classes and Examples
dataset = {
    "Malaysian Cuisine": ["Nasi Lemak", "Satay"],
    "Celestial Bodies": ["Jupiter", "Mars"],
    "Data Science": ["Gradient Descent", "Backpropagation"]
}

# --- Data Collection ---
all_items = []
all_embeddings = []
for category, items in dataset.items():
    for item in items:
        all_items.append((category, item))
        all_embeddings.append(get_gemma_embedding(item))

# --- Visualization 1: 3D Topography (3x2 Grid) ---
fig1 = plt.figure(figsize=(14, 12), facecolor='#111111')
fig1.suptitle("Gemma 3: Semantic Topography", color='white', fontsize=22, y=0.98)

X, Y = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))

for idx, (cat_item, vector) in enumerate(zip(all_items, all_embeddings)):
    category, item = cat_item
    matrix = create_smooth_matrix(vector)
    
    ax = fig1.add_subplot(3, 2, idx + 1, projection='3d')
    ax.set_facecolor('#111111') 
    
    surf = ax.plot_surface(X, Y, matrix, cmap='magma', 
                           linewidth=0, antialiased=True,
                           shade=True, rstride=1, cstride=1)
    
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f"{category}\n{item}", color='white', fontsize=11, pad=-5)
    ax.view_init(elev=35, azim=-45)

plt.subplots_adjust(wspace=0.1, hspace=0.3)

# --- Visualization 2: Similarity Heatmap ---
num_items = len(all_embeddings)
sim_matrix = np.zeros((num_items, num_items))
labels = [item[1] for item in all_items]

for i in range(num_items):
    for j in range(num_items):
        sim_matrix[i, j] = cosine_similarity(all_embeddings[i], all_embeddings[j])

fig2, ax_hm = plt.subplots(figsize=(10, 8), facecolor='#111111')
im = ax_hm.imshow(sim_matrix, cmap='magma')

# Formatting the heatmap
ax_hm.set_xticks(np.arange(num_items))
ax_hm.set_yticks(np.arange(num_items))
ax_hm.set_xticklabels(labels, color='white', rotation=45)
ax_hm.set_yticklabels(labels, color='white')
ax_hm.set_title("Sentence Similarity Heatmap", color='white', fontsize=18, pad=20)
ax_hm.set_facecolor('#111111')

# Add text annotations inside the heatmap cells
for i in range(num_items):
    for j in range(num_items):
        text = ax_hm.text(j, i, f"{sim_matrix[i, j]:.2f}",
                       ha="center", va="center", color="w", fontsize=10)

# Add colorbar
cbar = fig2.colorbar(im, ax=ax_hm)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

plt.tight_layout()
plt.show()