import math
import matplotlib.pyplot as plt  # For visualization

# import earlier code
from model.positioning import PositionalEncoding

# --- Visualization Code ---


def plot_positional_encoding(pe_module, max_len=50, d_model=128):
    """Plots the positional encoding matrix"""
    pe = pe_module.pe.squeeze(0).numpy()  # Remove batch dim and convert to numpy
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(pe, cmap="viridis")
    plt.xlabel("Embedding Dimension (i)")
    plt.xlim((0, d_model))
    plt.ylabel("Position (pos)")
    plt.ylim((max_len, 0))  # Invert y-axis to show position 0 at top
    plt.title("Positional Encoding Matrix")
    plt.colorbar()
    plt.show()


# Example Usage (assuming you have defined the PositionalEncoding class above)
d_model_vis = 128
max_len_vis = 50

pe_module = PositionalEncoding(d_model_vis, max_len_vis)
plot_positional_encoding(pe_module, max_len=max_len_vis, d_model=d_model_vis)
