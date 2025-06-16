from matplotlib.colors import ListedColormap 
import matplotlib.pyplot as plt

colors = ['#4682B4',   # 0: EMPTY (SteelBlue)
          '#8B4513',   # 1: BLOCK (SaddleBrown)
          '#FFFFFF',   # 2: AGENT (White)
          '#FFD700']   # 3: FOOD (Gold)
cmap = ListedColormap(colors)
# Define bounds for the color mapping to ensure exact mapping
bounds = [0, 1, 2, 3, 4] # Needs one more bound than colors
norm = plt.Normalize(vmin=0, vmax=3) # Normalize values to the colormap range