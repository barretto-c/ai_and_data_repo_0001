import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation


# 1. Define a basic supervised learning flow with end user
edges = [
    ("End User", "Input Data"),
    ("Input Data", "Model"),
    ("Model", "Prediction"),
    ("Prediction", "Loss Calculation"),
    ("Loss Calculation", "Model"),  # Feedback loop
    ("Prediction", "End User"),      # End user uses the prediction
]

# 2. Set up graph and positions
G = nx.DiGraph()
G.add_edges_from(edges)
pos = {
    "End User": (-1, 1),
    "Input Data": (0, 1),
    "Model": (1.5, 1),
    "Prediction": (3, 1),
    "Loss Calculation": (4.5, 1),
}


# 3. Draw static background (nodes, labels)
fig, ax = plt.subplots(figsize=(8, 3))
nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=1800)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)


# 4. Animation function
def update(i):
    ax.clear()
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=1800)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)
    # Draw completed edges in blue, current edge in red, rest in gray
    if i > 0:
        nx.draw_networkx_edges(G, pos, edgelist=edges[:i], edge_color='blue', width=4, ax=ax)
    if i < len(edges):
        nx.draw_networkx_edges(G, pos, edgelist=[edges[i]], edge_color='red', width=4, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges[i+1:], edge_color='lightgray', style='dashed', width=2, ax=ax)
    ax.set_title("Basic Supervised Learning Flow", fontsize=14)
    ax.set_axis_off()

# 5. Run animation
ani = animation.FuncAnimation(fig, update, frames=len(edges)+1, interval=1000, repeat=True)
plt.show()