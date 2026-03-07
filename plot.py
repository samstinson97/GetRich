import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y1, label="sin(x)", color="steelblue", linewidth=2)
ax.plot(x, y2, label="cos(x)", color="coral", linewidth=2)

ax.set_title("Sine and Cosine Waves")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot.png", dpi=150)
plt.show()
print("Plot saved to plot.png")
