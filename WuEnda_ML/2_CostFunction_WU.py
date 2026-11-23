
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480, 430, 630, 730])

# python中，函数定义通常放在主函数之前
def compute_cost(x, y, w, b): 
    m = x.shape[0]
    cost = 0
    cost_sum = 0
    for i in range (m):
        fwb = w*x[i]+b
        cost = (fwb-y[i])**2
        cost_sum += cost
    total_cost = (1/(2*m))*cost_sum
    return total_cost

w_range = np.linspace(-100, 400, 100) 
b_range = np.linspace(-100, 200, 100)
W, B = np.meshgrid(w_range, b_range)
J = compute_cost(x_train, y_train, W, B)
J_min = J.min()
J_max = J.max()
print(J_min)
print(J_max)

fig1, ax1 = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
surf = ax1.plot_surface(W, B, J, cmap='viridis', edgecolor='none', alpha=0.8)
ax1.set_title('Cost Function 3D Surface Plot')
ax1.set_xlabel('w')
ax1.set_ylabel('b')
ax1.set_zlabel('J(w, b)')
fig1.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
fig1.tight_layout()

fig2, ax2 = plt.subplots(figsize=(8, 6))
contour = ax2.contour(W, B, J, levels=30, cmap='viridis')
ax2.contourf(W, B, J, levels=30, cmap='viridis', alpha=0.8)
ax2.clabel(contour, inline=True, fontsize=8) # 标上数值
ax2.scatter(200, 0, color='red', marker='x', s=100, label='Best Solution')
ax2.set_title('Cost Function Contour Plot')
ax2.set_xlabel('w')
ax2.set_ylabel('b')
ax2.legend()
fig2.tight_layout()

plt.show()
