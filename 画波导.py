import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 创建图形和3D坐标轴
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 设置视角
ax.view_init(elev=30, azim=45)

# 波导尺寸参数
L, W, H = 10, 6, 4  # 外部尺寸
l, w, h = 8, 4, 2   # 内部空心尺寸

# 定义颜色参数
surface_color = 'royalblue'
alpha = 1

# 绘制前面（x=L）的分割面
def plot_front():
    # 左右边框
    for y_range in [(0, (W-w)/2), ((W+w)/2, W)]:
        Y, Z = np.meshgrid(np.linspace(*y_range, 2), 
                          np.linspace(0, H, 2))
        X = L * np.ones_like(Y)
        ax.plot_surface(X, Y, Z, color=surface_color, alpha=alpha)
    
    # 上下边框
    for z_range in [(0, (H-h)/2), ((H+h)/2, H)]:
        Y, Z = np.meshgrid(np.linspace((W-w)/2, (W+w)/2, 2),
                          np.linspace(*z_range, 2))
        X = L * np.ones_like(Y)
        ax.plot_surface(X, Y, Z, color=surface_color, alpha=alpha)

# 绘制顶部（z=H）的分割面
def plot_top():
    # 一整个面
    X, Y = np.meshgrid(np.linspace(0, L, 2),
                      np.linspace(0, W, 2))
    Z = H * np.ones_like(X)
    ax.plot_surface(X, Y, Z, color=surface_color, alpha=alpha)

# 绘制右侧面（y=W）的分割面
def plot_right():
    # 一整个面
    X, Z = np.meshgrid(np.linspace(0, L, 2),
                      np.linspace(0, H, 2))
    Y = W * np.ones_like(X)
    ax.plot_surface(X, Y, Z, color=surface_color, alpha=alpha)

# 内侧下面
X, Y = np.meshgrid(np.linspace((L-l)/2, L, 2),
                  np.linspace((W-w)/2, (W+w)/2, 2))
Z = np.ones_like(Y)
ax.plot_surface(X, Y, Z, color=surface_color, alpha=alpha)

# 内侧左面
# 直接按照4个边界点绘制面
X = np.array([[0, 0], [0, 0]])
Y = np.array([[0, W], [0, W]])
Z = np.array([[0, 0], [H, H]])
ax.plot_surface(X, Y, Z, color=surface_color, alpha=alpha)

# 执行绘图函数
plot_front()
plot_top()
plot_right()

# 图形设置
ax.set_xlim(0, L)
ax.set_ylim(0, W)
ax.set_zlim(0, H)
ax.axis('off')  # 关闭坐标轴

# 设置透明背景
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

ax.set_box_aspect([L, W, H])  # 设置坐标轴比例

# 焦距无穷大
ax.set_proj_type('ortho')


# 保存为透明PNG
plt.savefig('waveguide.png', transparent=True, dpi=300, bbox_inches='tight')
plt.close()

plt.show()