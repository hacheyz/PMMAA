# MATLAB-Style 3D Surface Plot
def plot3d(ax, x, y, z, cmap='viridis', grid='-'):
    ax.plot_surface(x, y, z, cmap=cmap)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['linestyle'] =  grid
    ax.yaxis._axinfo["grid"]['linestyle'] =  grid
    ax.zaxis._axinfo["grid"]['linestyle'] =  grid
    ax.xaxis._axinfo["grid"]['linewidth'] =  0.5
    ax.yaxis._axinfo["grid"]['linewidth'] =  0.5
    ax.zaxis._axinfo["grid"]['linewidth'] =  0.5
    ax.xaxis._axinfo["grid"]['color'] =  '0.9'
    ax.yaxis._axinfo["grid"]['color'] =  '0.9'
    ax.zaxis._axinfo["grid"]['color'] =  '0.9'
    ax.set_xlabel('$x$', fontsize=6, labelpad=-12)
    ax.set_ylabel('$y$', fontsize=6, labelpad=-12)
    ax.set_zlabel('$z$', fontsize=6, labelpad=-12)
    ax.tick_params(which='major', width=0.5, labelsize=5, axis='x', pad=-5)
    ax.tick_params(which='major', width=0.5, labelsize=5, axis='y', pad=-5)
    ax.tick_params(which='major', width=0.5, labelsize=5, axis='z', pad=-5)
    return ax