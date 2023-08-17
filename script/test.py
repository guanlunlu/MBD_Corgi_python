import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.patches as patches

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)

v= np.array([
    [0., -0.25],  
    [0.5, 0.],  
    [0., 0.25]
    ])

patch = patches.Polygon(v,closed=True, fc='r', ec='r')
ax.add_patch(patch)

def init():
    return patch,

def animate(i):
    v[:,0]+=i
    patch.set_xy(v)
    return patch,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 5), init_func=init,
                              interval=1000, blit=True)
plt.show()