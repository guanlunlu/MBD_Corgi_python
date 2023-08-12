""" import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

dt = 0.005
n=20
L = 1
particles=np.zeros(n,dtype=[("position", float , 2),
                           ("velocity", float ,2),
                           ("force", float ,2),
                           ("size", float , 1)])

particles["position"]=np.random.uniform(0,L,(n,2))
particles["velocity"]=np.zeros((n,2))
particles["size"]=0.5*np.ones(n)

fig = plt.figure(figsize=(7,7))
ax = plt.axes(xlim=(0,L),ylim=(0,L))
scatter=ax.scatter(particles["position"][:,0], particles["position"][:,1])

def update(frame_number):
   particles["force"]=np.random.uniform(-2,2.,(n,2))
   particles["velocity"] = particles["velocity"] + particles["force"]*dt
   particles["position"] = particles["position"] + particles["velocity"]*dt

   particles["position"] = particles["position"]%L
   scatter.set_offsets(particles["position"])
#    print(type(particles["position"]))
#    return scatter,

anim = FuncAnimation(fig, update, interval=10)
plt.show() """

""" import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd


a = np.random.rand(2000, 3)*10
t = np.array([np.ones(100)*i for i in range(20)]).flatten()
df = pd.DataFrame({"time": t ,"x" : a[:,0], "y" : a[:,1], "z" : a[:,2]})

def update_graph(num):
    data=df[df['time']==num]
    graph._offsets3d = (data.x, data.y, data.z)
    title.set_text('3D Test, time={}'.format(num))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

data=df[df['time']==0]
graph = ax.scatter(data.x, data.y, data.z)

ani = matplotlib.animation.FuncAnimation(fig, update_graph, 19, 
                               interval=40, blit=False)

plt.show() """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

def update(step, xs, ys, zs, point):
    # use the following two lines for matplotlib 3.4.3
    point.set_data(xs[step], ys[step])
    point.set_3d_properties(zs[step])
    
    # use the following two lines for matplotlib 3.5.1
    #point.set_data([xs[step]], [ys[step]])
    #point.set_3d_properties([zs[step]])
    return point

fig = plt.figure()
ax = p3.Axes3D(fig)
fig.add_axes(ax)

point = ax.plot([], [], [], '.')[0]

xs = np.arange(100)
ys = np.arange(100)
zs = np.arange(100)

ax.set_xlim3d([0,100])
ax.set_ylim3d([0,100])
ax.set_zlim3d([0,100])

anim = animation.FuncAnimation(fig, update, 100,
                fargs=(xs, ys, zs, point), interval=1, blit=False)

plt.show()