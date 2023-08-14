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

""" import numpy as np
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

plt.show() """

""" import linkleg_transform as lt
from LegKinematics import *
import numpy as np
import dill
import time
with open('./serialized_object/vec_OG_NP.pkl', 'rb') as d:
    vec_OG = dill.load(d)

t = time.time()
state = np.array([[np.deg2rad(45)],[0]])
phi = lt.getPhiRL(state)
v_OG = vec_OG(phi[0, 0], phi[1, 0])
print("symbolic",v_OG.T)
print("symbolic time elapsed", time.time()-t)
t = time.time()

print("numeric", FowardKinematics(state).T)
print("numeric time elapsed", time.time()- t) """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
 
# References
# https://gist.github.com/neale/e32b1f16a43bfdc0608f45a504df5a84
# https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
# https://riptutorial.com/matplotlib/example/23558/basic-animation-with-funcanimation
 
# ANIMATION FUNCTION
def func(num, dataSet, line, redDots):
    # NOTE: there is no .set_data() for 3 dim data...\
    print(dataSet[0:2, :num])
    print("--")
    print(dataSet[2, :num])
    print("++++++")
    line.set_data(dataSet[0:2, :num])    
    line.set_3d_properties(dataSet[2, :num])    
    redDots.set_data(dataSet[0:2, :num])    
    redDots.set_3d_properties(dataSet[2, :num]) 
    return line
 
 
# THE DATA POINTS
t = np.arange(0,20,0.2) # This would be the z-axis ('t' means time here)
x = np.cos(t)-1
y = 1/2*(np.cos(2*t)-1)
dataSet = np.array([x, y, t])
numDataPoints = len(t)
 
# GET SOME MATPLOTLIB OBJECTS
fig = plt.figure()
ax = Axes3D(fig)
redDots = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='r', marker='o')[0] # For scatter plot
# NOTE: Can't pass empty arrays into 3d version of plot()
line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='g')[0] # For line plot
 
# AXES PROPERTIES]
# ax.set_xlim3d([limit0, limit1])
ax.set_xlabel('X(t)')
ax.set_ylabel('Y(t)')
ax.set_zlabel('time')
ax.set_title('Trajectory of electron for E vector along [120]')
 
# Creating the Animation object
line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet,line,redDots), interval=50, blit=False)
# line_ani.save(r'Animation.mp4')
 
 
plt.show()