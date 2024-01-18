"""
按照arc*和FA-Harris的方法标记数据,构建两个点之间的时空管道
"""

from tkinter.tix import Tree
import numpy as np
import math

# point1=[123,123,345678,1]
# point2=[133,133,345688,1]

# point3 = [128,128,345683,1]

def draw_tube(corner1,corner2):
    tube = [corner1,corner2]

    return tube

def judge_corner(tube,event,circle_size=10): #circle_size为管道半径
    t = event[2]
    t0 = tube[0][2]
    t1 = tube[1][2]
    T = t1-t0
    scale = (t-t0)/T 

    center_x = scale*(tube[1][0]-tube[0][0])+tube[0][0]
    center_y = scale*(tube[1][1]-tube[0][1])+tube[0][1]

    dis = math.sqrt(math.pow(event[0]-center_x,2)+math.pow(event[1]-center_y,2))

    if dis <= circle_size and t <= t1 :
        return True
    else:
        return False

def judge_time(tube,event):
    t = event[2]
    t0 = tube[0][2]
    t1 = tube[1][2]
    if t <= t1:
        return True
    else:
        return False



# tube = draw_tube(point1,point2)
# print(judge_corner(tube,point3))
