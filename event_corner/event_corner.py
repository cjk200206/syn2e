"""文件操作，读入事件流，输出角点判断流"""

import numpy as np
import os
from .corner_tube import draw_tube, judge_corner, judge_time

# time_interval = 0.360/10*10e8

#读取帧角点
def frame_corner_tube(frame_corner_dir):
    frame_corner_file = sorted(os.listdir(frame_corner_dir))
    frame_corner = []
    for file in frame_corner_file:
        with open(os.path.join(frame_corner_dir,file)) as f:
            frame_corner.append(np.loadtxt(f))
    frame_corner = np.asarray(frame_corner)

    #加时间轴，极性轴，角点判断轴
    frame_corner_time = []
    for iter in range(len(frame_corner)):
        temp=np.insert(frame_corner[iter],2,values=0+0.04*10e8*iter,axis=1) #时间轴
        # 下面两个好像用不上
        # temp1=np.insert(temp,3,values=0,axis=1) #极性轴
        # temp2=np.insert(temp1,4,values=0,axis=1) #角点判断轴
        frame_corner_time.append(temp)
    frame_corner_time = np.asarray(frame_corner_time)

    #构建时空管道，判断事件角点数据
    tubes = []
    for iter in range(len(frame_corner[0])):
        tube = draw_tube(frame_corner_time[0][iter],frame_corner_time[1][iter])
        tubes.append(tube)

    return tubes    

#读取事件流并判断
def judge_event_corner(tubes,event_file_dir):
    event_files = sorted(os.listdir(event_file_dir))
    event_corners = []
    counter = 0
    tube_counter = 0
    break_counter = 0

    #标记事件角点并储存
    for file in event_files:
        event_file = os.path.join(event_file_dir,file)

        with open(event_file) as f:
            events = np.loadtxt(f).astype(np.int32)
        #标记事件流
            np.zeros(len(events))
        for iter in range(len(events)):
            if not judge_time(tubes[0],events[iter]): #如果不符合时间区间，直接跳过
                break_counter += 1
                break
            for tube in tubes:
                if judge_corner(tube,events[iter]):
                    event_corners.append(events[iter])
                    counter += 1

    event_corners = np.asarray(event_corners)

    return event_corners

def save_corners(event_corners,dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    np.savetxt(os.path.join(dest_dir,'event_corners.txt'),event_corners,fmt='%d')


if __name__ == "__main__":
    frame_corner_dir = 'datasets/syn_test/points/0'
    event_file_dir = 'datasets/syn_test/events/0'
    dest_dir = 'datasets/syn_test/event_corners'

    tubes = frame_corner_tube(frame_corner_dir)
    event_corners = judge_event_corner(tubes,event_file_dir)
    save_corners(event_corners,dest_dir)

                

