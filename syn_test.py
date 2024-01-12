"""创建一对生成图像，为后续转事件做准备"""
import numpy as np
import cv2
import os
import synthetic_dataset


img_save_path = ["datasets/syn_test/img/0","datasets/syn_test/img/1"]
points_save_path = ["datasets/syn_test/points/0","datasets/syn_test/points/1"]

os.makedirs(img_save_path[0],exist_ok=True)
os.makedirs(points_save_path[0],exist_ok=True)
os.makedirs(img_save_path[1],exist_ok=True)
os.makedirs(points_save_path[1],exist_ok=True)

for iter in range(100):
    image_raw = synthetic_dataset.generate_background(size=(260,346))
    # image_raw = synthetic_dataset.generate_pure_background()
    image = image_raw.copy()
    points,col = synthetic_dataset.draw_polygon_test(image)

    for i in range(2): 
        image = image_raw.copy()
        points = synthetic_dataset.move_test(image,points,col)
        cv2.imwrite(os.path.join(img_save_path[i],"{}.png".format(iter)),image)
        np.savetxt(os.path.join(points_save_path[i],"{}.txt".format(iter)),points)
        
print("finished!")


