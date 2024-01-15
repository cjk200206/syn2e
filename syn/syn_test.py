"""创建一对生成图像，为后续转事件做准备"""
import numpy as np
import cv2
import os
from . import synthetic_dataset


# img_save_path = "datasets/syn_test/img"
# points_save_path = "datasets/syn_test/points"





def syn_polygon(img_save_path,points_save_path,num_of_pics=100):
    os.makedirs(img_save_path,exist_ok=True)
    os.makedirs(points_save_path,exist_ok=True) 

    for iter in range(num_of_pics):
        image_raw = synthetic_dataset.generate_background(size=(260,346))
        # image_raw = synthetic_dataset.generate_pure_background()
        image = image_raw.copy()
        points,col = synthetic_dataset.draw_polygon(image)
        os.makedirs(os.path.join(img_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(points_save_path,str(iter)),exist_ok=True)

        for i in range(2): 
            image = image_raw.copy()
            points = synthetic_dataset.move_polygon(image,points,col)
            cv2.imwrite(os.path.join(img_save_path,str(iter),"{}.png".format(i)),image)
            np.savetxt(os.path.join(points_save_path,str(iter),"{}.txt".format(i)),points)
            
    print("finished!")

# if __name__ == "__main__":
#     # make_syn_frames(img_save_path,points_save_path,100)
