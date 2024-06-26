"""创建一对生成图像，为后续转事件做准备"""
import numpy as np
import cv2
import os
from . import synthetic_dataset
# import synthetic_dataset


img_save_path = "datasets/syn_test/img"
points_save_path = "datasets/syn_test/points"
corner_img_save_path = "datasets/syn_test/corner_img"


def syn_polygon(img_save_path,points_save_path,corner_img_save_path,num_of_pics=100,img_size = [260,346]):
    os.makedirs(img_save_path,exist_ok=True)
    os.makedirs(points_save_path,exist_ok=True) 
    os.makedirs(corner_img_save_path,exist_ok=True) 
    row = img_size[0]
    column = img_size[1]

    for iter in range(num_of_pics):
        image_raw = synthetic_dataset.generate_background(size=(row,column))
        # image_raw = synthetic_dataset.generate_pure_background()
        image = image_raw.copy()
        points,col = synthetic_dataset.draw_polygon(image)
        os.makedirs(os.path.join(img_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(points_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(corner_img_save_path,str(iter)),exist_ok=True)

        for i in range(2): 
            image = image_raw.copy() 
            corner_img = np.zeros([row,column]) #创建空白的帧角点图
            # corner_img = image_raw.copy() #保留原背景
            points = synthetic_dataset.move_polygon(image,points,col)

            for point in points:
                if point[0] >= 0 and point[0] < column and point[1] >= 0 and point[1] < row: 
                    # corner_img[int(point[1]),int(point[0])] = col #标记帧角点
                    corner_img[int(point[1]),int(point[0])] = 255 #标记帧角点

            cv2.imwrite(os.path.join(corner_img_save_path,str(iter),"{}.png".format(i)),corner_img) #画出帧角点图
            cv2.imwrite(os.path.join(img_save_path,str(iter),"{}.png".format(i)),image)
            np.savetxt(os.path.join(points_save_path,str(iter),"{}.txt".format(i)),points)
            
    print("syn_polygon finished!")

def syn_multiple_polygons(img_save_path,points_save_path,corner_img_save_path,num_of_pics=100,img_size = [260,346]):
    os.makedirs(img_save_path,exist_ok=True)
    os.makedirs(points_save_path,exist_ok=True) 
    os.makedirs(corner_img_save_path,exist_ok=True) 
    row = img_size[0]
    column = img_size[1]

    for iter in range(num_of_pics):
        image_raw = synthetic_dataset.generate_background(size=(row,column))
        # image_raw = synthetic_dataset.generate_pure_background()
        image = image_raw.copy()
        points_list,cols_list = synthetic_dataset.draw_multiple_polygons(image)
        os.makedirs(os.path.join(img_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(points_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(corner_img_save_path,str(iter)),exist_ok=True)

        for i in range(2): 
            image = image_raw.copy()
            corner_img = np.zeros([row,column]) #创建空白的帧角点图
            # corner_img = image_raw.copy() #保留原背景
            new_points = []
            for points,col in zip(points_list,cols_list):
                temp_points = synthetic_dataset.move_polygon(image,points,col)
                for point in temp_points:
                    if point[0] >= 0 and point[0] < column and point[1] >= 0 and point[1] < row: 
                        # corner_img[int(point[1]),int(point[0])] = col #标记帧角点
                        corner_img[int(point[1]),int(point[0])] = 255 #标记帧角点
                new_points = np.append(new_points,temp_points)
            new_points = np.reshape(new_points,(-1,2))
            

            cv2.imwrite(os.path.join(corner_img_save_path,str(iter),"{}.png".format(i)),corner_img) #画出帧角点图
            cv2.imwrite(os.path.join(img_save_path,str(iter),"{}.png".format(i)),image)
            np.savetxt(os.path.join(points_save_path,str(iter),"{}.txt".format(i)),new_points)
            
    print("syn_multiple_polygon finished!")

def syn_lines(img_save_path,points_save_path,corner_img_save_path,num_of_pics=100,img_size = [260,346]):
    os.makedirs(img_save_path,exist_ok=True)
    os.makedirs(points_save_path,exist_ok=True) 
    os.makedirs(corner_img_save_path,exist_ok=True) 
    row = img_size[0]
    column = img_size[1]

    for iter in range(num_of_pics):
        image_raw = synthetic_dataset.generate_background(size=(row,column))
        # image_raw = synthetic_dataset.generate_pure_background()
        image = image_raw.copy()
        points_list,cols_list,thicknesses_list = synthetic_dataset.draw_lines(image)
        os.makedirs(os.path.join(img_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(points_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(corner_img_save_path,str(iter)),exist_ok=True)

        for i in range(2): 
            image = image_raw.copy()
            new_points = []
            corner_img = np.zeros([row,column]) #创建空白的帧角点图
            # corner_img = image_raw.copy() #保留原背景
            for points,col,thickness in zip(points_list,cols_list,thicknesses_list):
                temp_points = synthetic_dataset.move_line(image,points[0],points[1],col,thickness)
                for point in temp_points:
                    if point[0] >= 0 and point[0] < column and point[1] >= 0 and point[1] < row: 
                        # corner_img[int(point[1]),int(point[0])] = col #标记帧角点
                        corner_img[int(point[1]),int(point[0])] = 255 #标记帧角点
                new_points = np.append(new_points,temp_points)
            new_points = np.reshape(new_points,(-1,2))

                    
            cv2.imwrite(os.path.join(corner_img_save_path,str(iter),"{}.png".format(i)),corner_img) #画出帧角点图
            cv2.imwrite(os.path.join(img_save_path,str(iter),"{}.png".format(i)),image)
            np.savetxt(os.path.join(points_save_path,str(iter),"{}.txt".format(i)),new_points)
            
    print("syn_lines finished!")

def syn_ellipses(img_save_path,points_save_path,corner_img_save_path,num_of_pics=100,img_size = [260,346]):
    os.makedirs(img_save_path,exist_ok=True)
    os.makedirs(points_save_path,exist_ok=True) 
    os.makedirs(corner_img_save_path,exist_ok=True) 
    row = img_size[0]
    column = img_size[1]

    for iter in range(num_of_pics):
        image_raw = synthetic_dataset.generate_background(size=(row,column))
        # image_raw = synthetic_dataset.generate_pure_background()
        image = image_raw.copy()
        points_list,cols_list,pram_list = synthetic_dataset.draw_ellipses(image)
        os.makedirs(os.path.join(img_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(points_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(corner_img_save_path,str(iter)),exist_ok=True) 

        for i in range(2): 
            image = image_raw.copy()
            corner_img = np.zeros([row,column]) #创建空白的帧角点图
            # corner_img = image_raw.copy() #保留原背景
            new_points = []
            for col,pram in zip(cols_list,pram_list):
                temp_points = synthetic_dataset.move_ellipses(image,col,pram)
                for point in temp_points:
                    if point[0] >= 0 and point[0] < column and point[1] >= 0 and point[1] < row: 
                        # corner_img[int(point[1]),int(point[0])] = col #标记帧角点
                        corner_img[int(point[1]),int(point[0])] = 255 #标记帧角点
                new_points = np.append(new_points,temp_points)
            new_points = np.reshape(new_points,(-1,2))


            cv2.imwrite(os.path.join(corner_img_save_path,str(iter),"{}.png".format(i)),corner_img) #画出帧角点图
            cv2.imwrite(os.path.join(img_save_path,str(iter),"{}.png".format(i)),image)
            np.savetxt(os.path.join(points_save_path,str(iter),"{}.txt".format(i)),new_points)
            
    print("syn_ellipses finished!")

def syn_star(img_save_path,points_save_path,corner_img_save_path,num_of_pics=100,img_size = [260,346]):
    os.makedirs(img_save_path,exist_ok=True)
    os.makedirs(points_save_path,exist_ok=True) 
    os.makedirs(corner_img_save_path,exist_ok=True) 
    row = img_size[0]
    column = img_size[1]

    for iter in range(num_of_pics):
        image_raw = synthetic_dataset.generate_background(size=(row,column))
        # image_raw = synthetic_dataset.generate_pure_background()
        image = image_raw.copy()
        points_list,cols_list,thicknesses_list = synthetic_dataset.draw_star(image)
        
        os.makedirs(os.path.join(img_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(points_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(corner_img_save_path,str(iter)),exist_ok=True)

        for i in range(2): 
            image = image_raw.copy()
            corner_img = np.zeros([row,column]) #创建空白的帧角点图
            # corner_img = image_raw.copy() #保留原背景
            new_points = synthetic_dataset.move_star(image,points_list,cols_list,thicknesses_list)
            for point in new_points:
                if point[0] >= 0 and point[0] < column and point[1] >= 0 and point[1] < row: 
                    corner_img[int(point[1]),int(point[0])] = 255 #标记帧角点

            cv2.imwrite(os.path.join(corner_img_save_path,str(iter),"{}.png".format(i)),corner_img) #画出帧角点图
            cv2.imwrite(os.path.join(img_save_path,str(iter),"{}.png".format(i)),image)
            np.savetxt(os.path.join(points_save_path,str(iter),"{}.txt".format(i)),new_points)
            
    print("syn_star finished!")

def syn_checkboard(img_save_path,points_save_path,corner_img_save_path,num_of_pics=100,img_size = [260,346]):
    os.makedirs(img_save_path,exist_ok=True)
    os.makedirs(points_save_path,exist_ok=True) 
    os.makedirs(corner_img_save_path,exist_ok=True) 
    row = img_size[0]
    column = img_size[1]

    for iter in range(num_of_pics):
        image_raw = synthetic_dataset.generate_background(size=(row,column))
        # image_raw = synthetic_dataset.generate_pure_background()
        image = [image_raw.copy(),image_raw.copy()]
    
        points_list = synthetic_dataset.draw_checkerboard_twice(image[0],image[1])
        
        os.makedirs(os.path.join(img_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(points_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(corner_img_save_path,str(iter)),exist_ok=True)

        for i in range(2): 
            corner_img = np.zeros([row,column]) #创建空白的帧角点图
            # corner_img = image_raw.copy() #保留原背景
            for point in points_list[i]:
                if point[0] >= 0 and point[0] < column and point[1] >= 0 and point[1] < row: 
                    corner_img[int(point[1]),int(point[0])] = 255 #标记帧角点
            cv2.imwrite(os.path.join(corner_img_save_path,str(iter),"{}.png".format(i)),corner_img) #画出帧角点图
            cv2.imwrite(os.path.join(img_save_path,str(iter),"{}.png".format(i)),image[i])
            np.savetxt(os.path.join(points_save_path,str(iter),"{}.txt".format(i)),points_list[i])
            
    print("syn_checkboard finished!")    

def syn_stripes(img_save_path,points_save_path,corner_img_save_path,num_of_pics=100,img_size = [260,346]):
    os.makedirs(img_save_path,exist_ok=True)
    os.makedirs(points_save_path,exist_ok=True) 
    os.makedirs(corner_img_save_path,exist_ok=True) 
    row = img_size[0]
    column = img_size[1]

    for iter in range(num_of_pics):
        image_raw = synthetic_dataset.generate_background(size=(row,column))
        # image_raw = synthetic_dataset.generate_pure_background()
        image = [image_raw.copy(),image_raw.copy()]
    
        points_list = synthetic_dataset.draw_stripes_twice(image[0],image[1])
        
        os.makedirs(os.path.join(img_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(points_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(corner_img_save_path,str(iter)),exist_ok=True)

        for i in range(2): 
            corner_img = np.zeros([row,column]) #创建空白的帧角点图
            # corner_img = image_raw.copy() #保留原背景
            for point in points_list[i]:
                if point[0] >= 0 and point[0] < column and point[1] >= 0 and point[1] < row: 
                    corner_img[int(point[1]),int(point[0])] = 255 #标记帧角点
            cv2.imwrite(os.path.join(corner_img_save_path,str(iter),"{}.png".format(i)),corner_img) #画出帧角点图
            cv2.imwrite(os.path.join(img_save_path,str(iter),"{}.png".format(i)),image[i])
            np.savetxt(os.path.join(points_save_path,str(iter),"{}.txt".format(i)),points_list[i])
            
    print("syn_stripes finished!")

def syn_cube(img_save_path,points_save_path,corner_img_save_path,num_of_pics=100,img_size = [260,346]):
    os.makedirs(img_save_path,exist_ok=True)
    os.makedirs(points_save_path,exist_ok=True)
    os.makedirs(corner_img_save_path,exist_ok=True) 
    row = img_size[0]
    column = img_size[1] 
    

    for iter in range(num_of_pics):
        image_raw = synthetic_dataset.generate_background(size=(row,column))
        # image_raw = synthetic_dataset.generate_pure_background()
        image = [image_raw.copy(),image_raw.copy()]
    
        points_list = synthetic_dataset.draw_cube_twice(image[0],image[1])
        
        os.makedirs(os.path.join(img_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(points_save_path,str(iter)),exist_ok=True)
        os.makedirs(os.path.join(corner_img_save_path,str(iter)),exist_ok=True)

        for i in range(2): 
            corner_img = np.zeros([row,column]) #创建空白的帧角点图
            # corner_img = image_raw.copy() #保留原背景   
            for point in points_list[i]:
                if point[0] >= 0 and point[0] < column and point[1] >= 0 and point[1] < row: 
                    corner_img[int(point[1]),int(point[0])] = 255 #标记帧角点
            cv2.imwrite(os.path.join(corner_img_save_path,str(iter),"{}.png".format(i)),corner_img) #画出帧角点图
            cv2.imwrite(os.path.join(img_save_path,str(iter),"{}.png".format(i)),image[i])
            np.savetxt(os.path.join(points_save_path,str(iter),"{}.txt".format(i)),points_list[i])
            
    print("syn_cube finished!")          

# def corner_img(corners_root,corner_img_save_root):
#     for path,dirs,files in os.walk(corners_root):
#         if len(dirs) == 0 and len(files) != 0:
#             np.loadtxt(files)

#             corner_dir = path
#             corner_img_dir = os.path.join(corner_img_save_root,os.path.relpath(corner_dir,corners_root))
#             os.makedirs(corner_img_dir)
#             cv2.imwrite(os.path.join(corner_img_dir,"{}.png".format()),points)


if __name__ == "__main__":
    syn_polygon(img_save_path,points_save_path,corner_img_save_path,3)

