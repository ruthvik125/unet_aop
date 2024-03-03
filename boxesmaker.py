import cv2
import os
import itk 
import numpy as np
import random 
image_dataset_path =  "C:\\Users\\Admin\\Downloads\\Pubic Symphysis-Fetal Head Segmentation and Angle of Progression\\Pubic Symphysis-Fetal Head Segmentation and Angle of Progression\\image_mha"
mask_dataset_path =  "C:\\Users\\Admin\\Downloads\\Pubic Symphysis-Fetal Head Segmentation and Angle of Progression\\Pubic Symphysis-Fetal Head Segmentation and Angle of Progression\\label_mha"
    
img_file_paths = [os.path.join(image_dataset_path, file) for file in os.listdir(image_dataset_path) if file.endswith(".mha")]
mask_file_paths = [os.path.join(mask_dataset_path, file) for file in os.listdir(mask_dataset_path) if file.endswith(".mha")]

def format_conv():
    for i in range(4000):
        itk_img = itk.imread(img_file_paths[i])
        itk_mask = itk.imread(mask_file_paths[i])
        
        file_name = os.path.basename(img_file_paths[i])
        name, extension = os.path.splitext(file_name)
        
        new_file_name = name + ".png"
        base = os.path.join("dataset","images")
        mask_base = os.path.join("dataset","masks")
        new_image_path=os.path.join(base,new_file_name)
        new_mask_path=os.path.join(mask_base,new_file_name)
        
        itk.imwrite(itk_img, new_image_path)
        itk.imwrite(itk_mask, new_mask_path)
        
def mask_vis():
        color_map = {0: [0,0,0],  # White color for 0
             1: [0, 0, 255],
             2:[255,255,255]
             }      # Red color for 1
        binary_image = cv2.imread("C:\\Users\\Admin\\unet_aop\\NEW_dataset\\masks\\1.jpg",0)
        colored_image = np.zeros((128,128, 3), dtype=np.uint8)
        for value, color in color_map.items():
            colored_image[binary_image == value] = color
            
        colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.blur(colored_image, (3,3))
        
        threshold=200
        canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        class_id=1
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        
        file_name = os.path.basename("C:\\Users\\Admin\\unet_aop\\NEW_dataset\\masks\\0.jpg")
        name,ext= os.path.splitext(file_name)
        name = f"dataset\\{name}.txt"
        f = open(name,"w")
        # Draw polygonal contour + bonding rects + circles
        for i in range(len(contours)):
            color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            cv2.drawContours(drawing, contours_poly, i, color)
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            if(boundRect[i][0]!=boundRect[i-1][0] and boundRect[i][1]!=boundRect[i-1][1]):
                c_x = boundRect[i][0]+(boundRect[i][2]/2)
                c_y = boundRect[i][1]+(boundRect[i][3]/2)
                c_x = c_x/256
                c_y= c_y/256
                w = boundRect[i][2]/256
                h = boundRect[i][2]/256
                f.write(f"{class_id} {c_x} {c_y} {w} {h} \n")
                class_id=class_id+1
        f.close()
        
        cv2.imshow('Contours', drawing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

def bounding_boxes_files():
    image_dataset_path="dataset\\images"
    mask_dataset_path =  "dataset\\masks"
    bounding_boxes_path = "dataset\\boxxes"
    
    img_file_paths = [os.path.join(image_dataset_path, file) for file in os.listdir(image_dataset_path) if file.endswith(".png")]
    mask_file_paths = [os.path.join(mask_dataset_path, file) for file in os.listdir(mask_dataset_path) if file.endswith(".png")]
    
    color_map = {0: [0,0,0],  # White color for 0
             1: [0, 0, 255],
             2:[255,255,255]
             }   
    # Red color for 1
    for k in range(4000):
        binary_image = cv2.imread(mask_file_paths[k],0)
        colored_image = np.zeros((256,256, 3), dtype=np.uint8)
        for value, color in color_map.items():
            colored_image[binary_image == value] = color
            
        colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.blur(colored_image, (3,3))
        
        threshold=130
        canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
        contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        class_id=1
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        
        file_name = os.path.basename(mask_file_paths[k])
        name,ext= os.path.splitext(file_name)
        name = f"dataset\\boxxes\\{name}.txt"
        f = open(name,"w")
        # Draw polygonal contour + bonding rects + circles
        for i in range(len(contours)):
            color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            cv2.drawContours(drawing, contours_poly, i, color)
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            if(boundRect[i][0]!=boundRect[i-1][0] and boundRect[i][1]!=boundRect[i-1][1]):
                c_x = boundRect[i][0]+(boundRect[i][2]/2)
                c_y = boundRect[i][1]+(boundRect[i][3]/2)
                c_x = c_x/256
                c_y= c_y/256
                w = boundRect[i][2]/256
                h = boundRect[i][2]/256
                f.write(f"{class_id} {c_x} {c_y} {w} {h} \n")
                class_id=class_id+1
        print(boundRect[0])
        f.close()
        
if __name__ == '__main__':
    mask_vis()