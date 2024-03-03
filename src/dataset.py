from torch.utils.data import dataloader,Dataset
import cv2
import itk
import os

class SegmentationDataset(Dataset):
    def __init__(self,imagePaths,maskPaths,transforms):
        self.imagePaths=imagePaths
        self.maskPaths=maskPaths
        self.transforms=transforms
    

    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, index):
        imagepath= self.imagePaths[index]
        
        
        itk_img  = itk.imread(imagepath)
        itk_mask = itk.imread(self.maskPaths[index])
        file_name = os.path.basename(imagepath)
        name, extension = os.path.splitext(file_name)

# Changing the extension to ".png"
        new_file_name = name + ".png"
        base = os.path.join("dataset","train")
        new_image_path=os.path.join(base,"images",new_file_name)
        new_mask_path=os.path.join(base,"masks",new_file_name)


        itk.imwrite(itk_img, new_image_path)
        itk.imwrite(itk_mask,new_mask_path)

        img = cv2.imread(new_image_path)
        image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        msk = cv2.imread(new_mask_path)
        mask = cv2.cvtColor(msk,cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image,mask)

