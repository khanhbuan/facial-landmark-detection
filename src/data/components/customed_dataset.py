import os
import xml.etree.ElementTree as ET
from scipy import io
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A

dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

class Customed_Dataset(Dataset):
    def __init__(self, is_train=True, transform = None):
        if is_train:
            path = dir + "/data/300W/labels_ibug_300W_train.xml"
        else:
            path = dir + "/data/300W/labels_ibug_300W_test.xml"

        self.images_path, self.key_points = self.data_extractor(path = path)
        self.bounding_boxes = self.bbox_extractor()

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, index):
        img_path = self.images_path[index] 
        keypoints = self.key_points[index]

        if img_path not in self.bounding_boxes.keys():
            temp_path = img_path[:-11] + ".png"
            if not os.path.isfile(temp_path):
                temp_path = img_path[:-11] + ".jpg"
            temp_img = cv2.imread(temp_path)
            bbox = self.bounding_boxes[temp_path]

            transform = A.Compose([
                A.HorizontalFlip(p=1),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

            class_labels = ['face']
            trans = []
            trans.append(bbox)   

            transformed = transform(image = temp_img, bboxes=trans, class_labels=class_labels)

            bbox2 = transformed["bboxes"]
            bbox2 = np.array(bbox2[0])
            bbox = bbox2
        
        else:
            bbox = self.bounding_boxes[img_path]
        
        return img_path, keypoints, bbox
    
    def data_extractor(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        images_path = []
        key_points = []
        for image in root.findall(".//image"):
            images_path.append(dir + "/data/300W/" + image.get("file"))
            kp = []
            parts = image.find("box").findall("part")
            for part in parts:
                kp.append(np.array([part.get("x"), part.get("y")], dtype = np.float32))
            key_points.append(kp)

        return images_path, key_points
    
    def bbox_extractor(self):
        paths = ["ibug", "helen_testset", "helen_trainset", "afw", "lfpw_testset", "lfpw_trainset"]
        dr = ["ibug/", "helen/testset/", "helen/trainset/", "afw/", "lfpw/testset/", "lfpw/trainset/"]
        bounding_boxes = {}
        for j in range(len(paths)):
            path = dir + "/data/Bounding Boxes/bounding_boxes_" + paths[j] + ".mat"
            mat = io.loadmat(path)
            for i in range(len(mat['bounding_boxes'][0])):
                img_path = mat['bounding_boxes'][0][i].item()[0].item()
                img_path = dir + "/data/300W/" + dr[j] + img_path
                bbox = mat['bounding_boxes'][0][i].item()[2][0]
                bounding_boxes[img_path] = bbox
        return bounding_boxes
    
if __name__ == "__main__":
    dataset = Customed_Dataset()