import PIL
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import rootutils
import torch
from yolo5face.get_model import get_model
import numpy as np
import cv2
import csv
import os
import faceBlendCommon as fbc
from models.model_module import Model_Module
from models.components.model import model

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

VISUALIZE_FACE_POINTS = False

dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

filters_config = {
    'anonymous':
        [{'path': "/filter/anonymous.png",
         'anno_path': "/filter/anonymous-annotated.csv",
         'morph': True, 'animated': False, 'has_alpha': True}],
    'anime':
        [{'path': "/filter/anime.png",
         'anno_path': "/filter/anime-annotated.csv",
         'morph': True, 'animated': False, 'has_alpha': True}],
    'jason-joker':
        [{'path': "/filter/jason-joker.png",
          'anno_path': "/filter/jason-joker-annotated.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'squid_game_front_man':
        [{'path': "/filter/squid_game_front_man.png",
          'anno_path': "/filter/squid_game_front_man.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'cat':
        [{'path': "/filter/cat-ears.png",
          'anno_path': "/filter/cat-ears-annotated.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
        
        {'path': "/filter/cat-nose.png",
          'anno_path': "/filter/cat-nose-annotated.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'dog':
        [{'path': "/filter/dog-ears.png",
          'anno_path': "/filter/dog-ears-annotated.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

         {'path': "/filter/dog-nose.png",
          'anno_path': "/filter/dog-nose-annotated.csv",
          'morph': False, 'animated': False, 'has_alpha': True}]
}

def get_keypoints(model, face_detector, img_path):
    transform = Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    image = PIL.Image.open(img_path)
    image = image.convert("RGB")
    input = np.array(image)
    boxes, _, _ = face_detector(input, target_size=256)
    num_faces = len(boxes)
    keypoints = []
    
    for i in range(num_faces):
        x_min, y_min, x_max, y_max = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        
        cropped_image = np.array(image.crop((x_min, y_min, x_max, y_max)))

        if len(cropped_image.shape) == 2:
            cropped_image = cropped_image[:,:, np.newaxis]
            cropped_image = np.concatenate([cropped_image] * 3, axis = -1)

        width, height, _ = cropped_image.shape

        detransform = Compose([
            A.Resize(width, height),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        transformed = transform(image=cropped_image)

        cropped_image = transformed["image"]
        
        input = cropped_image[None].to("cuda")

        output = model(input)
        points = torch.Tensor(output.cpu().reshape((68, 2))).to(torch.float32).detach().numpy()
        points = (points + 0.5) * np.array([224, 224])

        detransformed = detransform(image=np.zeros((224, 224)), keypoints=points)
        points = detransformed["keypoints"]
        points = points + np.array([x_min, y_min])
        keypoints.append(points)

    return num_faces, keypoints

def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    alpha = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) + 255

    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))
    
    return img, alpha

def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            if i > 67:
                break
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[int(row[0])] = (x, y)
            except ValueError:
                continue
        return points
    
def find_convex_hull(points):
    hull = []
    hullIndex = np.array(list(points.keys())).reshape(-1, 1)
    for idx in range(len(hullIndex)):
        hull.append(points[hullIndex[idx][0]])

    return hull, hullIndex

def load_filter(filter_name="anonymous"):
 
    filters = filters_config[filter_name]
 
    multi_filter_runtime = []
 
    for filter in filters:
        temp_dict = {}
 
        img1, img1_alpha = load_filter_img(dir + filter['path'], filter['has_alpha'])
        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha
 
        points = load_landmarks(dir + filter['anno_path'])
 
        temp_dict['points'] = points
 
        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)

            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)

            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue
 
        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap
 
        multi_filter_runtime.append(temp_dict)
 
    return filters, multi_filter_runtime

ckpt_path = dir + "/logs/train/runs/final/checkpoints/best.ckpt"

detector = Model_Module.load_from_checkpoint(net=model(), checkpoint_path=ckpt_path)
detector.eval()

face_detector = get_model("yolov5n", device='cuda', min_face=10)

img_path = "C:/Users/NXBGDVN/Pictures/A19A0109.jpg"

num_faces, keypoints = get_keypoints(detector, face_detector, img_path)

image = cv2.imread(img_path)

if VISUALIZE_FACE_POINTS:
    for i in range(num_faces):
        for (x, y) in keypoints[i]:
            point = (int(x), int(y))
            cv2.circle(image, point, 10, (255, 0, 0), -1)
    height, width, _ = image.shape
    down_points = (int(width*0.2), int(height*0.2))
    image = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
    cv2.imshow("landmarks", image)


output = np.zeros(image.shape)
temp2 = image

filters, multi_filter_runtime = load_filter("anonymous")

for id in range(num_faces):
    for idx, filter in enumerate(filters):
        filter_runtime = multi_filter_runtime[idx]
        img1 = filter_runtime['img']
        points1 = filter_runtime['points']
        img1_alpha = filter_runtime['img_a']

        if filter['morph']:
            hullIndex = filter_runtime['hullIndex']
            dt = filter_runtime['dt']
            hull1 = filter_runtime['hull']

            # create copy of frame
            warped_img = np.copy(image)

            # Find convex hull
            hull2 = []
            for j in range(0, len(hullIndex)):
                x, y = keypoints[id][hullIndex[j][0]]
                x, y = max(x, 0), max(y, 0)
                x, y = min(x, warped_img.shape[1] - 1), min(y, warped_img.shape[0] - 1)
                hull2.append((x, y))
            
            mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
            mask1 = cv2.merge((mask1, mask1, mask1))
            img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))
            
            # Warp the triangles
            for i in range(0, len(dt)):
                t1 = []
                t2 = []

                for j in range(0, 3):
                    t1.append(hull1[dt[i][j]])
                    t2.append(hull2[dt[i][j]])

                fbc.warpTriangle(img1, warped_img, t1, t2)
                fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)
    
            # Blur the mask before blending
            mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
            mask2 = (255.0, 255.0, 255.0) - mask1
    
            # Perform alpha blending of the two images
            temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
            temp2 = np.multiply(temp2, (mask2 * (1.0 / 255)))
            output = output + temp1
            
        else:
            dst_points = [keypoints[id][int(list(points1.keys())[0])], keypoints[id][int(list(points1.keys())[1])]]
            tform = fbc.similarityTransform(list(points1.values()), dst_points)
            # Apply similarity transform to input image
            trans_img = cv2.warpAffine(img1, tform, (image.shape[1], image.shape[0]))
            trans_alpha = cv2.warpAffine(img1_alpha, tform, (image.shape[1], image.shape[0]))
            mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

            # Blur the mask before blending
            mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
    
            mask2 = (255.0, 255.0, 255.0) - mask1
    
            # Perform alpha blending of the two images
            temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
            temp2 = np.multiply(temp2, (mask2 * (1.0 / 255)))
            output = output + temp1
        
    if id == num_faces - 1:
        output = output + temp2

frame = output = np.uint8(output)
height, width, _ = frame.shape
down_points = (int(width*0.2), int(height*0.2))
frame = cv2.resize(frame, down_points, interpolation= cv2.INTER_LINEAR)

cv2.imshow("Face Filter", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
