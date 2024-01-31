from torch.utils.data import Dataset
import PIL
import torch
import numpy as np

class Transformed_Dataset(Dataset):
    def __init__(
        self,
        dataset = None,
        transform = None,
    ):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        img_path, keypoints, bbox = self.dataset.__getitem__(index)
        image = PIL.Image.open(img_path)
        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        cropped_image = np.array(image.crop((x_min, y_min, x_max, y_max)))
        if len(cropped_image.shape) == 2:
            cropped_image = cropped_image[:,:, np.newaxis]
            cropped_image = np.concatenate([cropped_image] * 3, axis = -1)
        keypoints -= np.array([x_min, y_min])

        if self.transform:
            transformed = self.transform(image=cropped_image, keypoints=keypoints)
            cropped_image, keypoints = transformed["image"], transformed["keypoints"]

        keypoints = torch.Tensor(keypoints)
        _, width, height = cropped_image.shape
        keypoints = keypoints / np.array([width, height]) - 0.5
        keypoints = keypoints.reshape(keypoints.shape[0]*keypoints.shape[1])

        return cropped_image, keypoints.to(dtype=torch.float32)
    
if __name__ == "__main__":
    _ = Transformed_Dataset()