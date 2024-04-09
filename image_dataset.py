from torch.utils.data import Dataset
from PIL import Image

import os
import numpy as np



class ImageDataSet(Dataset):
    def __init__(self, root_dir, class_number, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.org_dir=os.path.join(root_dir,'org-img')
        self.label_dir=os.path.join(root_dir,'label-img')
        
        init_org_images = os.listdir(self.org_dir)       
        init_label_images = os.listdir(self.label_dir)

        self.org_images = []
        self.label_images = []


        for image in sorted(init_org_images):
            try:
                im = Image.open(os.path.join(self.org_dir, image))
                self.org_images.append(image)
            except IOError:
                continue

  
        for image in sorted(init_label_images):
            try:
                im = Image.open(os.path.join(self.label_dir, image)).convert("RGB")
                im = np.array(im)

                if class_number in im:
                    self.label_images.append(1)
                else:
                    self.label_images.append(0)

            except IOError:
                continue

        # print(f"# of Org Images: {len(self.org_images)}")
        # print(f"# of Label Images: {len(self.label_images)}")
        

    def __len__(self):
        return len(self.org_images)

    def __getitem__(self, idx):
        original_img_name = self.org_images[idx]
        label_image = self.label_images[idx]

        org_img_path = os.path.join(self.org_dir, original_img_name)

        original_image = Image.open(org_img_path).convert("RGB")

        if self.transform:
            original_image = self.transform(original_image)

        return original_image, label_image