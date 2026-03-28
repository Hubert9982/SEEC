import os
from torch.utils.data import Dataset
import PIL.Image as Image
import torchvision
import numpy as np


class ImgMaskDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_path = os.path.join(root, "images")
        self.mask_path = os.path.join(root, "masks")
        self.imgs = os.listdir(self.img_path)
        self.imgs.sort()
        self.masks = os.listdir(self.mask_path)
        self.masks.sort()

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.imgs[index])
        mask_path = os.path.join(self.mask_path, self.masks[index])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)


class CityscapesDataset(torchvision.datasets.Cityscapes):
    """
    Class Definitions


    Group	Classes
    flat	road · sidewalk · parking+ · rail track+
    human	person* · rider*
    vehicle	car* · truck* · bus* · on rails* · motorcycle* · bicycle* · caravan*+ · trailer*+
    construction	building · wall · fence · guard rail+ · bridge+ · tunnel+
    object	pole · pole group+ · traffic sign · traffic light
    nature	vegetation · terrain
    sky	    sky
    void	ground+ · dynamic+ · static+
    """

    def __init__(
        self,
        root,
        num_classes=8,
        split="train",
        mode="fine",
        target_type="instance",
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        super().__init__(root, split, mode, target_type, transform, target_transform, transforms)
        self.num_classes = num_classes
        self.area_by_cat_id = [1, 2, 4, 0, 7, 5, 3, 6] # from cal_area_city
        self.num_cat = len(self.area_by_cat_id)

        if self.num_classes > self.num_cat or num_classes < 1:
            raise ValueError

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert("RGB")

        targets = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])  # type: ignore[assignment]

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]  # type: ignore[assignment]

        # mask the target given the num_classes
        if self.num_classes < self.num_cat:
            target_data = np.array(target)
            target_data = target_data.clip(0, self.num_classes - 1)
            target = Image.fromarray(target_data)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
