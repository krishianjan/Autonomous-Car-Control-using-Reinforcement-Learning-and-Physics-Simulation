import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

class NuScenesDetectionDataset(Dataset):
    """PyTorch Dataset for 2D object detection on nuScenes camera images."""
    def __init__(self, nusc, split='train', transform=None, target_size=(800, 800)):
        """
        Args:
            nusc: NuScenes object (already loaded).
            split: 'train' or 'val' (uses scene splits from nuScenes).
            transform: albumentations or torchvision transforms.
            target_size: (height, width) for resizing images.
        """
        self.nusc = nusc
        self.split = split
        self.transform = transform
        self.target_size = target_size

        # Get list of scene tokens for this split
        self.scene_tokens = []
        for scene in nusc.scene:
            if scene['name'].split('-')[0] in ['scene', 'scene-0061']:  # mini has only train scenes; adjust if needed
                self.scene_tokens.append(scene['token'])
        # For mini, we'll manually use first 4 scenes for train, last 1 for val
        # Better: use official splits? Mini dataset doesn't have official splits, so we create a simple split.
        # We'll use first 4 scenes for train, last for val.
        train_scenes = self.scene_tokens[:4]
        val_scenes = self.scene_tokens[4:5] if len(self.scene_tokens) > 4 else []
        if split == 'train':
            self.scene_tokens = train_scenes
        else:
            self.scene_tokens = val_scenes

        # Build list of all samples (camera images) in these scenes
        self.samples = []
        for scene_token in self.scene_tokens:
            scene = nusc.get('scene', scene_token)
            sample_token = scene['first_sample_token']
            while sample_token:
                sample = nusc.get('sample', sample_token)
                # We'll use the front camera (CAM_FRONT)
                self.samples.append(sample)
                sample_token = sample['next']
        print(f"Loaded {len(self.samples)} samples for {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Get camera image
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        img_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_h, original_w = image.shape[:2]

        # Get annotations (bounding boxes) for this sample
        boxes = []
        labels = []
        ann_tokens = sample['anns']
        for ann_token in ann_tokens:
            ann = self.nusc.get('sample_annotation', ann_token)
            # Only consider visible objects (attribute not necessary)
            # Get category: we'll map to COCO classes: 1: vehicle, 2: pedestrian, 3: cyclist
            category = ann['category_name']
            if 'vehicle' in category:
                label = 1
            elif 'pedestrian' in category:
                label = 2
            elif 'bicycle' in category:
                label = 3
            else:
                continue  # ignore other objects

            # Get 3D bounding box and project to 2D (using camera intrinsics)
            # nuScenes provides 2D bbox in 'bbox' field? Actually ann has 'bbox' (2D) already? Let's check.
            # In nuScenes, sample_annotation has 'bbox' which is [x1, y1, x2, y2] in image coordinates (0-indexed, top-left to bottom-right).
            # But it's for the annotated object. We'll use that.
            bbox = ann['bbox']  # [x1, y1, x2, y2]
            # Ensure bbox is within image bounds
            x1 = max(0, bbox[0])
            y1 = max(0, bbox[1])
            x2 = min(original_w, bbox[2])
            y2 = min(original_h, bbox[3])
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(label)

        # Resize image and boxes to target size
        image_resized = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        scale_x = self.target_size[1] / original_w
        scale_y = self.target_size[0] / original_h
        boxes_resized = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            boxes_resized.append([x1, y1, x2, y2])

        # Convert to tensor
        image_tensor = torch.from_numpy(image_resized).permute(2,0,1).float() / 255.0
        boxes_tensor = torch.tensor(boxes_resized, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx]),
            'area': (boxes_tensor[:,2] - boxes_tensor[:,0]) * (boxes_tensor[:,3] - boxes_tensor[:,1]),
            'iscrowd': torch.zeros((len(boxes_tensor),), dtype=torch.int64)
        }
        return image_tensor, target
