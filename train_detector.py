import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from nuscenes.nuscenes import NuScenes
import sys
sys.path.append('src/perception')
from nuscenes_detection_dataset import NuScenesDetectionDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Paths
    data_root = './data/nuscenes'
    version = 'v1.0-mini'
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    # Create datasets
    train_dataset = NuScenesDetectionDataset(nusc, split='train', target_size=(640, 640))
    val_dataset = NuScenesDetectionDataset(nusc, split='val', target_size=(640, 640))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Model: Faster R-CNN with ResNet-50 FPN backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Modify number of classes (background + 3 classes: vehicle, pedestrian, cyclist)
    num_classes = 4  # background + 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer, milestones=[8, 12], gamma=0.1)

    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 20 == 0:
                print(f'Epoch {epoch}, Iter {i}, Loss: {losses.item():.4f}')

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} finished. Average loss: {avg_loss:.4f}')

        # Validation after each epoch
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                # We can compute mAP here (skip for brevity)
        print('Validation done.')

        # Save checkpoint
        torch.save(model.state_dict(), f'fasterrcnn_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
