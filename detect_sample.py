import torch
import cv2
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
import sys
sys.path.append('src/perception')
from nuscenes_detection_dataset import NuScenesDetectionDataset

def load_model(device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 4
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load('fasterrcnn_epoch_14.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

def visualize_predictions(image_tensor, predictions, threshold=0.5):
    image = image_tensor.permute(1,2,0).cpu().numpy()
    plt.imshow(image)
    ax = plt.gca()
    for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
        if score > threshold:
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            label_name = ['bg', 'vehicle', 'pedestrian', 'cyclist'][label]
            ax.text(x1, y1-5, f'{label_name}: {score:.2f}', color='red', fontsize=8)
    plt.axis('off')
    plt.show()

def main():
    data_root = './data/nuscenes'
    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=False)
    dataset = NuScenesDetectionDataset(nusc, split='val', target_size=(640,640))
    if len(dataset) == 0:
        print("No validation samples. Train first.")
        return
    image_tensor, target = dataset[0]  # get first sample
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])[0]
    visualize_predictions(image_tensor, prediction)

if __name__ == '__main__':
    main()
