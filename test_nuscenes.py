import os
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

# Path to your nuScenes mini dataset
nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=True)

# List available scenes
print("Number of scenes:", len(nusc.scene))

# Get the first scene
scene = nusc.scene[0]
print("First scene token:", scene['token'])

# Get the first sample in that scene
sample = nusc.get('sample', scene['first_sample_token'])
print("Sample token:", sample['token'])

# Load camera image (front camera)
cam_token = sample['data']['CAM_FRONT']
cam_data = nusc.get('sample_data', cam_token)
cam_filepath = os.path.join(nusc.dataroot, cam_data['filename'])

# Load image with matplotlib
img = plt.imread(cam_filepath)
plt.imshow(img)
plt.title('Front camera image')
plt.show()

# Load LiDAR point cloud
lidar_token = sample['data']['LIDAR_TOP']
lidar_data = nusc.get('sample_data', lidar_token)
lidar_filepath = os.path.join(nusc.dataroot, lidar_data['filename'])

# Load point cloud (binary)
import numpy as np
points = np.fromfile(lidar_filepath, dtype=np.float32).reshape(-1, 5)  # x,y,z,intensity,ring_index
print("Point cloud shape:", points.shape)

# Optional: visualize with open3d
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
o3d.visualization.draw_geometries([pcd])
