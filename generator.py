import open3d as o3d
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
import numpy as np
from virtual_camera import VirtualCamera
import cv2
import time
from pathlib import Path
cur_path=Path(__file__).parent
cur_path=str(cur_path)
## Read the Data (From obj file)
bpa_mesh = o3d.io.read_triangle_mesh(cur_path + '/model/BigConnecter.obj',True)
tri=np.asarray(bpa_mesh.triangles)
points_raw=np.asarray(bpa_mesh.vertices)
center_p=np.sum(points_raw,0)/len(points_raw)
rx=-np.pi/2
ry=0
rz=np.pi/2
R=o3d.geometry.get_rotation_matrix_from_xyz([rx,ry,rz])
bpa_mesh.rotate(R,center_p)
# o3d.visualization.draw_geometries([bpa_mesh])

## Setup a camera
rx=0
ry=np.pi
rz=0
x=1
y=1
z=570
W,H=2048,2048
focal=1024
tic=time.time()
cam = VirtualCamera(rx, ry, rz,x, y, z, focal*3.45e-6, [W, H])

## Create Mask
#  Get the triangle relationship
points=points_raw
print(points)
points=points.reshape(-1,3)
ans,cp = cam.project_world_to_pixel(points)
ans=ans.astype(int)

mask_bg=np.zeros((W,H,3))
mask_bg[:,:,:]=0
ans=ans[:,0:2]
for i in range(len(tri)):
    pt1=ans[tri[i][0]]
    pt2=ans[tri[i][1]]
    pt3=ans[tri[i][2]]
    tri_set=np.asarray([pt1,pt2,pt3])
    tri_set=tri_set.astype(int)
    tri_set=tri_set.reshape(-1,2)
    cv2.fillPoly(mask_bg,[tri_set],(0,255,0))

#  Calculate the mask
kernel = np.ones((3,3),np.uint8)
fushi = cv2.erode(mask_bg,kernel,iterations = 1)
mask=np.where(fushi[:,:,1]==255)

## Render model
camera_intrinsics=o3d.camera.PinholeCameraIntrinsic()
camera_intrinsics.set_intrinsics(W,H,focal,focal,W/2,H/2)
extrinsics=cam.RT_world_in_camera

render = rendering.OffscreenRenderer(2048, 2048)
material = rendering.Material()
material.base_color = [1, 1, 1, 1.0]
material.shader = "defaultLit"
material.albedo_img=o3d.io.read_image(cur_path + '/model/BigConnecter.jpg')
material.base_metallic=1.0
render.scene.add_geometry("pcd",bpa_mesh,material)
render.setup_camera(camera_intrinsics,extrinsics)
render.scene.scene.set_sun_light([0, 0.0, -1], [1, 1,1],
                                 3e6)
render.scene.scene.enable_sun_light(True)

## Read RGB and Depth image from open3d
img = np.asarray(render.render_to_image())
depth_img=np.asarray(render.render_to_depth_image())
depth_img=1-depth_img
depth_img=depth_img/np.amax(depth_img)*255
depth_img=depth_img.astype(int)
cv2.imwrite(cur_path + '/output/render_depth.png',255-depth_img)

img= cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
bg=cv2.imread(cur_path + '/background/bg1.jpeg')
bg=cv2.resize(bg,(2048,2048))
fuse=bg.copy()
fuse[mask[0],mask[1]]=img[mask[0],mask[1]]

cv2.imwrite(cur_path + '/output/render_rgb.jpg',fuse)
plt.axis('off')
plt.imshow(cv2.cvtColor(bg,cv2.COLOR_BGR2RGB))
