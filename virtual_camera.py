import numpy as np
import time
from numba import vectorize
from numba import jit

class VirtualCamera:
    def __init__(self, alpha, beta, gamma, Tx, Ty, Tz, focal, resolution) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.Tx = Tx
        self.Ty = Ty
        self.Tz = Tz
        self.focal = focal
        self.resolution = np.array(resolution)
        self.center = np.array(resolution)/2
        self.pixel_size_x = 3.45e-6
        self.pixel_size_y = 3.45e-6
        self.RT_camera_in_world = np.zeros((4, 4))
        self.RT_world_in_camera = np.zeros((4, 4))
        self.update()

    def update(self):
        Rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(self.alpha), np.sin(self.alpha), 0],
                       [0, -np.sin(self.alpha), np.cos(self.alpha), 0],
                       [0, 0, 0, 1]])
        Ry = np.array([[np.cos(self.beta), 0, -np.sin(self.beta), 0],
                       [0, 1, 0, 0],
                       [np.sin(self.beta), 0, np.cos(self.beta), 0],
                       [0, 0, 0, 1]])
        Rz = np.array([[np.cos(self.gamma), np.sin(self.gamma), 0, 0],
                      [-np.sin(self.gamma), np.cos(self.gamma), 0, 0],
                       [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        R1 = np.matmul(Rx, np.matmul(Ry, Rz))
        
        T1 = np.array([[1, 0, 0, -self.Tx],
                       [0, 1, 0, -self.Ty],
                       [0, 0, 1, -self.Tz],
                       [0, 0, 0, 1]])
        self.RT_world_in_camera = np.matmul(R1, T1)
    
    def project_world_to_camera(self, points):
        assert points.shape[1] == 3, 'The n points\' shape should be (n,3)'
        
        n =points.shape[0]
        points = points.T
        points = np.row_stack((points, np.ones((1, n))))
        points = self.RT_world_in_camera@points
        points = points.T
        
        
        return points[:, 0:3]

    def project_camera_to_image(self, points):
        assert points.shape[1] == 3, 'The n points\' shape should be (n,3)'
        n = points.shape[0]
        points = points.T
        z = points[2, :]
        # print(z)
        trans = np.array([[self.focal, 0, 0],
                          [0, self.focal, 0],
                          [0, 0, 1]])
        points = trans@points
        points = points/z
        points = points.T

        return points

    def project_image_to_pixel(self, points):
        assert points.shape[1] == 3, 'The n points\' shape should be (n,3)'
        n = points.shape[0]
        points = points.T
        trans = np.array([[1/self.pixel_size_x, 0, self.center[0]],
                          [0, 1/self.pixel_size_y, self.center[1]],
                          [0, 0, 1]])
        points = trans@points
        # points=np.matmul(trans,points)
        
        points = points.T
        
        return points
    def p2p(self,points):
        n =points.shape[0]
        points = points.T
        points = np.row_stack((points, np.ones((1, n))))
        points = self.RT_world_in_camera@points
        points=points[0:3,:]
        z = points[2, :]
        # print(z)
        trans = np.array([[self.focal, 0, 0],
                          [0, self.focal, 0],
                          [0, 0, 1]])
        points = trans@points
        points = points/z
        trans = np.array([[1/self.pixel_size_x, 0, self.center[0]],
                          [0, 1/self.pixel_size_y, self.center[1]],
                          [0, 0, 1]])
        # points = trans@points
        points=np.matmul(trans,points)
        points = points.T
        return points
    def project_world_to_pixel(self, points):
        # time0=time.time()
        camera_points = self.project_world_to_camera(points)
        # time1=time.time()
        # print(f'World to camera cost {time1-time0} seconds')
        image_points = self.project_camera_to_image(camera_points)
        # time2=time.time()
        # print(f'Camera to image cost {time2-time1} seconds')
        pixel_points = self.project_image_to_pixel(image_points)
        # time3=time.time()
        # print(f'Image to pixels cost {time3-time2} seconds')
        return pixel_points,camera_points

    def crop_pixel(self, points):
        W = self.resolution[0]
        H = self.resolution[1]

        # points = np.column_stack((points))
        
        index = np.where(points[:, 0] >= 0)
        index = np.squeeze(index)
        points = points[index, :]
        
        index = np.where(points[:, 0] <= W)
        index = np.squeeze(index)
        points = points[index, :]

        index = np.where(points[:, 1] >= 0)
        index = np.squeeze(index)
        points = points[index, :]

        index = np.where(points[:, 1] <= H)
        index = np.squeeze(index)
        points = points[index, :]

        points = points.astype(int)
        # index = points[:, 3]
        points = points[:, 0:2]
        return points
    def crop_pixel_general(self, points):
        W = self.resolution[0]
        H = self.resolution[1]

        

        index = np.where(points[:, 0] >= 0)
        index = np.squeeze(index)
        # points= points[index, :]
        points=np.take(points,index,0)

        
        index = np.where(points[:, 0] <= W)
        index = np.squeeze(index)
        points=np.take(points,index,0)

        index = np.where(points[:, 1] >= 0)
        index = np.squeeze(index)
        points=np.take(points,index,0)

        index = np.where(points[:, 1] <= H)
        index = np.squeeze(index)
        points=np.take(points,index,0)

        points = points.astype(int)
        points = points[:, 0:2]
        return points


if __name__ == "__main__":
    cam = VirtualCamera(np.pi/12, np.pi/6, np.pi/3,
                        1, 2, 3, 0.003, [1000, 1000])
    p = np.array([[1, 0, 0], [0, 1, 0],[0,0,0]])
    p = p.reshape(-1, 3)
    ans = (cam.project_world_to_pixel(p))
    print(ans)
    # Ans should be [[7.45539187e+02 1.24170659e+03 1.00000000e+00]
    # [1.96595426e+02 1.95919429e+03 1.00000000e+00]]

