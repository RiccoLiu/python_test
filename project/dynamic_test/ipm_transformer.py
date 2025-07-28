import os 
import cv2
import math
import argparse
import numpy as np

from loguru import logger

class IpmTransformer():
    def __init__(self, ipm_range, reso, start_forward = 1.0, is_distort=True):
        '''
            相机标定参数
        '''
        self.camera_matrix_ = np.array([
            [1070.3158, 0, 961.0000],
            [0,  1118.2645, 542.0000],
            [0, 0, 1]
        ])
        self.dist_coeffs_ = np.array([-0.077837, -0.060080, -0.001426, 0.006125, 0.050293])  # k1, k2, p1, p2[, k3]
        self.pyr_ = [3.9700, -1.100,  0.00]
        self.camera2wheel_ = 511 # mm
        self.wheel2ground_ = 750 # mm

        self.img_width_ = 1920
        self.img_height_ = 1080

        self.map_x_ = np.zeros((self.img_height_, self.img_width_), dtype=np.float32)
        self.map_y_ = np.zeros((self.img_height_, self.img_width_), dtype=np.float32)
        
        self.map_x_, self.map_y_ = cv2.initUndistortRectifyMap(
            cameraMatrix=self.camera_matrix_,
            distCoeffs=self.dist_coeffs_,
            R=None,
            newCameraMatrix=self.camera_matrix_,
            size=(self.img_width_, self.img_height_),
            m1type=cv2.CV_32FC1,
            map1=self.map_x_,
            map2=self.map_y_
        ) 

        '''
            IPM配置参数        
        '''
        self.act_width_ = ipm_range[0]
        self.act_height_ = ipm_range[1]
        
        self.pix_width_ = int(self.act_width_ / reso)
        self.pix_height_ = int(self.act_height_ / reso)
        
        self.ipm_frame_map_ = self._get_ipm_frame_map(self.pix_width_, self.pix_height_, reso, start_forward, is_distort)

    def get_pixel_uv(self, p_world, is_distort=True):
        r_w2c = self._get_rotation_w2c()
        t_w2c = self._get_translation_w2c()

        p_w = np.array(p_world).reshape(3, 1)            
        p_c = r_w2c @ p_w + t_w2c
        
        p_c_norm = p_c / p_c[2]
        pt_uv = self.camera_matrix_ @ p_c_norm

        u_undist = int(round(pt_uv[0, 0]))
        v_undist = int(round(pt_uv[1, 0]))

        if not self._out_of_img(u_undist, v_undist):
            if not is_distort:
                return (u_undist, v_undist)
            else:
                u_distort = int(self.map_x_[v_undist, u_undist])
                v_distort = int(self.map_y_[v_undist, u_undist])
                
                if not self._out_of_img(u_distort, v_distort):
                    return (u_distort, v_distort)
    
        return None

    def get_ipm_img(self, img, skyline=1080):
        ipm_img = np.zeros((self.pix_height_, self.pix_width_, 3), dtype=np.uint8)

        for h in range(self.pix_height_):
            for w in range(self.pix_width_):
                v, u = self.ipm_frame_map_.get((h, w), (0, 0))                
                # logger.info(f'h:{h}, w:{w}, v:{v}, u:{u}, img.shape:{img.shape}')
                if (skyline > v >= 0) and (img.shape[1] > u >= 0):
                    if v == 0 and u == 0:
                        continue
                    ipm_img[h,w] = img[v,u]

        return ipm_img


    def _get_ipm_frame_map(self, frame_width, frame_height, reso, start_forward=1.0, is_distort=True):
        frame_map = {}
        for h in range(frame_height):
            for w in range(frame_width):
                x = (-frame_width / 2.0 + w) * reso
                y = 0
                z = (frame_height - h)*reso + start_forward

                uv = self.get_pixel_uv(p_world=[x, y, z], is_distort=is_distort)
                if uv:
                    frame_map[(h, w)] = (uv[1], uv[0])

        return frame_map

    def _get_rotation_c2w(self):
        r_x = np.array([
            [1, 0, 0],
            [0, np.cos(math.radians(self.pyr_[0])), -np.sin(math.radians(self.pyr_[0]))],
            [0, np.sin(math.radians(self.pyr_[0])), np.cos(math.radians(self.pyr_[0]))]
        ])
        r_y = np.array([
            [np.cos(math.radians(self.pyr_[1])), 0, np.sin(math.radians(self.pyr_[1]))],
            [0, 1, 0],
            [-np.sin(math.radians(self.pyr_[1])), 0, np.cos(math.radians(self.pyr_[1]))]
        ])
        r_z = np.array([
            [np.cos(math.radians(self.pyr_[2])), -np.sin(math.radians(self.pyr_[2])), 0],
            [np.sin(math.radians(self.pyr_[2])), np.cos(math.radians(self.pyr_[2])), 0],
            [0, 0, 1]
        ])
        return r_z @ r_y @ r_x

    def _get_translation_c2w(self):
        return np.array([0, -(self.camera2wheel_ + self.wheel2ground_) / 1000, 0]).reshape(3, 1)

    def _get_rotation_w2c(self):
        r_c2w = self._get_rotation_c2w()    
        return r_c2w.transpose()

    def _get_translation_w2c(self):
        r_w2c = self._get_rotation_w2c()
        t_c2w = self._get_translation_c2w()
        return -r_w2c @ t_c2w

    def _out_of_img(self, u, v):
        return u < 0 or u >= self.img_width_ or v < 0 or v >= self.img_height_


if __name__ == '__main__':
    img_file = "/home/lc/work/sdk/dynamic_test/together/11-04-47-2024-11-14-Nor_000120_000134_29.jpg"
    
    img = cv2.imread(img_file)

    camera_matrix = np.array([
        [1070.3158, 0, 961.0000],
        [0,  1118.2645, 542.0000],
        [0, 0, 1]
    ])

    dist_coeffs = np.array([-0.077837, -0.060080, -0.001426, 0.006125, 0.050293])  # k1, k2, p1, p2[, k3]

    undis_img = cv2.undistort(src=img, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
    
    trans_undis = IpmTransformer(ipm_range=(15, 10), reso=0.02, start_forward=1.0, is_distort=False)
    ipm_img_undis = trans_undis.get_ipm_img(undis_img)
    
    cv2.imwrite('ipm_img_undis.jpg', ipm_img_undis)

    trans = IpmTransformer(ipm_range=(15, 10), reso=0.02, start_forward=1.0, is_distort=True)
    ipm_img = trans.get_ipm_img(img)

    cv2.imwrite('ipm_img.jpg', ipm_img)

