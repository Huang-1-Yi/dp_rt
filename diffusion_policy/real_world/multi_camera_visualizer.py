import time
import multiprocessing as mp
import numpy as np
import cv2
from threadpoolctl import threadpool_limits
from diffusion_policy.real_world.multi_realsense import MultiRealsense

class MultiCameraVisualizer(mp.Process):
    def __init__(self,
        realsense: MultiRealsense,
        row, col,
        window_name='Multi Cam Vis',
        vis_fps=60,
        fill_value=0,
        rgb_to_bgr=True,
        ):
        super().__init__()
        self.row = row
        self.col = col
        self.window_name = window_name
        self.vis_fps = vis_fps
        self.fill_value = fill_value
        self.rgb_to_bgr=rgb_to_bgr
        self.realsense = realsense
        # shared variables
        self.stop_event = mp.Event()

        self.pred = self.realsense.enable_sam2

    def start(self, wait=False):
        super().start()
    
    def stop(self, wait=False):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        pass

    def stop_wait(self):
        self.join()        
    
    def run(self):
        cv2.setNumThreads(1)
        threadpool_limits(1)
        channel_slice = slice(None)
        if self.rgb_to_bgr:
            channel_slice = slice(None,None,-1)

        vis_data = None
        vis_img = None
        if self.pred:
            while not self.stop_event.is_set():
                vis_data = self.realsense.get_vis(out=vis_data)
                color = vis_data['color']               # 原始图像
                color_mask = vis_data['color_mask']     # 预测图像单通道灰度图 (N,H,W)
                N, H, W, C = color.shape
                assert C == 3

                # mask = color_mask[-1,:,:].copy()
                # vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为3通道BGR图像
                # vis_img = vis_img[:, :, ::-1].copy()  # 转为RGB格式（仅当显示库需要时）
                # 调整画布尺寸：高度不变，宽度加倍（原始+预测）
                oh = H * self.row
                ow = W * self.col * 2  
                if vis_img is None:
                    vis_img = np.full((oh, ow, 3), 
                        fill_value=self.fill_value, dtype=np.uint8)
                for row in range(self.row):
                    for col in range(self.col):
                        idx = col + row * self.col
                        h_start = H * row
                        h_end = h_start + H
                        w_start = W * col
                        w_end = w_start + W
                        # 预测图像位置（右侧）
                        w_start_mask = w_end
                        w_end_mask = w_start_mask + W

                        if idx < N:
                            # opencv uses bgr
                            vis_img[h_start:h_end,w_start:w_end
                                    ] = color[idx,:,:,channel_slice]
                            # 处理预测图像（单通道灰度图转3通道）
                            pred_vis = cv2.cvtColor(color_mask[idx], cv2.COLOR_GRAY2BGR)
                            pred_vis = cv2.resize(pred_vis, (W, H))
                            if not self.rgb_to_bgr:  # 如果需要RGB格式
                                pred_vis = pred_vis[:, :, ::-1].copy()
                            vis_img[h_start:h_end, w_start_mask:w_end_mask
                                    ] = pred_vis
                cv2.imshow(self.window_name, vis_img)
                cv2.pollKey()
                time.sleep(1 / self.vis_fps)
        else:

            while not self.stop_event.is_set():
                vis_data = self.realsense.get_vis(out=vis_data)
                color = vis_data['color']
                # color = cv2.resize(
                #     vis_data['color'], 
                #     (640, 480)  # (width, height)
                # )
                N, H, W, C = color.shape
                assert C == 3
                oh = H * self.row
                ow = W * self.col
                if vis_img is None:
                    vis_img = np.full((oh, ow, 3), 
                        fill_value=self.fill_value, dtype=np.uint8)
                for row in range(self.row):
                    for col in range(self.col):
                        idx = col + row * self.col
                        h_start = H * row
                        h_end = h_start + H
                        w_start = W * col
                        w_end = w_start + W
                        if idx < N:
                            # opencv uses bgr
                            vis_img[h_start:h_end,w_start:w_end
                                ] = color[idx,:,:,channel_slice]
                cv2.imshow(self.window_name, vis_img)
                cv2.pollKey()
                time.sleep(1 / self.vis_fps)
