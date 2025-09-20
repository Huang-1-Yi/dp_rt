from typing import Optional, Callable, Generator
import numpy as np
import av
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs

# VideoRecorder_new用
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
import time


def read_video(
        video_path: str, dt: float,
        video_start_time: float=0.0, 
        start_time: float=0.0,
        img_transform: Optional[Callable[[np.ndarray], np.ndarray]]=None,
        thread_type: str="AUTO",
        thread_count: int=0,
        max_pad_frames: int=10
        ) -> Generator[np.ndarray, None, None]:
    frame = None
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = thread_type
        stream.thread_count = thread_count
        next_global_idx = 0
        for frame_idx, frame in enumerate(container.decode(stream)):
            # The presentation time in seconds for this frame.
            since_start = frame.time
            frame_time = video_start_time + since_start
            local_idxs, global_idxs, next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=start_time,
                dt=dt,
                next_global_idx=next_global_idx
            )
            if len(global_idxs) > 0:
                array = frame.to_ndarray(format='rgb24')
                img = array
                if img_transform is not None:
                    img = img_transform(array)
                for global_idx in global_idxs:
                    yield img
    # repeat last frame max_pad_frames times
    array = frame.to_ndarray(format='rgb24')
    img = array
    if img_transform is not None:
        img = img_transform(array)
    for i in range(max_pad_frames):
        yield img

class VideoRecorder:
    def __init__(self,
        fps,
        codec,
        input_pix_fmt,
        # options for codec
        **kwargs
    ):
        """
        input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
        """

        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        # runtime set
        self._reset_state()
    
    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0
    
    @classmethod
    def create_h264(cls,
            fps,
            codec='h264',
            input_pix_fmt='rgb24',
            output_pix_fmt='yuv420p',
            crf=18,
            profile='high',
            **kwargs
        ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={
                'crf': str(crf),
                'profile': profile
            },
            **kwargs
        )
        return obj


    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()

        self.container = av.open(file_path, mode='w')
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for k, v in self.kwargs.items():
            setattr(codec_context, k, v)
        self.start_time = start_time
    
    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError('Must run start() before writing!')
        
        n_repeats = 1
        if self.start_time is not None:
            local_idxs, global_idxs, self.next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1/self.fps,
                next_global_idx=self.next_global_idx
            )
            # number of appearance means repeats
            n_repeats = len(local_idxs)
        
        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h,w,c = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype

        frame = av.VideoFrame.from_ndarray(
            img, format=self.input_pix_fmt)
        for i in range(n_repeats):
            for packet in self.stream.encode(frame):
                self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            return

        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()

        # reset runtime parameters
        self._reset_state()


class VideoRecorder_new(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes
    class Command(enum.Enum):
        START_RECORDING = 0
        STOP_RECORDING = 1
    
    def __init__(self,
        fps,
        codec,
        input_pix_fmt,
        buffer_size=128,
        no_repeat=False,
        # options for codec
        **kwargs     
    ):
        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.buffer_size = buffer_size
        self.no_repeat = no_repeat
        self.kwargs = kwargs
        
        self.img_queue = None
        self.cmd_queue = None
        self.stop_event = None
        self.ready_event = None
        self.is_started = False
        self.shape = None
        
        self._reset_state()
        
    # ======== custom constructors =======
    @classmethod
    def create_h264(cls,
            fps,
            codec='h264',
            input_pix_fmt='rgb24',
            output_pix_fmt='yuv420p',
            crf=18,
            profile='high',
            **kwargs
        ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={
                'crf': str(crf),
                'profile': profile
            },
            **kwargs
        )
        return obj
    
    @classmethod
    def create_hevc_nvenc(cls,
            fps,
            codec='hevc_nvenc',
            input_pix_fmt='rgb24',
            output_pix_fmt='yuv420p',
            bit_rate=6000*1000,
            options={
                'tune': 'll', 
                'preset': 'p1'
            },
            **kwargs
        ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            bit_rate=bit_rate,
            options=options,
            **kwargs
        )
        return obj

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, shm_manager:SharedMemoryManager, data_example: np.ndarray):
        super().__init__()
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.img_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={
                'img': data_example,
                'repeat': 1
            },
            buffer_size=self.buffer_size
        )
        self.cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={
                'cmd': self.Command.START_RECORDING.value,
                'video_path': np.array('a'*self.MAX_PATH_LENGTH)
            },
            buffer_size=self.buffer_size
        )
        self.shape = data_example.shape
        self.is_started = True
        super().start()
    
    def stop(self):
        self.stop_event.set()

    def start_wait(self):
        self.ready_event.wait()
    
    def end_wait(self):
        self.join()
        
    def is_ready(self):
        return (self.start_time is not None) \
            and (self.ready_event.is_set()) \
            and (not self.stop_event.is_set())
        
    def start_recording(self, video_path: str, start_time: float=-1):
        """开始录制视频
        
        Args:
            video_path: 视频文件路径
            start_time: 录制开始时间戳（可选）
        """
        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.start_time = start_time
        self.cmd_queue.put({
            'cmd': self.Command.START_RECORDING.value,
            'video_path': video_path
        })
        print(f'开始录制视频到 {video_path} ,起始时间 {start_time}...')
    
    def stop_recording(self):
        self.cmd_queue.put({
            'cmd': self.Command.STOP_RECORDING.value
        })
        self._reset_state()
        print('停止录制视频')
    
    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError('Must run start() before writing!')
            
        n_repeats = 1
        if (not self.no_repeat) and (self.start_time is not None):
            local_idxs, global_idxs, self.next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1/self.fps,
                next_global_idx=self.next_global_idx
            )
            # number of apperance means repeats
            n_repeats = len(local_idxs)
        
        self.img_queue.put({
            'img': img,
            'repeat': n_repeats
        })
        # print(f'写入一帧图像，重复次数: {n_repeats}, 当前时间: {frame_time}')
    
    def get_img_buffer(self):
        """
        Get view to the next img queue memory
        for zero-copy writing
        """
        data = self.img_queue.get_next_view()
        img = data['img']
        return img
    
    def write_img_buffer(self, img: np.ndarray, frame_time=None):
        """
        Must be used with the buffer returned by get_img_buffer
        for zero-copy writing
        """
        if not self.is_ready():
            raise RuntimeError('Must run start() before writing!')
            
        n_repeats = 1
        if (not self.no_repeat) and (self.start_time is not None):
            local_idxs, global_idxs, self.next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1/self.fps,
                next_global_idx=self.next_global_idx
            )
            # number of apperance means repeats
            n_repeats = len(local_idxs)
        
        self.img_queue.put_next_view({
            'img': img,
            'repeat': n_repeats
        })

    # ========= interval API ===========
    def _reset_state(self):
        self.start_time = None
        self.next_global_idx = 0
    
    def run(self):
        # I'm sorry it has to be this complicated...
        self.ready_event.set()
        while not self.stop_event.is_set():
            video_path = None
            # ========= stopped state ============
            while (video_path is None) and (not self.stop_event.is_set()):
                try:
                    commands = self.cmd_queue.get_all()
                    for i in range(len(commands['cmd'])):
                        cmd = commands['cmd'][i]
                        if cmd == self.Command.START_RECORDING.value:
                            video_path = str(commands['video_path'][i])
                        elif cmd == self.Command.STOP_RECORDING.value:
                            video_path = None
                        else:
                            raise RuntimeError("Unknown command: ", cmd)
                except Empty:
                    time.sleep(0.1/self.fps)
            if self.stop_event.is_set():
                break
            assert video_path is not None
            # ========= recording state ==========
            with av.open(video_path, mode='w') as container:
                stream = container.add_stream(self.codec, rate=self.fps)
                h,w,c = self.shape
                stream.width = w
                stream.height = h
                codec_context = stream.codec_context
                for k, v in self.kwargs.items():
                    setattr(codec_context, k, v)
                
                # loop
                while not self.stop_event.is_set():
                    try:
                        command = self.cmd_queue.get()
                        cmd = int(command['cmd'])
                        if cmd == self.Command.STOP_RECORDING.value:
                            break
                        elif cmd == self.Command.START_RECORDING.value:
                            continue
                        else:
                            raise RuntimeError("Unknown command: ", cmd)
                    except Empty:
                        pass
                    
                    try:
                        with self.img_queue.get_view() as data:
                            img = data['img']
                            repeat = data['repeat']
                            frame = av.VideoFrame.from_ndarray(
                                img, format=self.input_pix_fmt)
                        for _ in range(repeat):
                            for packet in stream.encode(frame):
                                container.mux(packet)
                    except Empty:
                        time.sleep(0.1/self.fps)
                        
                # Flush queue
                try:
                    while not self.img_queue.empty():
                        with self.img_queue.get_view() as data:
                            img = data['img']
                            repeat = data['repeat']
                            frame = av.VideoFrame.from_ndarray(
                                img, format=self.input_pix_fmt)
                        for _ in range(repeat):
                            for packet in stream.encode(frame):
                                container.mux(packet)
                except Empty:
                    pass

                 # Flush stream
                for packet in stream.encode():
                    container.mux(packet)
