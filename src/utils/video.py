import os
import contextlib
import subprocess
import numpy as np
import skvideo.io

@contextlib.contextmanager
def virtual_display(display_id=":99", screen_size="1024x768x24"):
    """虚拟显示上下文管理器"""
    xvfb_process = None
    original_display = os.environ.get("DISPLAY")
    
    try:
        if "DISPLAY" not in os.environ:
            print(f"启动Xvfb虚拟显示 {display_id}...")
            xvfb_process = subprocess.Popen(
                ["Xvfb", display_id, "-screen", "0", screen_size]
            )
            os.environ["DISPLAY"] = display_id
            # 给Xvfb一点启动时间
            import time
            time.sleep(1)
        
        yield  # 这里执行主程序代码
        
    finally:
        # 清理：关闭Xvfb并恢复原来的DISPLAY设置
        if xvfb_process is not None and xvfb_process.poll() is None:
            print("关闭Xvfb进程...")
            xvfb_process.terminate()
            xvfb_process.wait()
        
        # 恢复原来的DISPLAY环境变量
        if original_display is not None:
            os.environ["DISPLAY"] = original_display
        elif "DISPLAY" in os.environ:
            del os.environ["DISPLAY"]



def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_video(filename, video_frames, fps=60, video_format='mp4'):
    assert fps == int(fps), fps
    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    )

def save_videos(filename, *video_frames, axis=1, **kwargs):
    ## video_frame : [ N x H x W x C ]
    video_frames = np.concatenate(video_frames, axis=axis)
    save_video(filename, video_frames, **kwargs)
