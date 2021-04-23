#%% imports
import numpy as np
import pandas as pd
import pims
import ffmpeg
from tqdm import tqdm
from main import isoVis
from vispy.util.quaternion import Quaternion

# %% load data and define parameters
anm = "ts45-4"
ss = "s10"
subset = "full"
vpath = "vid/S10Merged.avi"
fname = "output.mp4"
w, h = 1920, 1080
options = {}
proj = pd.read_feather("proj.feather")
behav = pd.read_feather("behav.feather").rename({"fmCam0": "frame"}, axis="columns")
proj_sub = proj[
    (proj["animal"] == anm) & (proj["session"] == ss) & (proj["subset"] == subset)
]
proj_sub = proj_sub.merge(
    behav[["animal", "session", "frame", "fmCam1"]], on=["animal", "session", "frame"]
)
proj_sub = proj_sub.sort_values("fmCam1").reset_index(drop=True)
vid = pims.Video(vpath)
# %% create isovis
isovis = isoVis(proj_sub, vid, size=(w, h), app="pyqt5")
isovis.create_native()
# isovis.show()
# im = isovis.render()

#%% write video
process = (
    ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgba", s="{}x{}".format(w, h))
    .output(fname, pix_fmt="yuv420p", vcodec="libx264", r=30, **options)
    .overwrite_output()
    .run_async(pipe_stdin=True)
)
for i in tqdm(range(500)):
    isovis.fm_change(i)
    isovis.sct_view.camera._quaternion = Quaternion.create_from_axis_angle(
        np.pi / 8, -i / 500, -(1 - i / 500), -1
    )
    isovis.sct_view.camera.center = (0, 0, 0)
    isovis.sct_view.camera.distance = 3
    isovis.sct_view.camera.view_changed()
    im = isovis.render()
    process.stdin.write(im.tobytes())
process.stdin.close()
process.wait()