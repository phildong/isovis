#%% imports
from itertools import cycle

import ffmpeg
import numpy as np
import pandas as pd
import pims
from bokeh.palettes import Category20_20, Category10_10
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from vispy import scene
from vispy.color import ColorArray
from vispy.visuals.filters import Alpha

# %% load data and define parameters
anm = "ts45-4"
ss = "s10"
subset = "full"
vpath = "data/S10Merged.avi"
options = {}
proj = pd.read_feather("data/proj.feather")
behav = pd.read_feather("data/behav.feather").rename(
    {"fmCam0": "frame"}, axis="columns"
)
proj_sub = proj[
    (proj["animal"] == anm) & (proj["session"] == ss) & (proj["subset"] == subset)
]
proj_sub = proj_sub.merge(
    behav[["animal", "session", "frame", "fmCam1"]], on=["animal", "session", "frame"]
)
proj_sub = proj_sub.sort_values("fmCam1").reset_index(drop=True)
proj_sub["comp0"] = gaussian_filter1d(proj_sub["comp0"], 5)
proj_sub["comp1"] = gaussian_filter1d(proj_sub["comp1"], 5)
proj_sub["comp2"] = gaussian_filter1d(proj_sub["comp2"], 5)
vid = pims.Video(vpath)
# %% define isovis
CONFIG = {
    "col_names": {
        "class": "state",
        "frame": "fmCam1",
        "x": "comp0",
        "y": "comp1",
        "z": "comp2",
    },
    "height": 1440,
    "width": 1920,
    "fps": 30,
}


class isoVis(scene.SceneCanvas):
    def __init__(
        self,
        data,
        vid,
        show_state=True,
        show_behav=True,
        colorize=True,
        title_dict=dict(),
        *args,
        **kwargs
    ) -> None:
        # init
        scene.SceneCanvas.__init__(self, *args, keys="interactive", **kwargs)
        self.unfreeze()
        self.grid = self.central_widget.add_grid(margin=10)
        self.show_behav = show_behav
        self.show_state = show_state
        self.title_dict = title_dict
        data = data.copy()
        # color data
        col_cls = CONFIG["col_names"]["class"]
        if colorize:
            data["cweak"] = data[col_cls].map(
                {
                    k: v
                    for k, v in zip(data[col_cls].unique(), cycle(Category20_20[0::2]))
                }
            )
            data["cstrong"] = data[col_cls].map(
                {k: v for k, v in zip(data[col_cls].unique(), cycle(Category10_10))}
            )
        else:
            data["cweak"] = "black"
            data["cstrong"] = "black"
        # scatter plot
        rcount = 0
        if show_state:
            self.sct_title = scene.Label(
                "State Space", color="black", face="Arial", bold=True, font_size=40
            )
            self.grid.add_widget(self.sct_title, row=rcount, col=0)
            rcount += 1
        self.sct_view = self.grid.add_view(row=rcount, col=0, row_span=8)
        rcount += 8
        self.sct_data = data
        cn = CONFIG["col_names"]
        self.mks = scene.Markers(parent=self.sct_view.scene)
        self.mks.antialias = 0  # https://github.com/vispy/vispy/issues/1583
        self.mks.set_data(
            pos=self.sct_data[[cn["x"], cn["y"], cn["z"]]].values,
            face_color=ColorArray(list(self.sct_data["cweak"].values)),
            edge_width=0,
            edge_color=None,
            size=4,
            scaling=False,
        )
        # self.ind = scene.Line(
        #     parent=self.sct_view.scene,
        #     pos=np.array([[0, 0, 0], [0.1, 0.1, 0.1], [0.1, 0.1, 0.2], [0, 0.1, 0.2]]),
        #     color="black",
        #     width=10,
        # )
        self.mks.attach(Alpha(0.9))
        if self.show_state:
            self.cur_mks = scene.Markers(parent=self.sct_view.scene)
            self.cur_mks.set_gl_state(depth_test=False)
            self.cur_mks.set_data(
                pos=np.expand_dims(
                    self.sct_data.iloc[0, :][[cn["x"], cn["y"], cn["z"]]].values, axis=0
                ),
                face_color=self.sct_data.iloc[0, :]["cstrong"],
                size=20,
                edge_width=2,
                edge_color="black",
                scaling=False,
            )

        self.axes = scene.XYZAxis(parent=self.sct_view.scene, width=100)
        self.sct_view.camera = scene.cameras.TurntableCamera(
            center=(0, 0, 0), azimuth=0, elevation=0
        )
        # behav cam
        if self.show_behav:
            self.im_view = self.grid.add_view(
                row=rcount, col=0, row_span=2, border_color="white"
            )
            rcount += 1
            self.im_data = vid
            fm0 = vid[int(self.sct_data.loc[0, CONFIG["col_names"]["frame"]])]
            self.im = scene.Image(parent=self.im_view.scene, data=fm0)
            self.im_view.camera = "panzoom"
            self.im_view.camera.flip = (False, True, False)
            self.im_view.camera.rect = (0, 0, fm0.shape[1], fm0.shape[0])
            self.im_view.camera.aspect = 1

    def fm_change(self, ifm):
        cn = CONFIG["col_names"]
        cur_row = self.sct_data.loc[ifm]
        if self.show_state:
            self.cur_mks.set_data(
                pos=np.expand_dims(
                    np.array([cur_row[cn["x"]], cur_row[cn["y"]], cur_row[cn["z"]]]),
                    axis=0,
                ),
                face_color=cur_row["cstrong"],
                size=20,
                edge_width=2,
                edge_color="black",
                scaling=False,
            )
            lb = cur_row[cn["class"]]
            self.sct_title.text = self.title_dict.get(lb, lb)
            self.sct_title._text_visual.color = cur_row["cstrong"]
        if self.show_behav:
            self.im.set_data(self.im_data[int(cur_row[cn["frame"]])])
        self.update()


#%% write intro video
fname = "intro.mp4"
nfm = 600
process = (
    ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="rgba",
        s="{}x{}".format(CONFIG["width"], CONFIG["height"]),
    )
    .output(fname, pix_fmt="yuv420p", vcodec="libx264", r=30, **options)
    .overwrite_output()
    .run_async(pipe_stdin=True)
)
isovis = isoVis(
    proj_sub,
    vid,
    show_behav=False,
    colorize=False,
    show_state=False,
    bgcolor="white",
    size=(CONFIG["width"], CONFIG["height"]),
    app="pyqt5",
)
isovis.create_native()
isovis.sct_view.camera.orbit(0, 45)
isovis.sct_view.camera.distance = 1.3
for i in tqdm(range(nfm)):
    isovis.sct_view.camera.orbit(180 / nfm, 0)
    isovis.sct_view.camera.view_changed()
    im = isovis.render()
    process.stdin.write(im.tobytes())
process.stdin.close()
process.wait()

#%% write cluster video
def vec_to_ae(v):
    # azi is angle from -y axis
    a = np.degrees(np.arctan2(v[1], v[0]))
    if a > 180:
        a = a - 270
    else:
        a = a + 90
    e = np.degrees(np.arctan2(v[2], np.sqrt(np.sum(v[0] ** 2 + v[1] ** 2))))
    return a, e


fname = "cluster.mp4"
# rename_dict = {"turn_left": "drink_left", "turn_right": "drink_right"}
nsmp = {"drink_left": 150, "drink_right": 150, "run_left": 400, "run_right": 400}
stp = {"drink_left": 1, "drink_right": 1, "run_left": 2, "run_right": 2}
transition_sigma = 10
view_ele_offset = 10
view_azi_offset = 8
view_dist_offset = 0.7
# proj_sub["state"] = proj_sub["state"].map(lambda k: rename_dict.get(k, k))
process = (
    ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="rgba",
        s="{}x{}".format(CONFIG["width"], CONFIG["height"]),
    )
    .output(fname, pix_fmt="yuv420p", vcodec="libx264", r=30, **options)
    .overwrite_output()
    .run_async(pipe_stdin=True)
)
isovis = isoVis(
    proj_sub,
    vid,
    show_behav=True,
    colorize=True,
    bgcolor="white",
    title_dict={
        "drink_left": "Drinking Left",
        "drink_right": "Drinking Right",
        "run_left": "Running Left",
        "run_right": "Running Right",
    },
    size=(CONFIG["width"], CONFIG["height"]),
    app="pyqt5",
)
isovis.create_native()
plan_ls = []
last_ang = 0
for st, st_df in proj_sub.groupby("state"):
    try:
        subdf = st_df.iloc[: nsmp[st] : stp[st]].copy()
    except KeyError:
        continue
    view_vec = subdf["comp0"].median(), subdf["comp1"].median(), subdf["comp2"].median()
    view_dist = np.sqrt(
        subdf["comp0"] ** 2 + subdf["comp1"] ** 2 + subdf["comp2"] ** 2
    ).max()
    a, e = vec_to_ae(view_vec)
    a0, a1 = a - view_azi_offset, a + view_azi_offset
    a0, a1 = sorted([a0, a1], key=lambda aa: np.abs(aa - last_ang))
    last_ang = a1
    subdf["azi"] = np.linspace(a0, a1, len(subdf))
    subdf["ele"] = np.sign(e) * np.clip(np.abs(e) - view_ele_offset, 0, 90)
    subdf["view_dist"] = view_dist
    subdf["tt_opacity"] = 1
    subdf.tt_opacity.iloc[:transition_sigma] = 0
    subdf.tt_opacity.iloc[-transition_sigma:] = 0
    plan_ls.append(subdf)
plan_df = pd.concat(plan_ls)
plan_df["azi"] = gaussian_filter1d(plan_df["azi"], transition_sigma)
plan_df["ele"] = gaussian_filter1d(plan_df["ele"], transition_sigma)
plan_df["view_dist"] = gaussian_filter1d(plan_df["view_dist"], transition_sigma)
plan_df["tt_opacity"] = gaussian_filter1d(
    plan_df["tt_opacity"].astype(float), transition_sigma
)
for ir, row in tqdm(plan_df.iterrows()):
    isovis.sct_view.camera.azimuth = row["azi"]
    isovis.sct_view.camera.elevation = row["ele"]
    isovis.sct_view.camera.distance = row["view_dist"] + view_dist_offset
    isovis.sct_view.camera.view_changed()
    try:
        isovis.sct_title._text_visual.detach(ft)
    except (NameError, ValueError):
        pass
    ft = Alpha(row["tt_opacity"])
    isovis.sct_title._text_visual.attach(ft)
    isovis.fm_change(ir)
    im = isovis.render()
    process.stdin.write(im.tobytes())
process.stdin.close()
process.wait()