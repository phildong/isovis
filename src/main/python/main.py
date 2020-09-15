import os
import sys
import json
import qdarkstyle
import re
import pims
import numpy as np
import pandas as pd
from pathlib import Path

from itertools import cycle
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QSlider,
    QPushButton,
)
from vispy import scene
from vispy.color import ColorArray
from vispy.visuals.filters import Alpha
from bokeh.palettes import Category20_20

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

FPATH = os.path.dirname(__file__)
with open(os.path.join(FPATH, "config.json"), mode="r") as f:
    CONFIG = json.load(f)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        # init
        QMainWindow.__init__(self)
        self.resize(CONFIG["width"], CONFIG["height"])
        self.setWindowTitle("Isomap Vispy Viewer")
        # playing timer
        self.isPlaying = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timer_play)
        self.timer.start(int(1000 / CONFIG["fps"]))
        # load data
        df = pd.read_feather(os.path.join(FPATH, CONFIG["df_path"])).astype(
            {CONFIG["col_names"]["class"]: str}
        )
        # meta widgets
        layout_meta = QVBoxLayout()
        self.metas = {d: df[d].unique() for d in CONFIG["meta_dims"]}
        self.meta = {d: v[0] for d, v in self.metas.items()}
        for dim, vals in self.metas.items():
            widget = QHBoxLayout()
            widget.addWidget(QLabel(dim))
            cbx = QComboBox()
            cbx.addItems(vals)
            cbx.currentIndexChanged[str].connect(
                lambda l, dim=dim: self.meta_change(l, dim)
            )
            widget.addWidget(cbx)
            layout_meta.addLayout(widget)
        self.df = df.set_index(list(self.meta.keys())).sort_index()
        # load video
        regex = re.compile(os.path.join(*CONFIG["vid_regex"]))
        flist = list(
            filter(
                bool,
                [
                    regex.search(str(p.relative_to(CONFIG["vid_root"])))
                    for p in Path(CONFIG["vid_root"]).rglob("*")
                ],
            )
        )
        self.fdf = (
            pd.DataFrame(
                [
                    [f.group(d) for d in CONFIG["meta_dims"]]
                    + [os.path.join(CONFIG["vid_root"], f.string)]
                    for f in flist
                ],
                columns=CONFIG["meta_dims"] + ["fpath"],
            )
            .set_index(list(self.meta.keys()))
            .sort_index()
        )
        # vispy
        self.data = (
            self.df.loc[tuple(self.meta.values())]
            .copy()
            .reset_index(drop=True)
            .sort_values(CONFIG["col_names"]["frame"])
            .reset_index(drop=True)
        )
        self.vid = pims.Video(self.fdf.loc[tuple(self.meta.values()), "fpath"])
        self.canvas = isoVis(
            data=self.data, vid=self.vid, size=(CONFIG["width"], CONFIG["height"])
        )
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        # player
        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.data.index.max())
        self.slider.setSingleStep(1)
        self.slider.setValue(0)
        self.slider.sliderReleased.connect(self.snap_frame)
        self.slider.valueChanged.connect(self.canvas.fm_change)
        self.ply_button = QPushButton("Play")
        self.ply_button.setCheckable(True)
        self.ply_button.setChecked(False)
        self.ply_button.pressed.connect(self.play)
        layout_player = QHBoxLayout()
        layout_player.addWidget(self.ply_button)
        layout_player.addWidget(self.slider)
        # class
        self.widget_class = QComboBox()
        self.widget_class.addItems(
            ["all"] + self.data[CONFIG["col_names"]["class"]].unique().tolist()
        )
        self.widget_class.setCurrentIndex(0)
        self.subfm = self.data.index
        self.widget_class.currentTextChanged.connect(self.class_change)
        lab_class = QLabel("class")
        layout_class = QHBoxLayout()
        layout_class.addWidget(lab_class)
        layout_class.addWidget(self.widget_class)
        # master layout
        self.layout_master = QGridLayout()
        self.layout_master.addWidget(self.canvas.native, 0, 0)
        self.layout_master.addLayout(layout_meta, 0, 1)
        self.layout_master.addLayout(layout_player, 1, 0)
        self.layout_master.addLayout(layout_class, 1, 1)
        self.layout_master.setColumnStretch(0, 0)
        self.layout_master.setColumnStretch(1, 1)
        self.layout_master.setRowStretch(0, 0)
        self.layout_master.setRowStretch(1, 1)
        widget_master = QWidget()
        widget_master.setLayout(self.layout_master)
        self.setCentralWidget(widget_master)

    def meta_change(self, lab, dim):
        self.meta[dim] = lab
        self.data = (
            self.df.loc[tuple(self.meta.values())]
            .copy()
            .reset_index(drop=True)
            .sort_values(CONFIG["col_names"]["frame"])
            .reset_index(drop=True)
        )
        self.vid = pims.Video(self.fdf.loc[tuple(self.meta.values()), "fpath"])
        # vispy
        self.layout_master.removeWidget(self.canvas.native)
        self.canvas.native.close()
        self.canvas = isoVis(
            data=self.data, vid=self.vid, size=(CONFIG["width"], CONFIG["height"])
        )
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        self.layout_master.addWidget(self.canvas.native, 0, 0)
        # update subfm and class
        self.subfm = self.data.index
        self.widget_class.clear()
        self.widget_class.addItems(
            ["all"] + self.data[CONFIG["col_names"]["class"]].unique().tolist()
        )
        self.widget_class.setCurrentIndex(0)
        # update slider
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.data.index.max())
        self.slider.setSingleStep(1)
        self.slider.setValue(0)
        self.slider.valueChanged.disconnect()
        self.slider.valueChanged.connect(self.canvas.fm_change)

    def play(self):
        if self.ply_button.isChecked():
            self.isPlaying = False
            self.ply_button.setText("Play")
        else:
            self.isPlaying = True
            self.ply_button.setText("Pause")

    def timer_play(self):
        if self.isPlaying:
            curfm = self.slider.value()
            try:
                nxtfm = self.subfm[np.searchsorted(self.subfm, curfm) + 1]
            except IndexError:
                nxtfm = self.subfm[np.searchsorted(self.subfm, curfm)]
            self.slider.setValue(nxtfm)

    def snap_frame(self):
        curfm = self.slider.value()
        nxtfm = self.subfm[np.searchsorted(self.subfm, curfm)]
        self.slider.setValue(nxtfm)

    def class_change(self, class_lab):
        if class_lab == "all" or class_lab == "":
            self.subfm = self.data.index
        else:
            self.subfm = self.data[
                self.data[CONFIG["col_names"]["class"]] == class_lab
            ].index
        self.snap_frame()


class isoVis(scene.SceneCanvas):
    def __init__(self, data, vid, *args, **kwargs) -> None:
        # init
        scene.SceneCanvas.__init__(self, *args, keys="interactive", **kwargs)
        self.unfreeze()
        self.grid = self.central_widget.add_grid(margin=10)
        # color data
        col_cls = CONFIG["col_names"]["class"]
        data["cweak"] = data[col_cls].map(
            {k: v for k, v in zip(data[col_cls].unique(), cycle(Category20_20[0::2]))}
        )
        data["cstrong"] = data[col_cls].map(
            {k: v for k, v in zip(data[col_cls].unique(), cycle(Category20_20[1::2]))}
        )
        # scatter plot
        sct_title = scene.Label("State Space", color="white")
        sct_title.height_max = 30
        self.grid.add_widget(sct_title, row=0, col=0)
        self.sct_view = self.grid.add_view(row=1, col=0, border_color="white")
        self.sct_data = data
        cn = CONFIG["col_names"]
        self.mks = scene.Markers(
            parent=self.sct_view.scene,
            pos=self.sct_data[[cn["x"], cn["y"], cn["z"]]].values,
            face_color=ColorArray(list(self.sct_data["cweak"].values)),
            size=5,
        )
        self.mks.attach(Alpha(0.8))
        self.cur_mks = scene.Markers(
            parent=self.sct_view.scene,
            pos=np.expand_dims(
                self.sct_data.iloc[0, :][[cn["x"], cn["y"], cn["z"]]].values, axis=0
            ),
            face_color=self.sct_data.iloc[0, :]["cstrong"],
        )
        self.cur_mks.set_gl_state(depth_test=False)
        self.axes = scene.XYZAxis(parent=self.sct_view.scene, width=100)
        self.sct_view.camera = "arcball"
        # behav cam
        im_title = scene.Label("Behavior Image", color="white")
        im_title.height_max = 30
        self.grid.add_widget(im_title, row=0, col=1)
        self.im_view = self.grid.add_view(row=1, col=1, border_color="white")
        self.im_data = vid
        fm0 = vid[int(self.sct_data.loc[0, CONFIG["col_names"]["frame"]])]
        self.im = scene.Image(parent=self.im_view.scene, data=fm0)
        self.im_view.camera = "panzoom"
        self.im_view.camera.flip = (False, True, False)
        self.im_view.camera.rect = (0, 0, fm0.shape[1], fm0.shape[0])
        self.im_view.camera.aspect = 1

    def fm_change(self, ifm):
        cn = CONFIG["col_names"]
        self.cur_mks.set_data(
            pos=np.expand_dims(
                self.sct_data.loc[ifm, [cn["x"], cn["y"], cn["z"]]].values,
                axis=0,
            ),
            face_color=self.sct_data.loc[ifm, "cstrong"],
        )
        self.im.set_data(
            self.im_data[int(self.sct_data.loc[ifm, CONFIG["col_names"]["frame"]])]
        )
        self.update()


if __name__ == "__main__":
    appctxt = ApplicationContext()
    appctxt.app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt5"))
    window = MainWindow()
    window.show()
    exit_code = appctxt.app.exec_()
    sys.exit(exit_code)