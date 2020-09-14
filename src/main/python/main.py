import os
import sys
import json
import qdarkstyle
import numpy as np
import pandas as pd

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
)
from vispy import scene
from vispy.app import Timer
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
        self.resize(700, 500)
        self.setWindowTitle("Isomap Vispy Viewer")
        # load data
        df = pd.read_feather(os.path.join(FPATH, CONFIG["df_path"]))
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
        # vispy
        data = self.df.loc[tuple(self.meta.values())].copy()
        self.canvas = isoVis(data=data)
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        # master layout
        self.layout_master = QGridLayout()
        self.layout_master.addWidget(self.canvas.native, 0, 0)
        self.layout_master.addLayout(layout_meta, 0, 1)
        self.layout_master.setColumnStretch(0, 5)
        self.layout_master.setColumnStretch(1, 1)
        widget_master = QWidget()
        widget_master.setLayout(self.layout_master)
        self.setCentralWidget(widget_master)

    def meta_change(self, lab, dim):
        self.meta[dim] = lab
        data = self.df.loc[tuple(self.meta.values())].copy()
        self.layout_master.removeWidget(self.canvas.native)
        self.canvas.native.close()
        self.canvas = isoVis(data=data)
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        self.layout_master.addWidget(self.canvas.native, 0, 0)


class isoVis(scene.SceneCanvas):
    def __init__(self, data=None, *args, **kwargs) -> None:
        # init
        scene.SceneCanvas.__init__(self, *args, keys="interactive", **kwargs)
        self.unfreeze()
        self.timer = Timer(
            interval=1 / 10,
            connect=self.on_timer,
            app=self.app,
            start=True,
            iterations=999,
        )
        self.counter = 0
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
        self.mks = scene.Markers(
            parent=self.sct_view.scene,
            pos=self.sct_data[["comp0", "comp1", "comp2"]].values,
            face_color=ColorArray(list(self.sct_data["cweak"].values)),
            size=5,
        )
        self.mks.attach(Alpha(0.8))
        self.cur_mks = scene.Markers(
            parent=self.sct_view.scene,
            pos=np.expand_dims(
                self.sct_data.iloc[0, :][["comp0", "comp1", "comp2"]].values, axis=0
            ),
            face_color=self.sct_data.iloc[0, :]["cstrong"],
            size=20,
        )
        self.cur_mks.set_gl_state(depth_test=False)
        self.axes = scene.XYZAxis(parent=self.sct_view.scene, width=100)
        self.sct_view.camera = "arcball"
        # behav cam
        im_title = scene.Label("Behavior Image", color="white")
        im_title.height_max = 30
        self.grid.add_widget(im_title, row=0, col=1)
        self.im_view = self.grid.add_view(row=1, col=1, border_color="white")
        self.im_data = np.random.random(size=(1000, 100, 200))
        self.im = scene.Image(parent=self.im_view.scene, data=self.im_data[0, :, :])
        self.im_view.camera = "panzoom"
        self.im_view.camera.rect = (0, 0, self.im_data.shape[1], self.im_data.shape[2])

    def on_timer(self, event=None):
        self.counter += 1
        self.cur_mks.set_data(
            pos=np.expand_dims(
                self.sct_data.iloc[self.counter, :][["comp0", "comp1", "comp2"]].values,
                axis=0,
            ),
            face_color=self.sct_data.iloc[self.counter, :]["cstrong"],
        )
        self.im.set_data(self.im_data[self.counter, :, :])
        self.update()


if __name__ == "__main__":
    appctxt = ApplicationContext()
    appctxt.app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt5"))
    window = MainWindow()
    window.show()
    exit_code = appctxt.app.exec_()
    sys.exit(exit_code)