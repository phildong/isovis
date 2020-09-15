# isovis

![isovis](isovis.gif)
IsoVis is a simple tool to visualize arbitrary 3d point clouds together with corresponding video frames, intended to help neuroscientist to identify interesting clusters in 3d projection of neural activities (state space) that may correspond to behavior stereotypes.

# installation

```bash
conda create -n isovis -c conda-forge -y python=3.6 vispy pandas numpy pyarrow qdarkstyle pyqt bokeh av pims scikit-image
conda activate isovis
pip install fbs
```

# usage

1. adjust settings under `src/main/python/config.json` (see below).
1. run `fbs run`.
1. use metadata dropdowns on the right to identify a single session. use "class" dropdown to constrain player to frames corresponding to the class label.
1. "State Space" camera control: `LMB` to orbit, `RMB or Wheel` to zoom, `SHIFT + LMB` to translate FOV.
1. "Behavior Image" camera contro: `LMB` to pan, `RMB or Wheel` to zoom.

# config

The `config.json` should contain a single dictionary with following keys:

- `df_path`: path to the dataframe defining 3d points.
- `meta_dims`: metadata dimensions (column names) in the dataframe that identify a single session.
- `col_names`: names of columns in the dataframe that are used to plot the 3d points.
- `height` and `width`: height and width of the window.
- `fps`: frame rate of playback.
- `vid_root`: root folder containing behavior videos of all the sessions.
- `vid_regex`: a list of regex strings specifying folder structures relative to `vid_root` and how metadata should be extracted from the path. Each item in the list correspond to a level of folder (top level first), and regex should contain named capture groups that can be matched with `meta_dims`.
