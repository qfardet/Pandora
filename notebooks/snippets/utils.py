# pylint: skip-file

import copy
import warnings
from typing import Tuple

import numpy as np
import xarray as xr

from pandora.constants import *

warnings.simplefilter(action='ignore')
import bokeh.plotting as bpl
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import ColorBar, BasicTicker, LinearColorMapper, Legend
from ipywidgets import interact, Layout
from bokeh.palettes import RdYlBu
from bokeh.io import push_notebook, show, output_notebook
import ipyvolume as ipv
from ipyvolume import widgets

def plot_disparity(input_disparity_map: xr.Dataset) -> None:
    """
    Plot disparity map with selective bit mask
    :param input_disparity_map: input disparity map
    :type  input_disparity_map: xr.dataset
    :return: None
    """

    output_notebook()
    disparity_map = add_validity_mask_to_dataset(input_disparity_map)

    min_d = np.nanmin(disparity_map['disparity_map'].data)
    max_d = np.nanmax(disparity_map['disparity_map'].data)
    mapper_avec_opti = LinearColorMapper(palette='Viridis256', low=min_d, high=max_d)

    color_bar = ColorBar(color_mapper=mapper_avec_opti, ticker=BasicTicker(),
                         label_standoff=12, border_line_color=None, location=(0, 0))
    size = 0.5
    dw = disparity_map['disparity_map'].shape[1]
    dh = disparity_map['disparity_map'].shape[0]

    fig = figure(title="Disparity map", width=800, height=450,
                 tools=['reset', 'pan', "box_zoom"], output_backend="webgl")

    fig.image(image=[np.flip(disparity_map['disparity_map'].data, 0)], x=1, y=0, dw=dw, dh=dh,
              color_mapper=mapper_avec_opti)

    x = np.where(disparity_map['nodata_border_left_mask'].data != 0)[1]
    y = dh - np.where(disparity_map['nodata_border_left_mask'].data != 0)[0]
    nodata_border_left_mask = fig.circle(x=x, y=y, size=size, color="red")
    nodata_border_left_mask.visible = False

    x = np.where(disparity_map['nodata_border_right_mask'].data != 0)[1]
    y = dh - np.where(disparity_map['nodata_border_right_mask'].data != 0)[0]
    nodata_border_right_mask = fig.circle(x=x, y=y, size=size, color="red")
    nodata_border_right_mask.visible = False

    x = np.where(disparity_map['incomplete_right_mask'].data != 0)[1]
    y = dh - np.where(disparity_map['incomplete_right_mask'].data != 0)[0]
    incomplete_right_mask = fig.circle(x=x, y=y, size=size, color="red")
    incomplete_right_mask.visible = False

    x = np.where(disparity_map['stopped_interp_mask'].data != 0)[1]
    y = dh - np.where(disparity_map['stopped_interp_mask'].data != 0)[0]
    stopped_interp_mask = fig.circle(x=x, y=y, size=size, color="red")
    stopped_interp_mask.visible = False

    x = np.where(disparity_map['filled_occlusion_mask'].data != 0)[1]
    y = dh - np.where(disparity_map['filled_occlusion_mask'].data != 0)[0]
    filled_occlusion_mask = fig.circle(x=x, y=y, size=size, color="red")
    filled_occlusion_mask.visible = False

    x = np.where(disparity_map['filled_mismatch_mask'].data != 0)[1]
    y = dh - np.where(disparity_map['filled_mismatch_mask'].data != 0)[0]
    filled_mismatch_mask = fig.circle(x=x, y=y, size=size, color="red")
    filled_mismatch_mask.visible = False

    x = np.where(disparity_map['masked_left_mask'].data != 0)[1]
    y = dh - np.where(disparity_map['masked_left_mask'].data != 0)[0]
    masked_left_mask = fig.circle(x=x, y=y, size=size, color="red")
    masked_left_mask.visible = False

    x = np.where(disparity_map['masked_right_mask'].data != 0)[1]
    y = dh - np.where(disparity_map['masked_right_mask'].data != 0)[0]
    masked_right_mask = fig.circle(x=x, y=y, size=size, color="red")
    masked_right_mask.visible = False

    x = np.where(disparity_map['occlusion_mask'].data != 0)[1]
    y = dh - np.where(disparity_map['occlusion_mask'].data != 0)[0]
    occlusion_mask = fig.circle(x=x, y=y, size=size, color="red")
    occlusion_mask.visible = False

    x = np.where(disparity_map['mismatch_mask'].data != 0)[1]
    y = dh - np.where(disparity_map['mismatch_mask'].data != 0)[0]
    mismatch_mask = fig.circle(x=x, y=y, size=size, color="red")
    mismatch_mask.visible = False

    x = np.where(disparity_map['filled_nodata'].data != 0)[1]
    y = dh - np.where(disparity_map['filled_nodata'].data != 0)[0]
    filled_nodata = fig.circle(x=x, y=y, size=size, color="red")
    filled_nodata.visible = False

    x = np.where(disparity_map['invalid_mask'].data != 0)[1]
    y = dh - np.where(disparity_map['invalid_mask'].data != 0)[0]
    invalid_mask = fig.circle(x=x, y=y, size=size, color="red")
    invalid_mask.visible = True

    legend = Legend(items=[
        ("Nodata border left_mask (invalid)", [nodata_border_left_mask]),
        ("Nodata border right_mask (invalid)", [nodata_border_right_mask]),
        ("Incomplete right mask", [incomplete_right_mask]),
        ("Stopped interp mask", [stopped_interp_mask]),
        ("Filled occlusion mask", [filled_occlusion_mask]),
        ("Filled mismatch mask", [filled_mismatch_mask]),
        ("Masked left mask (invalid)", [masked_left_mask]),
        ("Masked right mask (invalid)", [masked_right_mask]),
        ("Occlusion mask (invalid)", [occlusion_mask]),
        ("Mismatch mask (invalid)", [mismatch_mask]),
        ("Filled nodata", [filled_nodata]),
        ("All invalid types", [invalid_mask])], location="center", click_policy="hide")

    fig.add_layout(color_bar, 'right')
    fig.add_layout(legend, 'right')

    show(fig)


def compare_2_disparities(input_first_disp_map: xr.Dataset, first_title: str, input_second_disp_map: xr.Dataset,
                          second_title: str) -> None:
    """
    Show 2 disparity maps
    :param input_first_disp_map: disparity map
    :type input_first_disp_map: dataset
    :param first_title: disparity map title
    :type first_title: str
    :param input_second_disp_map: disparity map
    :type input_second_disp_map: dataset
    :param second_title: disparity map title
    :type second_title: str
    :return: none
    """
    output_notebook()
    size = 0.5

    first_disp_map = add_validity_mask_to_dataset(input_first_disp_map)
    second_disp_map = add_validity_mask_to_dataset(input_second_disp_map)

    min_d = np.nanmin(first_disp_map['disparity_map'].data)
    max_d = np.nanmax(first_disp_map['disparity_map'].data)
    mapper_avec_opti = LinearColorMapper(palette='Viridis256', low=min_d, high=max_d)

    color_bar = ColorBar(color_mapper=mapper_avec_opti, ticker=BasicTicker(),
                         label_standoff=12, border_line_color=None, location=(0, 0))

    dw = first_disp_map['disparity_map'].shape[1]
    dh = first_disp_map['disparity_map'].shape[0]

    if first_title == None:
        first_title = "First disparity map"
    if second_title == None:
        second_title = "Second disparity map"

    # First disparity map
    first_fig = figure(title=first_title, width=450, height=450,
                       tools=['reset', 'pan', "box_zoom"], output_backend="webgl")

    first_fig.image(image=[np.flip(first_disp_map['disparity_map'].data, 0)], x=1, y=0, dw=dw, dh=dh,
                    color_mapper=mapper_avec_opti)

    x = np.where(first_disp_map['invalid_mask'].data != 0)[1]
    y = dh - np.where(first_disp_map['invalid_mask'].data != 0)[0]
    first_inv_msk = first_fig.circle(x=x, y=y, size=size, color="red")

    legend = Legend(items=[
        ("inv msk", [first_inv_msk])],
        location="center", click_policy="hide")

    first_fig.add_layout(legend, 'below')
    first_fig.add_layout(color_bar, 'right')

    # Second disparity map
    second_fig = figure(title=second_title, width=450, height=450,
                        tools=['reset', 'pan', "box_zoom"], output_backend="webgl", x_range=first_fig.x_range,
                        y_range=first_fig.y_range)

    second_fig.image(image=[np.flip(second_disp_map['disparity_map'].data, 0)], x=1, y=0, dw=dw, dh=dh,
                     color_mapper=mapper_avec_opti)

    x = np.where(second_disp_map['invalid_mask'].data != 0)[1]
    y = dh - np.where(second_disp_map['invalid_mask'].data != 0)[0]
    second_inv_msk = second_fig.circle(x=x, y=y, size=size, color="red")

    legend = Legend(items=[
        ("inv msk", [second_inv_msk])],
        glyph_height=10,
        glyph_width=10, location="center", click_policy="hide")

    second_fig.add_layout(legend, 'below')
    second_fig.add_layout(color_bar, 'right')

    layout = column(row(first_fig, second_fig))

    show(layout)

def compare_2_error_maps(reference_error: np.ndarray, second_error: np.ndarray) -> None:
    """
    Show the percentage of error disminution or increase of the second error in respect to the reference error
    :param reference_error: reference error map
    :type reference_error: np.ndarray
    :param second_error: second error map
    :type second_error: np.ndarray
    :return: none
    """
    output_notebook()
    size = 0.5
    reference_error_base = reference_error.copy()
    reference_error_base[np.where(reference_error_base == 0)] = 1

    error_diff = (((second_error - reference_error)/reference_error_base)*100)
    min_d = np.nanmin(error_diff)
    max_d = np.nanmax(error_diff)
    #error_diff[np.where(error_diff>100)] = 100
    mapper_avec_opti = LinearColorMapper( palette = "Turbo256", low=-120, high=120)

    color_bar = ColorBar(color_mapper=mapper_avec_opti, ticker=BasicTicker(),
                         label_standoff=12, border_line_color=None, location=(0, 0))

    dw = reference_error.shape[1]
    dh = reference_error.shape[0]

    # First disparity map
    first_fig = figure(title="Error reduction or augmentation percentage", width=700, height=650,
                       tools=['reset', 'pan', "box_zoom"], output_backend="webgl")

    first_fig.image(image=[np.flip(error_diff, 0)], x=1, y=0, dw=dw, dh=dh,
                    color_mapper=mapper_avec_opti)

    first_fig.add_layout(color_bar, 'right')

    show(first_fig)

def compare_3_disparities_and_error(input_first_disp_map: xr.Dataset, first_title: str,
                                    input_second_disp_map: xr.Dataset, second_title: str,
                                    input_third_disp_map: xr.Dataset, third_title: str, error_map: np.array,
                                    error_title: str) -> None:
    """
    Show 3 disparity maps and error
    :param input_first_disp_map: disparity map
    :type input_first_disp_map: dataset
    :param first_title: disparity map title
    :type first_title: str
    :param input_second_disp_map: disparity map
    :type input_second_disp_map: dataset
    :param second_title: disparity map title
    :type second_title: str
    :param input_third_disp_map: disparity map
    :type input_third_disp_map: dataset
    :param third_title: disparity map title
    :type third_title: str
    :param error_map: error map
    :type error_map: np.array
    :param error_title: error title
    :type error_title: str
    :return: none
    """
    output_notebook()
    size = 0.5

    first_disp_map = add_validity_mask_to_dataset(input_first_disp_map)
    second_disp_map = add_validity_mask_to_dataset(input_second_disp_map)
    third_disp_map = add_validity_mask_to_dataset(input_third_disp_map)

    min_d = np.nanmin(first_disp_map['disparity_map'].data)
    max_d = np.nanmax(first_disp_map['disparity_map'].data)
    mapper_avec_opti = LinearColorMapper(palette='Viridis256', low=min_d, high=max_d)

    color_bar = ColorBar(color_mapper=mapper_avec_opti, ticker=BasicTicker(),
                         label_standoff=12, border_line_color=None, location=(0, 0))

    dw = first_disp_map['disparity_map'].shape[1]
    dh = first_disp_map['disparity_map'].shape[0]

    # First disparity map
    first_fig = figure(title=first_title, width=400, height=400,
                       tools=['reset', 'pan', "box_zoom"], output_backend="webgl")

    first_fig.image(image=[np.flip(first_disp_map['disparity_map'].data, 0)], x=1, y=0, dw=dw, dh=dh,
                    color_mapper=mapper_avec_opti)

    x = np.where(first_disp_map['invalid_mask'].data != 0)[1]
    y = dh - np.where(first_disp_map['invalid_mask'].data != 0)[0]
    first_inv_msk = first_fig.circle(x=x, y=y, size=size, color="red")
    legend = Legend(items=[
        ("inv msk", [first_inv_msk])],
        location="center", click_policy="hide")
    first_fig.add_layout(legend, 'below')
    first_fig.add_layout(color_bar, 'right')

    # Second disparity map
    second_fig = figure(title=second_title, width=400, height=400,
                        tools=['reset', 'pan', "box_zoom"], output_backend="webgl", x_range=first_fig.x_range,
                        y_range=first_fig.y_range)

    second_fig.image(image=[np.flip(second_disp_map['disparity_map'].data, 0)], x=1, y=0, dw=dw, dh=dh,
                     color_mapper=mapper_avec_opti)

    x = np.where(second_disp_map['invalid_mask'].data != 0)[1]
    y = dh - np.where(second_disp_map['invalid_mask'].data != 0)[0]
    second_inv_msk = second_fig.circle(x=x, y=y, size=size, color="red")
    legend = Legend(items=[
        ("inv msk", [second_inv_msk])],
        glyph_height=10,
        glyph_width=10, location="center", click_policy="hide")
    second_fig.add_layout(legend, 'below')
    second_fig.add_layout(color_bar, 'right')

    # Third disparity map
    third_fig = figure(title=third_title, width=400, height=400,
                       tools=['reset', 'pan', "box_zoom"], output_backend="webgl", x_range=first_fig.x_range,
                       y_range=first_fig.y_range)

    third_fig.image(image=[np.flip(third_disp_map['disparity_map'].data, 0)], x=1, y=0, dw=dw, dh=dh,
                    color_mapper=mapper_avec_opti)

    x = np.where(third_disp_map['invalid_mask'].data != 0)[1]
    y = dh - np.where(third_disp_map['invalid_mask'].data != 0)[0]
    third_inv_msk = third_fig.circle(x=x, y=y, size=size, color="red")

    legend = Legend(items=[
        ("inv msk", [third_inv_msk])],
        glyph_height=10,
        glyph_width=10, location="center", click_policy="hide")
    third_fig.add_layout(legend, 'below')
    third_fig.add_layout(color_bar, 'right')

    # Error plot
    error_fig = figure(title=error_title, width=400, height=400,
                       tools=['reset', 'pan', "box_zoom"], output_backend="webgl", x_range=first_fig.x_range,
                       y_range=first_fig.y_range)
    min_d = np.nanmin(error_map)
    max_d = np.nanmax(error_map)
    mapper_avec_opti = LinearColorMapper(palette='Reds256', low=min_d, high=max_d)

    color_bar = ColorBar(color_mapper=mapper_avec_opti, ticker=BasicTicker(),
                         label_standoff=12, border_line_color=None, location=(0, 0))
    error_fig.image(image=[np.flip(error_map, 0)], x=1, y=0, dw=dw, dh=dh, color_mapper=mapper_avec_opti)
    error_fig.add_layout(color_bar, 'right')

    layout = column(row(first_fig, second_fig), row(third_fig, error_fig))

    show(layout)


def compare_disparity_and_error(input_first_disp_map: xr.Dataset, first_title: str, error_map: np.array,
                                error_title: str) -> None:
    """
    Show disparity map and error
    :param left_disp_map: disparity map
    :type left_disp_map: dataset
    :param title: disparity map title
    :type title: str
    :param error_map: error map
    :type error_map: np.array
    :param error_title: error title
    :type error_title: str
    :return: none
    """
    output_notebook()
    size = 0.5

    # Disparity map
    first_disp_map = add_validity_mask_to_dataset(input_first_disp_map)

    min_d = np.nanmin(first_disp_map['disparity_map'].data)
    max_d = np.nanmax(first_disp_map['disparity_map'].data)
    mapper_avec_opti = LinearColorMapper(palette='Viridis256', low=min_d, high=max_d)

    color_bar = ColorBar(color_mapper=mapper_avec_opti, ticker=BasicTicker(),
                         label_standoff=12, border_line_color=None, location=(0, 0))

    dw = first_disp_map['disparity_map'].shape[1]
    dh = first_disp_map['disparity_map'].shape[0]

    first_fig = figure(title=first_title, width=400, height=400,
                       tools=['reset', 'pan', "box_zoom"], output_backend="webgl")
    first_fig.image(image=[np.flip(first_disp_map['disparity_map'].data, 0)], x=1, y=0, dw=dw, dh=dh,
                    color_mapper=mapper_avec_opti)

    x = np.where(first_disp_map['invalid_mask'].data != 0)[1]
    y = dh - np.where(first_disp_map['invalid_mask'].data != 0)[0]
    first_inv_msk = first_fig.circle(x=x, y=y, size=size, color="red")
    legend = Legend(items=[
        ("inv msk", [first_inv_msk])],
        location="center", click_policy="hide")

    first_fig.add_layout(legend, 'below')
    first_fig.add_layout(color_bar, 'right')

    # Error plot
    error_fig = figure(title=error_title, width=400, height=400,
                       tools=['reset', 'pan', "box_zoom"], output_backend="webgl", x_range=first_fig.x_range,
                       y_range=first_fig.y_range)
    min_d = np.nanmin(error_map)
    max_d = np.nanmax(error_map)
    mapper_avec_opti = LinearColorMapper(palette='Reds256', low=min_d, high=max_d)

    color_bar = ColorBar(color_mapper=mapper_avec_opti, ticker=BasicTicker(),
                         label_standoff=12, border_line_color=None, location=(0, 0))
    error_fig.image(image=[np.flip(error_map, 0)], x=1, y=0, dw=dw, dh=dh, color_mapper=mapper_avec_opti)
    error_fig.add_layout(color_bar, 'right')

    layout = column(row(first_fig, error_fig))
    show(layout)


def show_input_images(img_left: xr.Dataset, img_right: xr.Dataset) -> None:
    """
    Show input images and anaglyph
    :param left_disp_map: disparity map
    :type left_disp_map: dataset
    :param right_disp_map: right disparity map
    :type right_disp_map: dataset
    :return: none
    """
    output_notebook()

    dw = np.flip(img_left.im.data, 0).shape[1]
    dh = np.flip(img_right.im.data, 0).shape[0]
    width = 300
    height = 300

    # Image left
    img_left_data = img_left.im.data
    left_fig = bpl.figure(title="Left image", width=width, height=height)
    left_fig.image(image=[np.flip(img_left_data, 0)], x=1, y=0, dw=dw, dh=dh)

    # Image right
    img_right_data = img_right.im.data
    right_fig = bpl.figure(title="Right image", width=width, height=height, x_range=left_fig.x_range,
                           y_range=left_fig.y_range)
    right_fig.image(image=[np.flip(img_right_data, 0)], x=1, y=0, dw=dw, dh=dh)
    # Anaglyph
    img_left, img_right_align = xr.align(img_left, img_right)
    anaglyph_fig = bpl.figure(title="Anaglyph", width=width, height=height, x_range=left_fig.x_range,
                              y_range=left_fig.y_range)
    anaglyph_fig.image(image=[np.flip((img_left_data + img_right_data * 0.6), 0)], x=1, y=0, dw=dw, dh=dh)

    layout = column(row(left_fig, right_fig, anaglyph_fig))
    show(layout)


class ErrorPlot:

    def init_error(self, left_disp_map: xr.Dataset, ground_truth: xr.Dataset, title: str) -> None:
        """
        Initialize and plot error plot with threshold 1
        :param left_disp_map: disparity map
        :type left_disp_map: dataset
        :param ground_truth: ground truth
        :type ground_truth: dataset
        :param title: title
        :type title: str
        :return: none
        """
        output_notebook()
        # Initialize plot's parameters
        self.left_disp_map = left_disp_map
        self.ground_truth = ground_truth
        self.dw = left_disp_map['disparity_map'].shape[1]
        self.dh = left_disp_map['disparity_map'].shape[0]
        # Initialize error on default threshold 1
        total_bad_percentage, mean_error, std_error, invalid_percentage, error = compare_to_gt(left_disp_map,
                                                                                               ground_truth, 1, None)

        min_d = np.nanmin(error)
        max_d = np.nanmax(error)
        self.mapper_avec_opti = LinearColorMapper(palette='Reds256', low=min_d, high=max_d)

        color_bar = ColorBar(color_mapper=self.mapper_avec_opti, ticker=BasicTicker(),
                             label_standoff=12, border_line_color=None, location=(0, 0))
        # Show error plot
        self.error_plot = figure(title=title, width=500, height=450,
                                 tools=['reset', 'pan', "box_zoom"], output_backend="webgl")
        self.error_plot.image(image=[np.flip(error, 0)], x=1, y=0, dw=self.dw, dh=self.dh,
                              color_mapper=self.mapper_avec_opti)
        self.error_plot.add_layout(color_bar, 'right')

        show(self.error_plot, notebook_handle=True)

    def update(self, threshold: int = 1) -> None:
        """
        Update error plot when threshold is modified

        :param threshold: error threshold
        :type threshold: int
        :return: none
        """
        # Re calculate error on input threshold
        total_bad_percentage, mean_error, std_error, invalid_percentage, error = compare_to_gt(self.left_disp_map,
                                                                                               self.ground_truth,
                                                                                               threshold, None)
        # Update error plot and statistics
        self.error_plot.image(image=[np.flip(error, 0)], x=1, y=0, dw=self.dw, dh=self.dh,
                              color_mapper=self.mapper_avec_opti)

        print("Threshold = {0:.2f}".format(threshold))
        print("Total bad error point percentage = {0:.2f}%".format(total_bad_percentage))
        print("Mean error = {0:.2f}".format(mean_error))
        print("Invalid percentage = {0:.2f}".format(invalid_percentage))
        print("Std error = {0:.2f}".format(std_error))

        push_notebook()


def error_plot(left_disp_map: xr.Dataset, ground_truth: xr.Dataset, title: str) -> None:
    """
    Plot error map with interactive threshold

    :param left_disp_map: disparity map
    :type left_disp_map: dataset
    :param ground_truth: ground truth
    :type ground_truth: dataset
    :param title: title
    :type title: str
    :return: none
    """
    # Create error object and initialize error plot
    error = ErrorPlot()
    error.init_error(left_disp_map, ground_truth, title)
    # Create interactive slider for threshold
    interact(error.update, threshold=(1, 50))


def get_error(left_disp_map: xr.Dataset, ground_truth: xr.Dataset, threshold: int = 1) -> np.array:
    """
    Return error map

    :param left_disp_map: disparity map
    :type left_disp_map: dataset
    :param ground_truth: ground truth
    :type ground_truth: dataset
    :param threshold: error threshold
    :type threshold: int
    :return: error_map
    :rtype: np.array
    """
    total_bad_percentage, mean_error, std_error, invalid_percentage, error_map = compare_to_gt(left_disp_map,
                                                                                               ground_truth, threshold,
                                                                                               None)
    return error_map


def plot_1_cost_volume(cv: xr.Dataset, left_disp_map: xr.Dataset, title: str) -> None:
    """
    Plot 3d cost volume

    :param cv: cost volume
    :type cv: dataset
    :param left_disp_map: disparity map
    :type left_disp_map: dataset
    :return: none
    """
    print(title)
    get_3D_cost_volume(cv, left_disp_map)
    ipv.show()

def get_3D_cost_volume(cv: xr.Dataset, left_disp_map: xr.Dataset) -> None:
    """
    Plot 3d cost volume

    :param cv: cost volume
    :type cv: dataset
    :param left_disp_map: disparity map
    :type left_disp_map: dataset
    :return: none
    """

    nb_rows, nb_cols, nb_disps = cv['cost_volume'].shape

    X,Y = np.meshgrid(np.arange(nb_cols), np.arange(nb_rows))
    X = np.float32(X)
    Y = np.float32(Y)
    Z = left_disp_map['disparity_map'].data

    color_disp = np.ravel(Z)
    color_disp = color_disp - np.nanmin(color_disp)
    color_disp = color_disp * 1. / np.nanmax(color_disp)
    color_disp = np.repeat(color_disp[:, np.newaxis], 3, axis=1)

    fig = ipv.figure()
    scatter = ipv.scatter(np.ravel(X),np.ravel(Y),np.ravel(Z),marker='point_2d', size=10, color=color_disp)
    ipv.ylim(nb_rows,0)
    ipv.style.box_off()
    ipv.style.use('minimal')
    
    return ipv.gcc()    

def add_mask(all_validity_mask: np.array, msk_type: int) -> np.array:
    """
    Create mask for a given bit

    :param all_validity_mask: mask for all bits
    :type all_validity_mask: np.array
    :return: msk
    :rtype: np.array
    """
    # Mask initialization to 0 (all valid)
    msk = np.full(all_validity_mask.shape, 0)
    # Identify and fill invalid points
    inv_idx = np.where((all_validity_mask & msk_type) != 0)
    msk[inv_idx] = 1
    return msk


def add_validity_mask_to_dataset(input_disp_map: xr.Dataset) -> xr.Dataset:
    """
    Adds validity mask to imput dataset

    :param input_disp_map: disparity map
    :type input_disp_map: dataset
    :return: input_disp_map
    :rtype: dataset
    """

    disp_map = copy.deepcopy(input_disp_map)

    # Invalid
    disp_map['invalid_mask'] = xr.DataArray(
        np.copy(add_mask(disp_map['validity_mask'].values, PANDORA_MSK_PIXEL_INVALID)),
        dims=['row', 'col'])
    # Bit 0: Edge of the left image or nodata in left image
    disp_map['nodata_border_left_mask'] = xr.DataArray(np.copy(add_mask(disp_map['validity_mask'].values,
                                                                        PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER)),
                                                       dims=['row', 'col'])
    # Bit 1: Disparity interval to explore is missing or nodata in the right image
    disp_map['nodata_border_right_mask'] = xr.DataArray(np.copy(add_mask(disp_map['validity_mask'].values,
                                                                         PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING)),
                                                        dims=['row', 'col'])
    # Bit 2: Incomplete disparity interval in right image
    disp_map['incomplete_right_mask'] = xr.DataArray(np.copy(add_mask(disp_map['validity_mask'].values,
                                                                      PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE)),
                                                     dims=['row', 'col'])
    # Bit 3: Unsuccesful sub-pixel interpolation
    disp_map['stopped_interp_mask'] = xr.DataArray(np.copy(add_mask(disp_map['validity_mask'].values,
                                                                    PANDORA_MSK_PIXEL_STOPPED_INTERPOLATION)),
                                                   dims=['row', 'col'])
    # Bit 4: Filled occlusion 
    disp_map['filled_occlusion_mask'] = xr.DataArray(np.copy(add_mask(disp_map['validity_mask'].values,
                                                                      PANDORA_MSK_PIXEL_FILLED_OCCLUSION)),
                                                     dims=['row', 'col'])
    # Bit 5: Filled mismatch
    disp_map['filled_mismatch_mask'] = xr.DataArray(np.copy(add_mask(disp_map['validity_mask'].values,
                                                                     PANDORA_MSK_PIXEL_FILLED_MISMATCH)),
                                                    dims=['row', 'col'])
    # Bit 6: Pixel is masked on the mask of the left image
    disp_map['masked_left_mask'] = xr.DataArray(np.copy(add_mask(disp_map['validity_mask'].values,
                                                                 PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT)),
                                                dims=['row', 'col'])
    # Bit 7: Disparity to explore is masked on the mask of the right image
    disp_map['masked_right_mask'] = xr.DataArray(np.copy(add_mask(disp_map['validity_mask'].values,
                                                                  PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT)),
                                                 dims=['row', 'col'])
    # Bit 8: Pixel located in an occlusion region
    disp_map['occlusion_mask'] = xr.DataArray(np.copy(add_mask(disp_map['validity_mask'].values,
                                                               PANDORA_MSK_PIXEL_OCCLUSION)),
                                              dims=['row', 'col'])
    # Bit 9: Mismatch
    disp_map['mismatch_mask'] = xr.DataArray(np.copy(add_mask(disp_map['validity_mask'].values,
                                                              PANDORA_MSK_PIXEL_MISMATCH)),
                                             dims=['row', 'col'])
    # Bit 10: Filled nodata
    disp_map['filled_nodata'] = xr.DataArray(np.copy(add_mask(disp_map['validity_mask'].values,
                                                              PANDORA_MSK_PIXEL_FILLED_NODATA)),
                                             dims=['row', 'col'])
    return disp_map


def compare_to_gt(disp_map: xr.Dataset, ground_truth: xr.Dataset, error_threshold: int,
                  no_data_gt_value: float = None) -> \
        Tuple[float, float, float, float, np.array]:
    """
    Compute difference between a disparity map (estimated by a stereo tool) and ground_truth.
    Point p is considered as an error if disp_map(p)-ground_truth(p) > threshold
    Statistics (mean, median, standard deviation) are computed regarded error points

    :param disp: disparity map
    :type disp: dataset
    :param ground_truth: ground_truth
    :type ground_truth: dataset
    :param error_threshold: threshold
    :type error_threshold: int
    :param no_data_gt_value: value of ground truth no data
    :type no_data_gt_value: float
    :param invalid_point: True if disparity map contains invalid value (must be NaN)
    :type invalid_point: bool
    :return:
            - total_bad_percentage
            - mean_error
            - std_error
            - map error
    :rtype: float, float, float, 2d numpy array
    """
    disp = disp_map.disparity_map.data
    gt = ground_truth.disparity_map.data
    # Compare Sizes
    if disp.size != gt.size:
        raise ValueError("Ground truth and disparity map must have the same size")

    # Difference between disp_map and ground truth
    error = abs(disp - gt)
    # Do not consider errors lower than the error threshold
    error[np.where(error < error_threshold)] = 0
    # Number of points
    num_points = disp.shape[0] * disp.shape[1]

    # If occlusion mask exists, number of occlusion points is computed.
    # Occlusion points become NaN on error array
    num_occl = 0
    if ground_truth.validity_mask is not None:
        mask = ground_truth.validity_mask
        # Occlusion point value must be different to 0
        occl_coord = np.where(mask != 0)
        num_occl = len(occl_coord[0])
        error[occl_coord] = np.nan
    else:
        mask = np.zeros(ground_truth.disparity_map.shape)

    # All no_data_gt_values become NaN value
    num_no_data_gt = 0
    if no_data_gt_value is not None:
        if no_data_gt_value == np.inf:
            no_data_coord = np.where(np.isinf(gt) & (mask == 0))
        else:
            no_data_coord = np.where((gt == no_data_gt_value) & (mask == 0))
        num_no_data_gt = len(no_data_coord[0])
        error[no_data_coord] = np.nan

    # Invalid point on disparity map
    invalid_coord = np.where(np.isnan(disp) & (mask == 0) & (gt != no_data_gt_value))
    num_invalid = len(invalid_coord[0])
    error[invalid_coord] = np.nan
    # Number of bad points
    bad_coord = np.where(error > 0)
    num_bad = len(bad_coord[0])

    # Percentage of total bad points (bad + invalid)
    total_bad_percentage = ((num_bad + num_invalid) / float(num_points - num_no_data_gt - num_occl)) * 100

    inf_idx = np.where(np.isinf(error))
    error[inf_idx] = np.nan
    # Mean error
    mean_error = float(np.nanmean(error))
    # Standard deviation
    std_error = float(np.nanstd(error))
    # Percentage of invalid points
    invalid_percentage = (num_invalid / float(num_points)) * 100.0

    return total_bad_percentage, mean_error, std_error, invalid_percentage, error
