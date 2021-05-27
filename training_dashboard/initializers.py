import bqplot as bq
from ipywidgets import Output


def initialize_live_epoch_plot(metric=None,
                               colors=None,
                               min_loss=None,
                               max_loss=None,
                               min_metric=None,
                               max_metric=None):
    """
    Initialize the plot for training loss and metrics within the current epoch. This chart resets at the end of each
     epoch.

    :param metric: the metric to be plotted with the loss. If there are more than one metrics passed to the callback,
    the metric received here is the first one. TODO: give this choice to the user
    :param colors: colors for the loss and metric line plots
    :param min_loss: minimum bound for loss to fix y-axis minimum
    :param max_loss: maximum bound for loss to fix y-axis maximum
    :param min_metric: minimum bound for metric to fix y-axis minimum
    :param max_metric: maximum bound for metric to fix y-axis maximum
    :return: a bqplot Figure object
    """

    if colors is None:
        colors = {"loss": "red", "metric": "blue"}

    # create scales
    x_axis_scale = bq.OrdinalScale()
    y_axis_scale = bq.LinearScale()
    if min_loss is not None:
        y_axis_scale.min = min_loss
    if max_loss is not None:
        y_axis_scale.max = max_loss

    # create loss line and corresponding axes
    loss_line = bq.Lines(x=[],
                         y=[],
                         scales={"x": x_axis_scale, "y": y_axis_scale},
                         colors=[colors["loss"]],
                         labels=["loss"],
                         display_legend=True)

    x_axis = bq.Axis(scale=x_axis_scale, label='batch', grid_lines='solid')
    y_axis_loss = bq.Axis(scale=y_axis_scale, label='training_loss', grid_lines='dashed', orientation="vertical")

    if metric:
        y_axis_metric_scale = bq.LinearScale()
        if min_metric is not None:
            y_axis_metric_scale.min = min_metric
        if max_metric is not None:
            y_axis_metric_scale.max = max_metric
        metric_line = bq.Lines(x=[],
                               y=[],
                               scales={"x": x_axis_scale, "y": y_axis_metric_scale},
                               colors=[colors["metric"]],
                               labels=[metric],
                               display_legend=True)
        y_axis_metric = bq.Axis(scale=y_axis_metric_scale, label="training_" + metric, grid_lines='solid',
                                orientation="vertical", side='right')

    marks = [loss_line, metric_line] if metric else [loss_line]

    fig = bq.Figure(marks=marks,
                    axes=[x_axis, y_axis_loss, y_axis_metric] if metric else [x_axis, y_axis_loss],
                    animation_duration=0,
                    legend_location="bottom",
                    title=f"Epoch 0")
    return fig


def initialize_loss_plot(validation=False,
                         colors=None,
                         min_loss=None,
                         max_loss=None):
    """
    Initialize the plot for training and validation losses.

    :param validation: flag, whether validation data is being used or not. If false, will not plot validation loss
    :param colors: colors for the loss and metric line plots
    :param min_loss: minimum bound for loss to fix y-axis minimum
    :param max_loss: maximum bound for loss to fix y-axis maximum
    :return: a bqplot Figure object
    """
    if colors is None:
        colors = {"train": "red", "val": "green"}

    # create scales
    x_axis_scale = bq.OrdinalScale()
    y_axis_scale = bq.LinearScale()
    if min_loss is not None:
        y_axis_scale.min = min_loss
    if max_loss is not None:
        y_axis_scale.max = max_loss

    # create loss line and corresponding axes
    loss_line = bq.Lines(x=[],
                         y=[],
                         scales={"x": x_axis_scale, "y": y_axis_scale},
                         colors=[colors["train"]],
                         labels=["train"],
                         display_legend=True)

    if validation:
        val_loss_line = bq.Lines(x=[],
                                 y=[],
                                 scales={"x": x_axis_scale, "y": y_axis_scale},
                                 colors=[colors["val"]],
                                 labels=["val"],
                                 display_legend=True)
    x_axis = bq.Axis(scale=x_axis_scale, label='epoch', grid_lines='solid')
    y_axis = bq.Axis(scale=y_axis_scale, label='loss', grid_lines='solid', orientation="vertical")

    marks = [loss_line, val_loss_line] if validation else [loss_line]

    fig = bq.Figure(marks=marks,
                    axes=[x_axis, y_axis],
                    animation_duration=0,
                    legend_location="bottom",
                    title="Loss")
    return fig


def initialize_metrics_plots(metrics=None,
                             validation=False,
                             colors=None,
                             min_metric_dict=None,
                             max_metric_dict=None):
    """
    Initialize the charts for training and validation metrics.

    :param metrics: list of metrics which will be plotted
    :param validation: flag, whether validation data is being used or not. If false, will not plot validation loss
    :param colors: colors for the loss and metric line plots
    :param min_metric_dict: dictionary mapping of metric to minimum bound to fix y-axis minimum
    :param max_metric_dict: dictionary mapping of metric to maximum bound to fix y-axis maximum
    :return: a bqplot Figure object
    """

    if max_metric_dict is None:
        max_metric_dict = {}
    if min_metric_dict is None:
        min_metric_dict = {}
    if colors is None:
        colors = {"train": "red", "val": "green"}
    if metrics is None:
        metrics = []
    figs = []

    # create a separate figure for each metric
    for metric in metrics:
        x_axis_scale = bq.OrdinalScale()
        y_axis_scale = bq.LinearScale()
        if min_metric_dict.get(metric, None) is not None:
            y_axis_scale.min = min_metric_dict[metric]
        if max_metric_dict.get(metric, None) is not None:
            y_axis_scale.max = max_metric_dict[metric]

        train_metric_line = bq.Lines(x=[],
                                     y=[],
                                     scales={"x": x_axis_scale, "y": y_axis_scale},
                                     colors=[colors["train"]],
                                     labels=["train"],
                                     display_legend=True)
        if validation:
            val_metric_line = bq.Lines(x=[],
                                       y=[],
                                       scales={"x": x_axis_scale, "y": y_axis_scale},
                                       colors=[colors["val"]],
                                       labels=["val"],
                                       display_legend=True)

        x_axis = bq.Axis(scale=x_axis_scale, label='epoch', grid_lines='solid')
        y_axis = bq.Axis(scale=y_axis_scale, label=metric, grid_lines='solid', orientation="vertical")

        marks = [train_metric_line, val_metric_line] if validation else [train_metric_line]

        fig = bq.Figure(marks=marks,
                        axes=[x_axis, y_axis],
                        animation_duration=0,
                        legend_location="bottom",
                        title=metric)
        figs.append(fig)
    return figs


def initialize_dataframe_output():
    """
    Initialize the output scope for history dataframe.
    :return: an Output object
    """
    return Output()
