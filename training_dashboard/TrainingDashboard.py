import pandas as pd
from ipywidgets import AppLayout, Layout, HBox
import tensorflow as tf
from .initializers import (initialize_loss_plot, initialize_live_epoch_plot, initialize_metrics_plots,
                           initialize_dataframe_output)
from .utils import highlight_min_max


class TrainingDashboard(tf.keras.callbacks.Callback):
    def __init__(self, validation=False, min_loss=None, max_loss=None, metrics=None, min_metric_dict=None,
                 max_metric_dict=None, batch_step=10):
        """
        :param validation: flag, whether validation data is being used or not. If false, will not plot validation losses
        and metrics
        :param min_loss: minimum bound for loss to fix y-axis minimum
        :param max_loss: maximum bound for loss to fix y_axis maximum
        :param metrics: list of metrics which will be plotted
        :param min_metric_dict: dictionary mapping of metric to minimum bound to fix y-axis minimum
        :param max_metric_dict: dictionary mapping of metric to maximum bound to fix y-axis maximum
        :param batch_step: step size for live plotting batch-level loss and metrics. Too small a number would impact
        training speed.
        """

        super().__init__()
        if max_metric_dict is None:
            max_metric_dict = {}
        if min_metric_dict is None:
            min_metric_dict = {}
        if metrics is None:
            metrics = []

        self.validation = validation
        self.metrics = metrics
        self.batch_step = batch_step
        self.loss_plot = initialize_loss_plot(validation=validation,
                                              min_loss=min_loss,
                                              max_loss=max_loss)
        self.live_epoch_plot = initialize_live_epoch_plot(metric=metrics[0] if metrics else None,
                                                          min_loss=min_loss,
                                                          max_loss=max_loss,
                                                          min_metric=min_metric_dict.get(metrics[0],
                                                                                         None) if metrics else None,
                                                          max_metric=max_metric_dict.get(metrics[0],
                                                                                         None) if metrics else None)
        self.metrics_plots = initialize_metrics_plots(metrics,
                                                      validation=validation,
                                                      min_metric_dict=min_metric_dict,
                                                      max_metric_dict=max_metric_dict)
        self.dataframe_output = initialize_dataframe_output()
        self.loss_history = {"loss": [], "val_loss": []}
        self.metrics_history = {mode + metric: [] for mode in ["", "val_"] for metric in metrics}
        self.history_dataframe = pd.DataFrame({**self.loss_history, **self.metrics_history})

    def _display_plots(self):

        layouts = [AppLayout(left_sidebar=self.live_epoch_plot, center=self.loss_plot, pane_widths=[1, 1, 0])]

        for i in range(0, len(self.metrics_plots), 2):
            if i + 1 < len(self.metrics_plots):
                layouts.append(
                    AppLayout(left_sidebar=self.metrics_plots[i], center=self.metrics_plots[i + 1],
                              pane_widths=[1, 1, 0]))
            else:
                layouts.append(AppLayout(center=self.metrics_plots[i], pane_widths=[0.5, 1, 0.5]))
        for layout in layouts:
            display(layout)

        box_layout = Layout(display='flex',
                            flex_flow='column',
                            align_items='center')
        box = HBox(children=[self.dataframe_output], layout=box_layout)
        display(box)

        with self.dataframe_output:
            self.dataframe_output.clear_output()
            display(self.history_dataframe.style.apply(highlight_min_max))

    def on_train_batch_end(self, batch, logs=None):
        # we consider only the first metric if a list of metrics has been provided
        if batch % self.batch_step == 0:
            # update x-axis
            for mark in self.live_epoch_plot.marks:
                mark.x = list(mark.x) + [batch]

            # update y-axis
            self.live_epoch_plot.marks[0].y = list(self.live_epoch_plot.marks[0].y) + [logs["loss"]]

            if self.metrics:
                self.live_epoch_plot.marks[1].y = list(self.live_epoch_plot.marks[1].y) + [logs[self.metrics[0]]]

    def on_train_begin(self, logs=None):
        self._display_plots()

    def on_epoch_begin(self, epoch, logs=None):
        # clear live epoch plot
        for mark in self.live_epoch_plot.marks:
            mark.x = []
            mark.y = []

        # set live epoch plot title to display the current epoch number
        self.live_epoch_plot.title = f"Epoch {epoch + 1}"

    def on_epoch_end(self, epoch, logs=None):
        # update x-axis
        for plot in [self.loss_plot, *self.metrics_plots]:
            for mark in plot.marks:
                mark.x = list(mark.x) + [epoch]

        # update train/val loss y-axis
        self.loss_plot.marks[0].y = list(self.loss_plot.marks[0].y) + [logs["loss"]]
        if self.validation:
            self.loss_plot.marks[1].y = list(self.loss_plot.marks[1].y) + [logs["val_loss"]]

        # update train/val metrics y-axis
        for plot in self.metrics_plots:
            metric = plot.axes[1].label
            plot.marks[0].y = list(plot.marks[0].y) + [logs[metric]]
            if self.validation:
                plot.marks[1].y = list(plot.marks[1].y) + [logs["val_" + metric]]

        # update history
        self.loss_history["loss"].append(logs['loss'])
        self.loss_history["val_loss"].append(logs['val_loss'])

        for metric in self.metrics:
            for mode in ["", "val_"]:
                self.metrics_history[mode + metric].append(logs[mode + metric])

        self.history_dataframe = pd.DataFrame({**self.loss_history, **self.metrics_history})

        with self.dataframe_output:
            self.dataframe_output.clear_output()
            display(self.history_dataframe.style.apply(highlight_min_max))
