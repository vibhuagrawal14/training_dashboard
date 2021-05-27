<p align="center">
  
  <img width="600" src="https://i.imgur.com/pl6mUzn.png"/>
  <br>
  <strong>A no-BS, dead-simple training visualizer for tf-keras</strong>
  <br>
  <a href="https://badge.fury.io/py/training-dashboard"><img src="https://badge.fury.io/py/training-dashboard.svg" alt="PyPI version" height="18"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="PyPI version" height="18"></a>
  
</p>

# TrainingDashboard

Plot inter-epoch and intra-epoch loss and metrics within a jupyter notebook with a simple callback. Features:
* Plots the training loss and a training metric, updated at the end of each batch
* Plots training and validation losses, updated at the end of each epoch
* For each metric, plots training and validation values, updated at the end of each epoch
* Tabulates losses and metrics (both train and validation) and highlights the highest and lowest values in each column  
      
<p align="center">
  <img width="900" src="https://i.imgur.com/SBdQurw.gif"/>
</p>
  
**Why should I use this over tensorboard?**  
This is way simpler to use.

**What about `livelossplot`?**  
AFAIK, `livelossplot` does not support intra-epoch loss/metric plotting. Also, `TrainingDashboard` uses `bqplot` for plotting, which provides support for much more interactive elements like tooltips (currently a TODO). On the other hand, `livelossplot` is a much more mature project, and you should use it if you have a specific use case. 


## Installation 

TrainingDashboard can be installed from PyPI with the following command:

```bash 
pip install training-dashboard
```

Alternatively, you can clone this repository and run the following command from the root directory:

```bash
pip install .
```

## Usage

TrainingDashboard is a tf-keras callback and should be used as such. It takes the following optional arguments:
- `validation` (bool): whether validation data is being used or not
- `min_loss` (float): the minimum possible value of the loss function, to fix the lower bound of the y-axis
- `max_loss` (float): the maximum possible value of the loss function, to fix the upper bound of the y-axis
- `metrics` (list): list of metrics that should be considered for plotting
- `min_metric_dict` (dict): dictionary mapping each (or a subset) of the metrics to their minimum possible value, to fix the lower bound of the y-axis
- `max_metric_dict` (dict): dictionary mapping each (or a subset) of the metrics to their maximum possible value, to fix the upper bound of the y-axis
- `batch_step` (int): step size for plotting the results within each epoch. If the time to process each batch is very small, plotting at each step may cause the training to slow down significantly. In such cases, it is advisable to skip a few batches between each update.

```python
from training_dashboard import TrainingDashboard
model.fit(X,
          Y,
          epochs=10,
          callbacks=[TrainingDashboard()])
```

or, a more elaborate example:
```python
from training_dashboard import TrainingDashboard
dashboard = TrainingDashboard(validation=True, # because we are using validation data and want to track its metrics
                             min_loss=0, # we want the loss axes to be fixed on the lower end
                             metrics=["accuracy", "auc"], # metrics that we want plotted
                             batch_step=10, # plot every 10th batch
                             min_metric_dict={"accuracy": 0, "auc": 0}, # minimum possible value for metrics used
                             max_metric_dict={"accuracy": 1, "auc": 1}) # maximum possible value for metrics used
model.fit(x_train,
          y_train,
          batch_size=512,
          epochs=25,
          verbose=1,
          validation_split=0.2,
          callbacks=[dashboard])
```

For a more detailed example, check `mnist_example.ipynb` inside the `examples` folder.

## Support

Reach out to me at one of the following places!

Twitter: @vibhuagrawal  
Email: vibhu[dot]agrawal14[at]gmail

## License  

Project is distributed under [MIT License](https://github.com/vibhuagrawal14/training_dashboard/blob/main/LICENSE).
