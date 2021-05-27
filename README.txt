# Training Dashboard
A no bullshit, dead simple training visualizer for tf-keras.


## Usage

TrainingDashboard is meant to be used as a callback passed to the fit() function.

```python
  from training_dashboard import TrainingDashboard
  callback = TrainingDashboard(validation=True,
                               min_loss=0,
                               metrics=["accuracy", "auc"],
                               batch_step=10,
                               min_metric_dict={"accuracy": 0, "auc": 0},
                               max_metric_dict={"accuracy": 1, "auc": 1})
  model.fit(x_train,
            y_train,
            batch_size=512,
            epochs=25,
            verbose=1,
            validation_split=0.2,
            callbacks=[callback])
```

## Example Output

<img src="https://i.imgur.com/D91YmwT.gif" width="700"/>

