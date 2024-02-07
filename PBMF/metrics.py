import tensorflow as tf
from .loss import PredictiveBiomarkerLoss
import pandas as pd
from lifelines import CoxPHFitter

def get_hr(y_true, y_pred):
    yp = tf.cast((-1*y_pred[:, 0] > 0) & (y_true[:, 3] == 1), tf.float32)

    if tf.reduce_mean(tf.math.multiply(y_true[:, 0] , yp)) > tf.reduce_mean(tf.math.multiply(y_true[:, 0] , (1-yp))):
        y_pred = tf.cast(y_pred > 0, dtype=tf.float32)
    else:
        y_pred = tf.cast(y_pred < 0, dtype=tf.float32)

    exposed_data =   y_true[(y_pred[:, 0] == 1) & (y_true[:, 3] == 1)] # B+ Treat
    unexposed_data = y_true[(y_pred[:, 0] == 0) & (y_true[:, 3] == 1)] # B- Treat

    exposed_events = tf.reduce_sum(exposed_data[:, 1])
    exposed_time = tf.reduce_sum(exposed_data[:, 0])
    exposed_hazard_rate = exposed_events / exposed_time

    unexposed_events = tf.reduce_sum(unexposed_data[:, 1])
    unexposed_time = tf.reduce_sum(unexposed_data[:, 0])
    unexposed_hazard_rate = unexposed_events / unexposed_time

    hr = exposed_hazard_rate / unexposed_hazard_rate

    if (tf.math.is_nan(hr) == True) or (hr < 0.1):
        return 1.
    else:
        return hr

class MeanTimeMetric(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        # Specify the name of the metric as "custom_metric".
        super().__init__(name="pseudo_hr", **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", dtype=tf.float32, initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        val = get_hr(y_true, y_pred)

        self.sum.assign_add(val)
        self.count.assign_add(1.)

    def result(self):
        return self.sum / self.count

    def reset_state(self):
        self.sum.assign(0)
        self.count.assign(0)