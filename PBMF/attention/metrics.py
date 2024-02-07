import tensorflow as tf
from tensorflow.keras import Model
import pandas as pd
import numpy as np
import lifelines
from lifelines import KaplanMeierFitter

def logrank_hazard_ratio_treatment(y_true, y_pred, treatment=1):
    B_plus = y_pred[:,0] # B+
    B_minus = y_pred[:,1] # B-
    
    treatment_A = tf.cast(y_true[:,3] == treatment, tf.float32) # Target treatment
    
    O=y_true[:, 1]
    E=y_true[:, 2]
    
    #t-logrank for Treatment arm A (Treatment)
    O_plus = tf.reduce_sum(B_plus * O * treatment_A)
    E_plus = tf.reduce_sum(B_plus * E * treatment_A)
    
    O_minus = tf.reduce_sum(B_minus * O * treatment_A)
    E_minus = tf.reduce_sum(B_minus * E * treatment_A)
    

    HR = (O_plus / E_plus) / (O_minus / E_minus)
    
    return HR

def logrank_hazard_ratio_control(y_true, y_pred, treatment=0):
    B_plus = y_pred[:,0] # B+
    B_minus = y_pred[:,1] # B-
    
    treatment_A = tf.cast(y_true[:,3] == treatment, tf.float32) # Target treatment
    
    O=y_true[:, 1]
    E=y_true[:, 2]
    
    #t-logrank for Treatment arm A (Treatment)
    O_plus = tf.reduce_sum(B_plus * O * treatment_A)
    E_plus = tf.reduce_sum(B_plus * E * treatment_A)
    
    O_minus = tf.reduce_sum(B_minus * O * treatment_A)
    E_minus = tf.reduce_sum(B_minus * E * treatment_A)
    

    HR = (O_plus / E_plus) / (O_minus / E_minus)
    
    return HR