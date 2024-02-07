import tensorflow as tf
from tensorflow.keras import Model
import pandas as pd
import numpy as np
import lifelines
from lifelines import KaplanMeierFitter


# Lambdas are pre-computed constants for the loss function. 
# Check derivation notes for where to lambdas come from
def lambdas_(time, event):
    """
    lambda_i= Sum_t Omega_t / N_t * I(T_i>t)
    y is survival time with shape (n, )
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations=time, event_observed=event ) #fit km to your data

    #extract event table
    e_table=pd.DataFrame(kmf.event_table).reset_index() # reset index so that time is part of the data. 
    omegat_over_Nt=e_table["observed"]/e_table["at_risk"] # Omege_t/N_t term in the equation. 
    
    lamdas=np.zeros(time.shape[0]) #where to save lambda values 
    for ix, Ti in enumerate(time):

        lamdas[ix]=np.sum(omegat_over_Nt* ( Ti>e_table["event_at"]))
    return lamdas


# Predictive Loss Function

class PredictiveBiomarkerLossFunction:
    __name__ = 'PredictiveBiomarkerLoss'
    def __init__(self, omega=0.05):
        self.omega = omega

    def __call__(self, y_true, y_pred):
        return PredictiveBiomarkerLoss(y_true, y_pred, self.omega)


def PredictiveBiomarkerLoss(y_true, y_pred, omega=0.5):
    """
        :param y_true: n x 4 tensor: n, (time, event, lambda, treatment)
        :param y_pred: n x 1 tensor: n, beta*X
        :return: predictive loss statistics:
                the log-rank test statistics is taken as:
                E1/2: number of expected cases in group 1/2
                O1/2: number of observed cases in group 1/2
                
                (E1-O1)**2 / E1 + (E2-O2)**2 / E2
                
                the predictive loss then becomes:
                
                [(E1-O1)**2 / E1 + (E2-O2)**2/E2]_{for biomarker_postive} + 
                [(E1-O1)**2 / E1 + (E2-O2)**2/E2]_{for_biomarker_negative}
                
                Indicators are approximated by sigmoids
                
                Check out the formula on the logrank loss in the supplements
    """
    #add checks on target data! 
    
    ind_approx_1 = 1.0/(1.0+tf.exp(-y_pred[:,0] / omega))    
    ind_approx_2 = 1.0/(1.0+tf.exp(y_pred[:,0] / omega))
    
    treatment_A = tf.cast(y_true[:,3] == 1, tf.float32)
    treatment_B = tf.cast(y_true[:,3] == 0, tf.float32)

    #t-logrank for Treatment arm A 
    g1_num_A = tf.reduce_sum(ind_approx_1*(y_true[:, 1] - y_true[:, 2]) * treatment_A)
    g1_denom_A = tf.reduce_sum(ind_approx_1*y_true[:, 2] * treatment_A)

    g2_num_A = tf.reduce_sum(ind_approx_2*(y_true[:, 1] - y_true[:, 2]) * treatment_A)
    g2_denom_A = tf.reduce_sum(ind_approx_2 * y_true[:, 2] * treatment_A)
    
    
    #t-logrank for Treatment arm B
    g1_num_B = tf.reduce_sum(ind_approx_1*(y_true[:, 1] - y_true[:, 2]) * treatment_B)
    g1_denom_B = tf.reduce_sum(ind_approx_1*y_true[:, 2] * treatment_B)

    g2_num_B = tf.reduce_sum(ind_approx_2*(y_true[:, 1] - y_true[:, 2]) * treatment_B)
    g2_denom_B = tf.reduce_sum(ind_approx_2 * y_true[:, 2] * treatment_B)
    
    
    #Original design
    test_statistics_1 = (g1_num_A*g1_num_A / (g1_denom_A+1.0)) + (g1_num_B*g1_num_B / (g1_denom_B+1.0)) # group 1, trt A vs B
    test_statistics_2 = (g2_num_A*g2_num_A / (g2_denom_A+1.0)) + (g2_num_B*g2_num_B / (g2_denom_B+1.0)) #group 2, trt A vs B
    
    #Updated modification
    test_statistics_3 = (g1_num_A*g1_num_A / (g1_denom_A+1.0)) + (g2_num_A*g2_num_A / (g2_denom_A+1.0))  #Trt A group 1 vs 2
    test_statistics_4 = (g1_num_B*g1_num_B / (g1_denom_B+1.0)) + (g2_num_B*g2_num_B / (g2_denom_B+1.0))  #Trt B group 1 vs 2
    
    
    loss=tf.math.log(test_statistics_4)-tf.math.log(test_statistics_3) #3rd (latest) modification
    # loss = (-1) * (test_statistics_1+test_statistics_2) # 2rd generation design
    # loss2=tf.math.log(test_statistics_2)-tf.math.log(test_statistics_1)
  
    
    return loss




