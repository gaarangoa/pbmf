import tensorflow as tf
from tensorflow.keras import Model
import pandas as pd
import numpy as np
import lifelines
from lifelines import KaplanMeierFitter

# Lambdas are pre-computed constants for the loss function. 
# Check derivation notes for where to lambdas come from
def lambdas_(time, event, **kwargs):
    """
    lambda_i= Sum_t Omega_t / N_t * I(T_i>t)
    y is survival time with shape (n, )
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations=time, event_observed=event ) #fit km to your data

    #extract event table
    e_table=pd.DataFrame(kmf.event_table).reset_index() # reset index so that time is part of the data. 
    omegat_over_Nt=e_table["observed"]/e_table["at_risk"] # Omege_t/N_t term in the equation. 
    omega_sum = np.sum(omegat_over_Nt)
    lamdas=np.zeros(time.shape[0]) #where to save lambda values 
    for ix, Ti in enumerate(time):
        lamdas[ix]=np.sum(omegat_over_Nt * (Ti>=e_table["event_at"]))
    return lamdas   

# Predictive Loss Function
class PredLoss:
    __name__ = 'PredLoss'
    def __init__(self, omega=0.05, minp=0.5, w1=1., w2=1., w3=1., w4=1.0, **kwargs):
        self.omega = omega
        self.minp = minp
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.return_components = kwargs.get('return_components', False)
        self.Ebeta = kwargs.get('e_beta', 1)

    def __call__(self, y_true, y_pred):
        return PredictiveBiomarkerLoss(y_true, y_pred, self.omega, self.minp, w1=self.w1, w2=self.w2, w3=self.w3, w4=self.w4, return_components=self.return_components)

def PredictiveBiomarkerLoss(y_true, y_pred, omega=0.5, minp=0.5, w1=1.0, w2=1.0, w3=1.0, w4=1.0, return_components=False, Ebeta=1):
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
                
                Indicators are approximated by sigmoid
                
                Check out the formula on the logrank loss in the supplements

        biomarker_loss(y_true, y_pred, return_components=False)

        Computes the loss function for a binary classification problem with a biomarker.

        Args:
            y_true (tf.float32): The true labels.
            y_pred (tf.float32): The predicted probabilities.
            return_components (bool, optional): If True, returns the individual component losses in addition to the final loss value. Defaults to False.

        Returns:
            tf.float32: The final loss value.
            Optional[list]: The individual component losses, if `return_components` is True.

        Description:
            This function computes the loss function for a binary classification problem with a biomarker. It combines several different components:

            1. Proportion loss term that penalizes the proportion of negative biomarker results.
            2. Margin ranking loss term that ensures that the biomarker positive results are always ranked higher than the biomarker negative results.
            3. Treatment ratio loss term that compares the number of biomarker positive results in the treated group to the number of biomarker positive results in the control group.
            4. Composite loss function that combines the above terms.

            The `return_components` parameter is a boolean that determines whether to return only the final loss value or also include the individual component losses.
    """
    #add checks on target data! 
    
    ind_approx_1 = y_pred[:,0] # B+
    ind_approx_2 = y_pred[:,1] # B-
    
    treatment_A = tf.cast(y_true[:,3] == 1, tf.float32) # Target treatment
    treatment_B = tf.cast(y_true[:,3] == 0, tf.float32) # Control treatment

    O=y_true[:, 1]
    E=y_true[:, 2] / Ebeta

    #t-logrank for Treatment arm A (Treatment)
    g1_num_A = tf.reduce_sum(ind_approx_1 * (O - E) * treatment_A)
    g1_denom_A = tf.reduce_sum(ind_approx_1 * E * treatment_A)

    g2_num_A = tf.reduce_sum(ind_approx_2 * (O - E) * treatment_A)
    g2_denom_A = tf.reduce_sum(ind_approx_2 * E * treatment_A)
    
    #t-logrank for Treatment arm B (Control)
    g1_num_B = tf.reduce_sum(ind_approx_1 * (O - E) * treatment_B)
    g1_denom_B = tf.reduce_sum(ind_approx_1 * E * treatment_B)

    g2_num_B = tf.reduce_sum(ind_approx_2 * ( O - E) * treatment_B)
    g2_denom_B = tf.reduce_sum(ind_approx_2 * E * treatment_B)
    
    #Original design
    test_statistics_1 = (g1_num_A*g1_num_A / (g1_denom_A+1.0)) + (g1_num_B*g1_num_B / (g1_denom_B+1.0)) # group 1, trt A vs B
    test_statistics_2 = (g2_num_A*g2_num_A / (g2_denom_A+1.0)) + (g2_num_B*g2_num_B / (g2_denom_B+1.0)) #group 2, trt A vs B
    
    #Updated modification
    test_statistics_3 = (g1_num_A*g1_num_A / (g1_denom_A+1.0)) + (g2_num_A*g2_num_A / (g2_denom_A+1.0))  #Trt A group 1 vs 2 TrtA==1
    test_statistics_4 = (g1_num_B*g1_num_B / (g1_denom_B+1.0)) + (g2_num_B*g2_num_B / (g2_denom_B+1.0))  #Trt B group 1 vs 2 TrtB==0 (Control)
    
    # loss=tf.math.log(test_statistics_4)-tf.math.log(test_statistics_3) #3rd (latest) modification
    
    # Proportion loss: We want to make sure that the biomarker has at least x.x fraciton of the population
    g1_counts = tf.reduce_sum(ind_approx_1)
    g2_counts = tf.reduce_sum(ind_approx_2)
    total = g1_counts + g2_counts
    if minp > 0.0:
        pr = (g1_counts / total) # proportion of biomarker positive
        Pr = tf.math.square((pr/minp) - 1) # proportion of biomarker negative
    else:
        Pr = 0.0
        pr = 0.0    
    # Lets use a sort-of-gaussian kernel to penalize the proportion.
    # pr2 = 1-tf.math.exp(-0.5*tf.pow(pr - 0.5, 2))
    # maxPr2 = tf.reduce_max(pr2)
    # Pr2 = pr2 / maxPr2

    # Pr = tf.pow(minp-pr, 5) + tf.pow(minp-pr, 4)
    # Pr3 = tf.maximum(0.0, 1 - pr/minp)

    # Pr = tf.math.exp(-2*pr/minp)

    # Margin Ranking Loss 
    # we will make sure that the biomarker positive is always the ind_aprox_1
    # marging = 0.0 
    # dr_A = tf.maximum(0.0, (g2_num_A - g1_num_A + marging)/(g2_num_A + g1_num_A + marging))
    # dr_B = tf.maximum(0.0, (g1_num_B - g2_num_B + marging)/(g1_num_B + g2_num_B + marging))
    
    # dr_A = tf.maximum(0.0, (g1_num_A / g1_denom_A) - tf.math.abs(g2_num_A / g2_denom_A) + marging) # Penalizes positive values   
    dr_A = tf.math.exp(-2*(g1_num_A - g2_num_A) / (g1_num_A + g2_num_A))
    dr_B = tf.math.exp(-2*(-g1_num_B + g2_num_B) / (g1_num_B + g2_num_B))

    # dr_B = tf.maximum(0.0, tf.math.square(g1_num_B) - tf.math.square(g2_num_B) + marging)
    
    
    # main loss - treatment ratio
    tratio =  test_statistics_4 / test_statistics_3
    # tratio =  test_statistics_2 / test_statistics_1
    #tratio =  (test_statistics_4 + test_statistics_3)/ test_statistics_3


    # composite loss
    # loss = w1*(tratio) + w2*Pr + w3*dr_A  + w4*dr_B
    loss = w1*tratio + w2*Pr

    if return_components == False:
        return loss
    else: 
        return loss, tratio, dr_A, dr_B, Pr, pr, g1_counts, g2_counts, g1_num_A, g1_num_B, g2_num_A, g2_num_B, test_statistics_3, test_statistics_4, g1_denom_A, g1_denom_B, g2_denom_A, g2_denom_B


def biomarker_index_old(y_true, y_pred):
    ind_approx_1 = y_pred[:,0] # B+
    ind_approx_2 = y_pred[:,1] # B-
    
    treatment_A = tf.cast(y_true[:,3] == 1, tf.float32) # Target treatment
    
    #t-logrank for Treatment arm A (Treatment)
    g1_num_A = tf.reduce_sum(ind_approx_1*(y_true[:, 1] - y_true[:, 2]) * treatment_A)
    g2_num_A = tf.reduce_sum(ind_approx_2*(y_true[:, 1] - y_true[:, 2]) * treatment_A)
    

    if g1_num_A < g2_num_A:
        return 0
    else:
        return 1


def biomarker_index(y_true, y_pred):
    B_plus = y_pred[:,0] # B+
    B_minus = y_pred[:,1] # B-
    
    treatment_A = tf.cast(y_true[:,3] == 1, tf.float32) # Target treatment
    
    O=y_true[:, 1]
    E=y_true[:, 2]
    
    #t-logrank for Treatment arm A (Treatment)
    O_plus = tf.reduce_sum(B_plus * O * treatment_A)
    E_plus = tf.reduce_sum(B_plus * E * treatment_A)
    
    O_minus = tf.reduce_sum(B_minus * O * treatment_A)
    E_minus = tf.reduce_sum(B_minus * E * treatment_A)
    

    HR = (O_plus / E_plus) / (O_minus / E_minus)
    
    if HR < 1:
        return 0
    else:
        return 1

def logrank_hazard_ratio(y_true, y_pred, treatment=1):
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