#  Author:  Jananjoy Rajkumar <jrajkumar@rocketscience.one>
import numpy as np
from sklearn.metrics import confusion_matrix
""" Implementation of ROI calculation for wrong or bad prediction """

class ReturnOfInvestment():
    def __init__(self):
               '''Constructor of the ReturnOfInvestment class '''  

    def calculate_ROI(
                self,
                lebel_data,
                predicted_data,
                FPR_THRESHOLD = 0.5,
                FNR_THRESHOLD = 0.2,
                cost_per_FP = 100,
                cost_per_FN = 300
                ):
        cm = confusion_matrix(lebel_data, predicted_data)
        cm_param = {'tn': cm[0, 0], 'fp': cm[0, 1],'fn': cm[1, 0], 'tp': cm[1, 1]}
        print("confusion_matrix",cm_param)
        
        # total_positves =  (FN+TP) being the total number of positives
        totalPositives = cm_param['fn'] + cm_param['tp']
        #total_negatives = (FP+ TN)  is the total number of ground truth negatives.
        totalNegatives = cm_param['fp'] + cm_param['tn']
        
        # Caluclate the False Positive Rate(FPR) 
        # The False Positive Rate  FPR = FP / (FP+ TN) 
        FPR_denominator = cm_param['fp'] + cm_param['tn'] 
        FPR = cm_param['fp']/FPR_denominator
        print("False Positive Rate(FPR)",FPR)

        #Caluclate the False Negative Rate(FNR) 
        #The False Negative Rate is FNR =  FN / (FN + TP)
        FNR_denominator = cm_param['fn'] + cm_param['tp']
        FNR = cm_param['fn']/FNR_denominator
        print("False Negative Rate(FNR)",FNR)

        #Caluclate the Return of Investment (ROI) for wrong prediction
        net_FPR = np.sign(FPR_THRESHOLD - FPR) * totalNegatives * cost_per_FP
        net_FNR = np.sign(FNR_THRESHOLD - FNR) * totalPositives * cost_per_FN

        print("predict_ROI_FP =>",net_FPR)
        print("predict_ROI_FN =>",net_FNR)
        
        ROI_gain = net_FPR + net_FNR
        return ROI_gain    