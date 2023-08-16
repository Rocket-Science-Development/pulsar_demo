#  Author:  Jananjoy Rajkumar <jrajkumar@rocketscience.one>
import numpy as np
from sklearn.metrics import confusion_matrix
""" Implementation of ROI calculation for wrong or bad prediction """
#Hard Coded value for testing
FPR_THRESHOLD = 0.5
FPN_THRESHOLD = 0.2 

class ReturnOfInvestment():
    def __init__(self):
        #Hard coded for testing
        self.cost_per_FP = 100
        self.cost_per_FN = 300
        '''Constructor of the ReturnOfInvestment class '''  

    def calculate_ROI(self,lebel_data,predicted_data):
        cm = confusion_matrix(lebel_data, predicted_data)
        cm_param = {'tn': cm[0, 0], 'fp': cm[0, 1],'fn': cm[1, 0], 'tp': cm[1, 1]}
        print("confusion_matrix",cm_param)
        
        #Caluclate the True Positive Rate(TPR) 
        TPR_denominator = cm_param['tp'] + cm_param['fn'] 
        TPR =  cm_param['tp']/TPR_denominator
        print(" True Positive Rate(TPR)",TPR)

        #Caluclate the True Negative Rate(TNR) 
        TNR_denominator = cm_param['tn'] + cm_param['fn']
        TNR =cm_param['tn']/TNR_denominator
        print("True Negative Rate(TNR)",TNR)

        #Caluclate the Return of Investment (ROI) for wrong prediction
        totalNegatives = cm_param['tn'] + cm_param['fn']  #TN + FN
        totalPositives = cm_param['tp'] + cm_param['fp']  #TP + FP
        net_FPR = np.sign(FPR_THRESHOLD - TPR) * totalNegatives*self.cost_per_FP
        net_FNR = np.sign(FPN_THRESHOLD - TNR) * totalPositives*self.cost_per_FN

        print("predict_ROI_FP =>",net_FPR )
        print("predict_ROI_FN =>",net_FNR )
        
        ROI_gain = net_FPR + net_FNR
        
        return ROI_gain    
