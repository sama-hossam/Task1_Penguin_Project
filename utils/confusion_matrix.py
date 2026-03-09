def confusion_matrix(y_target,y_pred):
    TP=0
    TN=0
    FP=0
    FN=0

    for i in range (len(y_target)):
        if y_target[i]==1 and y_pred[i]==1:
            TP+=1

        elif y_target[i]== -1 and y_pred[i]==-1:
            TN+=1

        elif y_target[i]== -1 and y_pred[i]==1:
            FP+=1

        elif y_target[i]==1 and y_pred[i]==-1:
            FN+=1      
    return TP,TN,FP,FN          