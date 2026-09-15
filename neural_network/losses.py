import numpy as np

class CategoricalCrossEntropy:
    def forward(self,y_pred,y_actual):
        # avoid log(0)
        y_pred_clipped = np.clip(y_pred,1e-12,1-1e-12)
        sample_losses = -np.sum(y_actual*np.log(y_pred_clipped),axis=1)
        return np.mean(sample_losses)

    def backward(self,y_pred,y_actual):
        n_samples = y_pred.shape[0]
        # fused softmax + cross entropy gradient
        return (y_pred-y_actual)/n_samples


def one_hot(y,num_classes=10):
    encoded = np.zeros((y.size,num_classes))
    encoded[np.arange(y.size),y]=1 # fancy indexing
    return encoded