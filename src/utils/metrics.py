import torchmetrics

class Metrics:
    def score(yhat,y):
        """Accuracy Score

        Args:
            yhat (float): probabilities
            y (integer(0/1)): class label 

        Returns:
            float: range 0-1
        """
        accuracy = torchmetrics.Accuracy()
        return accuracy(yhat,y)
    def f1(yhat,y):
        """F1-Score

        Args:
            yhat (float): probabilities
            y (integer(1/0)): class label

        Returns:
            float: range 0-1
        """
        f1 = torchmetrics.F1Score()
        return f1(yhat,y)
    