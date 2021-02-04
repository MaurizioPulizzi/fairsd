from abc import ABC, abstractmethod

class QualityFunction(ABC):
    """Abstract class.

    If the user wants to create a customized quality function, it is recommended to extend this class
    """
    @abstractmethod
    def evaluate(self, y_true, y_pred, sensitive_features):
        """Evaluate the quality of a description.

        Parameters
        ----------
        y_true : list, pandas.series or numpy array
        y_pred : list, pandas.series or numpy array
        sensitive_features : list, pandas.series or numpy array

        Returns
        -------
        double
            Real number indicating rhe calculated quality.
        """
        pass