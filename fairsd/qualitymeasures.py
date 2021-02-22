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


class EqualOpportunityDiff(QualityFunction):
    def evaluate(self, y_true, y_pred, sensitive_features):
        s0y_true = (y_true & ~sensitive_features).sum()
        s0y_true = 1 if s0y_true == 0 else s0y_true
        p_s0 = (y_true & y_pred & ~sensitive_features).sum() / s0y_true
        s1y_true = (y_true & sensitive_features).sum()
        s1y_true = 1 if s1y_true == 0 else s1y_true
        p_s1 = (y_true & y_pred & sensitive_features).sum() / s1y_true
        return p_s0-p_s1