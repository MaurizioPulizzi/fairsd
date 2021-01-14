import numpy as np
from heapq import heappush, heappop

from Orange.data.pandas_compat import table_from_frame
import pandas as pd
from abc import ABC, abstractmethod

import fairsd.discretization as discr

class QualityFunction(ABC):
    """Abstract class.

    It is suggested to extend this class to create new quality functions for the evaluation of the subgroups.
    """
    @abstractmethod
    def evaluate(self, description, task):
        """Evaluate the quality of a description.

        Parameters
        ----------
        description : Description
            Is the description of the subgroup to evaluate.
        task : SubgroupDiscoveryTask
            Contains all the parameters that can be useful to calculate the quality,
            like the dataset and the target.

        Returns
        -------
        double
            Real number indicating rhe calculated quality.
        """
        pass


class EqualOpportunity(QualityFunction):
    def evaluate(self, description, task):
        data = task.data
        target_attr = task.target.y_true
        target_val = task.target.target_value
        pred_attr = task.target.y_pred

        p_subset = data[description.to_boolean_array(data) & (data[target_attr] == target_val)]
        p_sg = p_subset.shape[0]
        tp_sg = p_subset[(p_subset[pred_attr] == target_val)].shape[0]

        p_complement = data[description.complement_to_boolean_array(data) & (data[target_attr] == target_val)]
        p_c = p_complement.shape[0]
        tp_c = p_complement[(p_complement[pred_attr] == target_val)].shape[0]

        if p_sg == 0: return 0
        if p_c == 0: return 0
        return (tp_c / p_c) - (tp_sg/p_sg)


class BinaryTarget:
    """Contains the target for the subgroup discovery task.

    The target can be boolean or Nominal. Only the value contained in the parameter "target_value" will be considered as
    the true value. In this way also if the target is nominal, it will still be treated as a boolean.
    """

    def __init__(self, y_true, y_pred=None, target_value=False):
        """
        Parameters
        ----------
        y_true : string
            Contains the label of the target.
        dataset: pandas.DataFrame
            The dataset is required because it will be checked if the others parameters are coherent
            (present inside the dataset).
        y_pred: String, optional
            Contains the label of the predicted attribute.
        target_value: bool or String, optional

        """

        self.y_true = y_true
        self.y_pred = y_pred
        self.target_value = target_value


class Selector:
    """Selector, also know as Descriptor in the literature.

    Thi object is formed by an attribute name and an attribute value (or a lower bound plus an upper bound
    if the selector is numeric).
    """

    def __init__(self, attribute_name, attribute_value=None, up_bound=None, low_bound=None, to_discretize=False, is_numeric=False):
        """
        Parameters
        ----------
        attribute_name : string
        attribute_value : string or bool, default None
            To set only if the selector is not numeric.
        up_bound : double or int, default None
             To set iff the selector is numeric and already discretized.
         low_bound : double or int, default None
             To set iff the selector is numeric and already discretized.
       to_discretize : bool, default False
            To set at True iff the selector is numeric and not still discretized. In this case
            the up_bound and low_bound attributes will be meaningless
        is_numeric : bool, default False
            To set at true iff the descriptor is numeric
        """

        self.attribute_name = attribute_name
        self.is_numeric = is_numeric
        if is_numeric:
            self.up_bound = up_bound
            self.low_bound = low_bound
        else:
            self.attribute_value = attribute_value
        self.to_discretize = to_discretize # The current implementation could work also without this parameter

    def get_attribute_name(self):
        return self.attribute_name
    def is_to_discretize(self):
        return self.to_discretize
    def is_present_in(self, other_descriptors):
        """
        :param: other_descriptors: list of Descriptors
        :return: bool
        """
        for other in other_descriptors:
            if self.attribute_name == other.attribute_name:
                if self.is_numeric and self.up_bound==other.up_bound and self.low_bound==other.low_bound:
                    return True
                elif self.is_numeric == False and self.attribute_value==other.attribute_value:
                    return True
        return False


class Description:
    """List of Descriptors.

    Semantically it is to be interpreted as the conjunction of all the Selectors contained in the list:
    a dataset record will match the description if each single Selector of the description will match with this record.
    """
    def __init__(self, selectors=None):
        """
        :param selectors : list of Selector
        """
        if selectors == None:
            self.selectors=[]
        else:
            self.selectors=selectors
        self.support = None

    def set_support(self, support):
        """
        :param support : int
            this parameter is needed to compare two different description (see __lt__() function)
        :return: void
        """
        self.support = support

    def __repr__(self):
        """Represent the description as a string.

        :return : String
        """
        descr=""
        for s in self.selectors:
            if s.is_numeric:
                descr = descr + s.attribute_name + " = '(" + str(s.low_bound) +", "+ str(s.up_bound)+"]' AND "
            else:
                descr = descr+ s.attribute_name+" = '"+s.attribute_value+"' AND "
        if descr != "":
            descr = descr[:-4]
        return descr

    def __lt__(self, other):
        """Compare the current description (self) with another description (other).

        To compare two descriptions, one compares their quality first (according to the quality function), but
        in the current implementation, quality is not a parameter of the description and the comparison happens "outside".
        Currently this method will therefore only be called if two descriptions have equal quality.

        :param other: Description
        :return: bool
        """
        if self.support != other.support:
            return self.support < other.support
        else:
            return len(self.selectors) > len(other.selectors)

    def to_boolean_array(self, dataset):
        """
        Parameters
        ----------
        dataset : pandas.DataFrame

        Returns
        -------
        array of boolean:
            The array will have the length of the passed  dataset (number of rows).
            Each element of the array will be true iff the description (self) match the corresponding row of the dataset.
            If a description is empty, the returned array will have all elements equal to True.
        """
        s = np.full(dataset.shape[0], True)
        for i in range(0, len(self.selectors)):
            if self.selectors[i].is_numeric:
                if self.selectors[i].low_bound is not None:
                    s = s & (dataset[self.selectors[i].attribute_name] > self.selectors[i].low_bound)
                if self.selectors[i].up_bound is not None:
                    s = s & (dataset[self.selectors[i].attribute_name] <= self.selectors[i].up_bound)
            else:
                s =( (s) & (dataset[self.selectors[i].attribute_name] == self.selectors[i].attribute_value))
        return s

    def complement_to_boolean_array(self, dataset):  #not useful
        return np.invert(self.to_boolean_array(dataset))


    def size(self, dataset): # evaluate if delete this method ############
        """ Return the support of the description and set the support in case this parameters was not set.

        :param dataset: pandas.DataFrame
        :return: int
        """
        if self.support is None:
            self.support = dataset[self.to_boolean_array(dataset)].shape[0]
        return self.support

    def get_attributes(self):
        """Return the list of the attribute names in the description.

        :return: list of String
        """
        attributes = []
        for sel in self.selectors:
            attributes.append(sel.get_attribute_name())
        return attributes

    def is_present_in(self, beam):
        """
        :param beam : array of tuples <_, Description>

        :return: bool
            True if the current description (self) is present in the list.
            The current object (Description) has to have the same parameters of another object present in the list.
            It is not needed that the current object must be physically in the list (same cells of memory).
        """
        for _, descr in beam:
            equals=True
            for sel in self.selectors:
                if sel.is_present_in(descr.selectors) == False:
                    equals = False
                    break
            if equals:
                return True
        return False


class SearchSpace:
    """Will contain the set of all the descriptors to take into consideration for creating a Description.

    Attributes
    ----------
    nominal_selectors : list of Selector(s)
        will contain all possible non numeric selectors.
    numeric_selectors : list of Selector(s)
        will contain all possible  numeric selectors.
    """

    def __init__(self, dataset, ignore=None, nominal_features=None, numeric_features=None):
        """
        :param dataset : pandas.DataFrame
        :param ignore : list of String(s)
            list the attributes to not take into consideration for creating the search space.
        :param nominal_features : list that contain a subgroup of nominal features
        :param numeric_features : list that contain a subgroup of numeric features

        Notes
        -----
        The type of features not present in numeric_features and in nominal_features will be deduced based on the attributes

        In this implementation all the numeric selectors created and inserted into the search space
        will be to discretize
        """
        if ignore is None:
            ignore = []
        if nominal_features is None:
            nominal_features = []
        if numeric_features is None:
            numeric_features = []
        self.nominal_selectors = []
        self.numeric_selectors = []
        # create nominal selectors
        #nominal features with explicit type phassed
        for col in nominal_features:
            if col not in ignore:
                values = dataset[col].unique()
                for x in values:
                    self.nominal_selectors.append(Selector(col, attribute_value=x))
        #nominal features without explicit type phassed
        dtypes_subs = dataset.select_dtypes(exclude=['number'])
        for col in dtypes_subs.columns:
            if col not in ignore + nominal_features + numeric_features:
                values = dataset[col].unique()
                for x in values:
                    self.nominal_selectors.append(Selector(col, attribute_value=x))

        # numerical selectors
        for col in numeric_features:
            if col not in ignore:
                self.numeric_selectors.append(Selector(col, to_discretize=True, is_numeric=True))
        dtypes_subs = dataset.select_dtypes(include=['number'])
        for col in dtypes_subs.columns:
            if col not in ignore + nominal_features + numeric_features:
                self.numeric_selectors.append(Selector(col, to_discretize=True, is_numeric=True))

    def extract_search_space(self, dataset, discretizer, current_description=None):
        """This method return the subset of the search space to explore
         for expanding the description "current_description".

        All descriptors containing attributes present in the current_description will be removed
        from the returned subset.

        Parameters
        ----------
        dataset : pandas.Dataframe
        discretizer : Discretizer
        current_description : Description

        Rreturns
        --------
        list of Selectors
            the subset of the search space to explore

        Notes
        -----
        "selectors" will contain the subset of the search space to return. All the selectors not yet discretized
        in the original search space, will be discretized before being inserted in the "selectors" list
        """
        to_exclude = current_description.get_attributes() if current_description is not None else []
        selectors = []
        for selector in self.nominal_selectors:
            if selector.attribute_name not in to_exclude:
                selectors.append(selector)
        for selector in self.numeric_selectors:
            if selector.attribute_name not in to_exclude:
                if selector.is_to_discretize():
                    selectors.extend(discretizer.discretize(dataset, current_description, selector.get_attribute_name()))
                else :
                    selectors.append(selector)
        return selectors


class Discretizer:
    """Class for the discretization of the numeric attributes."""

    def __init__(self, discretization_type, target=None, min_groupsize=1):
        """

        :param discretization_type : enumerated
            can be only {"mdlp"}
        :param target: String, optional
        :param min_groupsize: int, optional
        """
        if discretization_type !='mdlp':
            raise RuntimeError('discretization_type mus be "mdlp" OR...')
        self.discretization_type = discretization_type
        self.discretizer=discr.MDLP(min_groupsize, force=True)
        self.target = target

    def discretize(self, data, description, feature): #### to test
        """
        Parameters
        ----------
        data: pandas.DataFrame
            Is the dataset.
        description: Description
            The discretization will be based only on those tuples of the dataset that match the description.
        feature: String
            Is the name of the numeric attribute of the dataset to discretize.

        Returns
        -------
        list of Descriptor(s)
            Will be created e returned (in a list) one Descriptor for each bin created in the discretization phase.

        Notes
        -----
        The Orange library in modified.
        In particular the disc() function of the Orange library, in this implementation, return the cut points
        instead of the discretized dataset.
        For this reason this function (discretize()) can work only in the original virtual environment of the project.
        PS: I'm sorry Hilde, I know these things shouldn't be done.
        """
        subset= data[description.to_boolean_array(data)]
        y = subset[self.target]
        x = subset[feature]

        cut_points=self.discretizer.findCutPoints(x, y)

        selectors = []
        if len(cut_points) < 1:
            return selectors

        selectors.append(Selector(feature, is_numeric=True))
        for cp in cut_points:
            selectors[-1].up_bound=cp
            selectors.append(Selector(feature, low_bound=cp, is_numeric=True))
        return selectors


class SubgroupDiscoveryTask:
    def __init__(self,
            X, # pandas dataframe or numpy array with features
            y_true, # numpy array, pandas dataframe, or pandas Series with ground truth labels
            y_pred = None, # numpy array, pandas dataframe, or pandas Series with classifier's predicted labels
            feature_names=None, # optional, list with column names in case users supply a numpy array X
            nominal_features = None, #optional, list of nominal features
            numeric_features = None, #optional, list of nominal features
            qf='equal_opportunity_difference', # str or callable (see my comment `Fairlearn integration')
            discretizer='mdlp', # str or Discretizer object
            result_set_size=10,
            depth=3,
            min_quality=0.1,
            min_support=250
        ):

        if isinstance(X, np.ndarray):
            if feature_names is None:
                raise RuntimeError('Since X is a Numpy array, the feature_names parameter must contain the column names of the features')
            self.data = pd.DataFrame(X, feature_names)
        else:
            self.data = X
        self.search_space = SearchSpace(self.data, None, nominal_features, numeric_features)
        self.data['y_true'] = y_true
        if y_pred is not None:
            self.data['y_pred'] = y_pred

        self.target= BinaryTarget('y_true', 'y_pred', target_value=1) ######### currently target_value is not used

        if qf == 'equal_opportunity_difference':
            self.qf = EqualOpportunity()
        #elif:
        else:
            raise RuntimeError('Quality function not known')

        self.discretizer = Discretizer(discretization_type=discretizer, target='y_true')

        self.result_set_size = result_set_size
        self.depth = depth
        self.min_quality = min_quality
        self.min_support = min_support


class BeamSearch:
    """This class is used to execute the Beam Search Algorithm."""
    def __init__(self, beam_width=20):
        """
        :param beam_width : int
        """
        self.beam_width = beam_width

    def execute(self, task):
        """
        This method execute the Beam Search

        :param task : SubgroupDiscoveryTask
        :return: list of tuples <double, string>
            each tuple in the list will contain: <quality, subgroup description>

        Notes
        -----
        The list_of_beam variable is: a list of list of tuples. The i-th element of list_of_beam, at the end,
        will contain the most interesting descriptions formed by i descriptors, together with their quality.
        """
        if self.beam_width < task.result_set_size:
            raise RuntimeError('Beam width in the beam search algorithm is smaller than the result set size!')

        list_of_beam=list()
        list_of_beam.append(list())
        list_of_beam[0] =[(0,Description())]

        depth = 0
        while depth < task.depth:
            list_of_beam.append(list())
            #print(depth)
            for (_, last_sg) in list_of_beam[depth]:
                ss = task.search_space.extract_search_space(task.data, task.discretizer, current_description=last_sg)
                for sel in ss:

                    new_selectors = list(last_sg.selectors)
                    new_selectors.append(sel)
                    new_description=Description(new_selectors)

                    if new_description.size(task.data)<task.min_support:
                        continue
                    #check for duplicates
                    if new_description.is_present_in(list_of_beam[depth+1]):
                        continue

                    quality=task.qf.evaluate(new_description,task)
                    #new_description.set_support(support)

                    if len(list_of_beam[depth+1]) < self.beam_width:
                        heappush(list_of_beam[depth+1], (quality, new_description))
                    elif quality > list_of_beam[depth+1][0][0]:
                        heappop(list_of_beam[depth+1])
                        heappush(list_of_beam[depth+1], (quality, new_description))
            depth +=1

        subgroups=list()
        for l in list_of_beam:
            for ll in l:
                if len(subgroups) < task.result_set_size:
                    heappush(subgroups, (ll[0],ll[1].__repr__()))
                elif ll[0] > subgroups[0][0]:
                    heappop(subgroups)
                    heappush(subgroups,(ll[0],ll[1].__repr__()))
        return subgroups