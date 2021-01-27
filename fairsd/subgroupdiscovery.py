import numpy as np
from heapq import heappush, heappop
import pandas as pd
from abc import ABC, abstractmethod

import fairsd.discretization as discr
import fairlearn.metrics as flm
import inspect

quality_function_options = [
    'equalized_odds_difference',
    'equalized_odds_ratio',
    'demographic_parity_difference',
    'demographic_parity_ratio'
]

quality_function_parameters = [
    'y_true',
    'y_pred',
    'sensitive_features'
]

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
            if self.attribute_name == other.attribute_name and self.is_numeric== other.is_numeric:
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

    def __repr__(self):
        """Represent the description as a string.

        :return : String
        """
        descr=""
        for s in self.selectors:
            if s.is_numeric:
                descr = descr + s.attribute_name + " = '(" + str(s.low_bound) +", "+ str(s.up_bound)+"]' AND "
            else:
                descr = descr+ s.attribute_name+" = '"+str(s.attribute_value)+"' AND "
        if descr != "":
            descr = descr[:-4]
        return descr

    def __lt__(self, other):
        """Compare the current description (self) with another description (other).

        :param other: Description
        :return: bool
        """
        if self.quality != other.quality:
            return self.quality < other.quality
        elif self.support != other.support:
            return self.support < other.support
        else:
            return len(self.selectors) > len(other.selectors)

    def to_boolean_array(self, dataset, set_attributes=False):
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

        if set_attributes:
            #set size, relative size and target share
            self.support=sum(s)
        return s

    def size(self, dataset): # evaluate if delete this method ############
        """ Return the support of the description and set the support in case this parameters was not set.

        :param dataset: pandas.DataFrame
        :return: int
        """
        if self.support is None:
            self.to_boolean_array(dataset, set_attributes=True)
        return self.support

    def get_attributes(self):
        """Return the list of the attribute names in the description.

        :return: list of String
        """
        attributes = []
        for sel in self.selectors:
            attributes.append(sel.get_attribute_name())
        return attributes

    def get_selectors(self):
        return self.selectors

    def is_present_in(self, beam):
        """
        :param beam : array of Description

        :return: bool
            True if the current description (self) is present in the list.
            The current object (Description) has to have the same parameters of another object present in the list.
            It is not needed that the current object must be physically in the list (same cells of memory).
        """
        for descr in beam:
            equals=True
            for sel in self.selectors:
                if sel.is_present_in(descr.selectors) == False:
                    equals = False
                    break
            if equals:
                return True
        return False

    def set_quality(self, q):
        self.quality=q

    def get_quality(self):
        return self.quality


class SearchSpace:
    """Will contain the set of all the descriptors to take into consideration for creating a Description.

    Attributes
    ----------
    nominal_selectors : list of Selector(s)
        will contain all possible non numeric selectors.
    numeric_selectors : list of Selector(s)
        will contain all possible  numeric selectors.
    """

    def __init__(self, dataset, ignore=None, nominal_features=None, numeric_features=None, dynamic_discretization=False, discretizer=None):
        """
        :param dataset : pandas.DataFrame
        :param ignore : list of String(s)
            list the attributes to not take into consideration for creating the search space.
        :param nominal_features : list that contain a subgroup of nominal features
        :param numeric_features : list that contain a subgroup of numeric features
        :param dynamic_discretization: boolean
        :param discretizer: Discretizer object

        Notes
        -----
        The type of features not present in numeric_features and in nominal_features will be deduced based on the attributes

        In this implementation all the numeric selectors created and inserted into the search space
        will be to discretize
        """
        self.dynamic_discretization=dynamic_discretization #####NOT USED
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
        if dynamic_discretization:
            for col in numeric_features:
                if col not in ignore:
                    self.numeric_selectors.append(Selector(col, to_discretize=True, is_numeric=True))
            dtypes_subs = dataset.select_dtypes(include=['number'])
            for col in dtypes_subs.columns:
                if col not in ignore + nominal_features + numeric_features:
                    self.numeric_selectors.append(Selector(col, to_discretize=True, is_numeric=True))
        else:
            for col in numeric_features:
                if col not in ignore:
                    self.numeric_selectors.extend(discretizer.discretize(dataset, Description(), col))
            dtypes_subs = dataset.select_dtypes(include=['number'])
            for col in dtypes_subs.columns:
                if col not in ignore + nominal_features + numeric_features:
                    self.numeric_selectors.extend(discretizer.discretize(dataset, Description(), col))

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
        if current_description is None:
            current_description = Description()
        to_exclude = current_description.get_attributes()
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
        list of Selector(s)
            Will be created e returned (in a list) one Selector for each bin created in the discretization phase.
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
            qf='equalized_odds_ratio', # str (########################### MUST BE CALLABLE ALSO)
            discretizer='mdlp', # str or Discretizer object
            dynamic_discretization=False,
            result_set_size=10,
            depth=3,
            min_quality=0.1,
            min_support=200
        ):

        if isinstance(X, np.ndarray):
            if feature_names is None:
                raise RuntimeError('Since X is a Numpy array, the feature_names parameter must contain the column names of the features')
            self.data = pd.DataFrame(X, columns=feature_names)
            print(self.data)
        else:
            self.data = X.copy()

        self.data['y_true'] = y_true
        if y_pred is not None:
            self.data['y_pred'] = y_pred

        self.discretizer = Discretizer(discretization_type=discretizer, target='y_true')
        self.search_space = SearchSpace(self.data, ['y_true', 'y_pred'], nominal_features, numeric_features,
                                        dynamic_discretization, self.discretizer)


        self.target= BinaryTarget('y_true', 'y_pred', target_value=1) ######### currently target_value is not used
        self.qf=self.set_qualityfuntion(qf)

        self.result_set_size = result_set_size
        self.depth = depth
        self.min_quality = min_quality
        self.min_support = min_support

    def set_qualityfuntion(self, qf):
        if isinstance(qf, str):
            if qf not in quality_function_options:
                raise ValueError('Quality function not known')
            else:
                return getattr(flm, qf)

        if not callable(qf):
            RuntimeError('Supplied metric object must be callable or string')
        sig = inspect.signature(qf).parameters
        for par in quality_function_parameters:
            if par not in sig:
                raise ValueError("Please use the funtions in the fairlearn.metrics package as quality functions")
        return qf


class ResultSet:
    def __init__(self, descriptions_list, x_size):
        self.descriptions_list=descriptions_list
        self.X_size = x_size

    def to_dataframe(self):
        lod=list()
        for d in self.descriptions_list:
            row = [d.quality, d.__repr__(), d.support, d.support/self.X_size]
            lod.append(row)
        columns = ['quality', 'description', 'size', 'relative_size']
        index = [("sg"+str(x)) for x in range(len(self.descriptions_list))]
        return pd.DataFrame(lod, index=index, columns=columns)

    def extract_sg_feature(self,sg_number, data):
        if(sg_number>=len(self.descriptions_list) or sg_number<0):
            raise RuntimeError("The requested subgroup doesn't exists")
        return pd.Series(self.descriptions_list[sg_number].to_boolean_array(data), name=str(sg_number))


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
        list_of_beam[0] =[Description()]

        depth = 0
        while depth < task.depth:
            list_of_beam.append(list())
            #print(depth)
            for last_sg in list_of_beam[depth]:
                ss = task.search_space.extract_search_space(task.data, task.discretizer, current_description=last_sg)
                for sel in ss:

                    new_selectors = list(last_sg.selectors)
                    new_selectors.append(sel)
                    new_description=Description(new_selectors)

                    # check for duplicates
                    if new_description.is_present_in(list_of_beam[depth + 1]):
                        continue

                    sg_belonging_feature = new_description.to_boolean_array(task.data, set_attributes=True)
                    #check min support
                    if new_description.size(task.data)<task.min_support:
                        continue
                    #evaluate subgroup
                    quality=task.qf(y_true = task.data['y_true'], y_pred = task.data['y_pred'],
                                    sensitive_features =sg_belonging_feature )
                    if quality < task.min_quality:
                        continue
                    new_description.set_quality(quality)

                    if len(list_of_beam[depth+1]) < self.beam_width:
                        heappush(list_of_beam[depth+1], new_description)
                    elif quality > list_of_beam[depth+1][0].quality:
                        heappop(list_of_beam[depth+1])
                        heappush(list_of_beam[depth+1], new_description)
            depth +=1

        subgroups=list()
        for l in list_of_beam[1:]:
            subgroups.extend(l)
        subgroups.sort(reverse=True)
        return ResultSet(subgroups[:task.result_set_size], task.data.shape[0])


class DSSD:
    def __init__(self, beam_width=20, a=0.9):
        self.beam_width=beam_width
        self.a= 1-a


    def execute(self, task):
        if self.beam_width < task.result_set_size:
            raise RuntimeError('Beam width in the beam search algorithm is smaller than the result set size!')

        # PHASE 1 - MODIFIED BEAM SEARCH
        list_of_beam = list()
        self.redundancy_aware_beam_search(list_of_beam, task)

        # PHASE 2 - DOMINANCE PRUNING
        subgroups = list()
        subgroups.extend(list_of_beam[1])
        if len(list_of_beam)>1:
            for l in list_of_beam[2:]:
                self.dominance_pruning(l, subgroups, task)

        # PHASE 3 - SUBGROUP SELECTION
        tuples_sg_matrix = []
        quality_array = []
        support_array = list()
        for descr in subgroups:
            tuples_sg_matrix.append(descr.to_boolean_array(task.data))
            quality_array.append(descr.get_quality())
            support_array.append(descr.support)
        support_array = np.array(support_array)
        quality_array = np.array(quality_array)
        tuples_sg_matrix = np.array(tuples_sg_matrix)
        final_sgs = []
        self.beam_creation(tuples_sg_matrix, support_array, quality_array, subgroups, final_sgs, task.result_set_size)

        final_sgs.sort(reverse=True)
        return ResultSet(final_sgs, task.data.shape[0])


    def redundancy_aware_beam_search(self, list_of_beam, task):
        list_of_beam.append(list())
        list_of_beam[0] = [Description()]

        depth = 0
        while depth < task.depth:
            # Generation of the beam with number of descriptors = depth+1

            list_of_beam.append(list())
            # print(depth)

            tuples_sg_matrix = []  # boolean matrix where rows are candidates subgroups and columns are tuples of the dataset
            # tuples_sg_matrix[i][j] == true iff subgroup i contain tuple j
            quality_array = []  # contains the quality of each candidate subgroup
            support_array = []  # contains the support of each candidate subgroup
            decriptions_list = list()  # contains the description object of each candidate subgroup

            # generation of candidates subgroups
            for last_sg in list_of_beam[depth]:  # for each subgroup in the previous beam
                ss = task.search_space.extract_search_space(task.data, task.discretizer, current_description=last_sg)

                # generation of all the possible extensions of the description last_sg
                for sel in ss:

                    new_selectors = list(last_sg.selectors)
                    new_selectors.append(sel)
                    new_description = Description(new_selectors)

                    # check for duplicates
                    if new_description.is_present_in(decriptions_list):
                        continue

                    sg_belonging_feature = new_description.to_boolean_array(task.data, set_attributes=True)
                    support = new_description.support
                    # check min support
                    if support < task.min_support:
                        continue
                    # evaluate subgroup
                    quality = task.qf(y_true=task.data['y_true'], y_pred=task.data['y_pred'],
                                      sensitive_features=sg_belonging_feature)
                    if quality < task.min_quality:
                        continue

                    tuples_sg_matrix.append(sg_belonging_feature)
                    support_array.append(support)
                    quality_array.append(quality)
                    decriptions_list.append(new_description)

            # CREATION OF THE BEAM
            support_array = np.array(support_array)
            quality_array = np.array(quality_array)
            tuples_sg_matrix = np.array(tuples_sg_matrix)
            self.beam_creation(tuples_sg_matrix, support_array, quality_array, decriptions_list,
                               list_of_beam[depth + 1], self.beam_width)

            depth += 1

    def dominance_pruning(self, subgroups, pruned_sgs, task):

        for desc in subgroups:
            selectors = desc.get_selectors()

            generalizable = False
            for i in range(len(selectors)):

                #creation of a generalized description by excluding the i-th descriptor
                new_sel_list = []
                for j in range(len(selectors)):
                    if i != j:
                        new_sel_list.append(selectors[j])
                new_des = Description(new_sel_list)

                sg_belonging_feature = new_des.to_boolean_array(task.data, set_attributes=True)
                quality = task.qf(y_true=task.data['y_true'], y_pred=task.data['y_pred'],
                                  sensitive_features=sg_belonging_feature)

                if quality >= desc.get_quality():
                    generalizable=True
                    if new_des.is_present_in(pruned_sgs):
                        continue
                    new_des.set_quality(quality)
                    pruned_sgs.append(new_des)

            if generalizable == False:
                pruned_sgs.append(desc)

    def beam_creation(self, tuples_sg_matrix, support_array, quality_array, decriptions_list, beam, beam_width):
        # sort in a way that, in case of equal quality, groups with highter support are preferred
        sorted_index = np.argsort(support_array)
        support_array = support_array[sorted_index]
        quality_array = quality_array[sorted_index]
        tuples_sg_matrix = tuples_sg_matrix[sorted_index]
        decriptions_list.sort(key=lambda x: x.support, reverse=True)

        # selection of the sg with highest quality
        index_of_max = np.argmax(quality_array)
        descr = decriptions_list[index_of_max]
        descr.set_quality(quality_array[index_of_max])
        beam.append(descr)
        quality_array[index_of_max] = 0

        a_tothe_c_array = np.ones(tuples_sg_matrix.shape[1])
        for i in range(1, beam_width):
            # a_tothe_c updating
            best_sg_arr = tuples_sg_matrix[index_of_max]
            best_sg_arr = 1 - self.a * best_sg_arr
            #  updating
            a_tothe_c_array = np.multiply(a_tothe_c_array, best_sg_arr)

            # weight creation
            alpha_matrix = np.multiply(a_tothe_c_array, tuples_sg_matrix)
            weights = np.divide(np.sum(alpha_matrix, axis=1), support_array)

            # selection of the sg with highest quality
            weighted_quality_array = np.multiply(quality_array, weights)
            index_of_max = np.argmax(weighted_quality_array)
            descr = decriptions_list[index_of_max]
            descr.set_quality(quality_array[index_of_max])
            # isertion of the description in the beam
            beam.append(descr)
            quality_array[index_of_max] = 0




