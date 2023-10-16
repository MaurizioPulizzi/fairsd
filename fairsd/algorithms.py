import numpy as np
import pandas as pd
import fairlearn.metrics as flm
import inspect
import logging
from .sgdescription import Description
from .searchspace import SearchSpace
from .searchspace import Discretizer
"""
The class SubgroupDiscoveryTask is an adaptation of the homonymous class of the pysubgroup library.
"""

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


class SubgroupDiscoveryTask:
    """This is an interface class and will contain all the parameters useful for the sg discovery algorithms.
    """
    def __init__(self,
            X, # pandas dataframe or numpy array with features
            y_true, # numpy array, pandas dataframe, or pandas Series with ground truth labels
            y_pred = None, # numpy array, pandas dataframe, or pandas Series with classifier's predicted labels
            feature_names=None, # optional, list with column names in case users supply a numpy array X
            sensitive_features = None, #list of sensitive features names (str)
            nominal_features = None, #optional, list of nominal features
            numeric_features = None, #optional, list of nominal features
            qf='equalized_odds_difference', # str or callable object
            discretizer='equalfreq', # str
            num_bins = 6,
            dynamic_discretization=True, #boolean
            result_set_size=5, # int
            depth=3, # int
            min_quality=0, # float
            min_support=200, #int
            min_support_ratio=0.1, #float
            max_support_ratio=1.0, #float
            logging_level=logging.INFO
        ):
        """
        Parameters
        ----------
        X : pandas dataframe or numpy array
        y_true : numpy array, pandas dataframe, or pandas Series
            represent the ground truth
        y_pred : numpy array, pandas dataframe, or pandas Series
            contain the predicted values
        feature_names : list of string
            this parameter is necessary if the user supply X in a numpy array
        sensitive_features: list of string
            this list contains the names of the sensitive features
        nominal_features : optional, list of strings
            list of nominal features
        numeric_features : optional, list of strings
            list of nominal features
        qf : string or callable object
        discretizer : string
            can be "mdlp", "equalfrequency" or "equalwidth"
        num_bins : int
            maximum number of bins that a numerical feature discretization operation will produce
        dynamic_discretization : boolean
        result_set_size : int
        depth : int
            maximum number of descriptors in a description
        min_quality : float
        min_support : int
            minimum size of a subgroup
        min_support_ratio : float
            minimum proportion of a subgroup compared to the whole dataset size
        max_support_ratio : float
            maximum proportion of a subgroup compared to the whole dataset size
        logging_level : int
            logging level
        """
        logging.basicConfig(level=logging_level)
        self.inputChecking(X, y_true, y_pred, feature_names, sensitive_features, nominal_features, numeric_features,
                           discretizer, dynamic_discretization, result_set_size, depth, min_quality, min_support, min_support_ratio)
        if isinstance(X, np.ndarray):
            self.data = pd.DataFrame(X, columns=feature_names)
        else:
            self.data = X.copy()

        self.data['y_true'] = y_true
        if y_pred is not None:
            self.data['y_pred'] = y_pred
            self.there_is_y_pred =True
        else:
            self.there_is_y_pred = False
        self.sensitive_features=sensitive_features
        self.discretizer = Discretizer(discretization_type=discretizer, target='y_true', num_bins=num_bins)
        self.search_space = SearchSpace(self.data, ['y_true', 'y_pred'], nominal_features, numeric_features,
                                        dynamic_discretization, self.discretizer, sensitive_features)


        self.qf=self.set_qualityfuntion(qf)

        self.result_set_size = result_set_size
        self.depth = depth
        self.min_quality = min_quality
        self.min_support = min_support
        self.min_support_ratio = min_support_ratio
        self.max_support_ratio = max_support_ratio


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
                raise ValueError("Please use the functions in the fairlearn.metrics package as quality functions or "
                                 "other fuctions with the same interface")
        return qf

    def inputChecking(self,
              X,  # pandas dataframe or numpy array
              y_true,  # numpy array, pandas dataframe, or pandas Series with ground truth labels
              y_pred,  # numpy array, pandas dataframe, or pandas Series with classifier's predicted labels
              feature_names,  # optional, list with column names in case users supply a numpy array X
              sensitive_features,
              nominal_features,  # optional, list of nominal features
              numeric_features,  # optional, list of nominal features
              discretizer,  # str
              dynamic_discretization,  # boolean
              result_set_size,  # int
              depth,  # int
              min_quality,  # float
              min_support,  # int
              min_support_ratio # float
        ):
        if not (isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray)):
            raise TypeError("X must be of type numpy.ndarray or pandas.DataFrame")
        if not (isinstance(y_true, pd.DataFrame) or isinstance(y_true, np.ndarray) or isinstance(y_true, pd.Series)):
            raise TypeError("y_true must be of type numpy.ndarray, pandas.Series or pandas.DataFrame")
        if X.shape[0] != y_true.size:
            raise RuntimeError("X and y_true have two different dimensions")
        if y_pred is not None:
            if not (isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, np.ndarray) or isinstance(y_pred, pd.Series)):
                raise TypeError("y_pred must be of type numpy.ndarray, pandas.Series or pandas.DataFrame")
            if y_pred.size != y_true.size:
                raise RuntimeError("y_pred and y_true have two different dimensions")
        if isinstance(X, np.ndarray):
            if (not isinstance(feature_names, list)) or len(feature_names) != X.shape[1]:
                raise RuntimeError("If X is a numpy.ndarray, feature_names must contain the names of the colums")
        if sensitive_features is not None and not isinstance(sensitive_features, list):
            raise RuntimeError("sensitive_features input must be of list type or None")
        if nominal_features is not None and not isinstance(nominal_features, list):
            raise RuntimeError("nominal_features input must be of list type or None")
        if numeric_features is not None and not isinstance(nominal_features, list):
            raise RuntimeError("numeric_features input must be of list type or None")
        if not isinstance(discretizer, str):
            raise TypeError("discretizer input must be of string type")
        if discretizer == "mdlp":
            t = pd.DataFrame(y_true).iloc[:, 0].unique()
            if not (t ==[1,0]).all() and not (t ==[0,1]).all():
                raise RuntimeError("MDLP discretization supports only binary target")
        if not isinstance(dynamic_discretization, bool):
            raise TypeError("dynamic_discretization input must be of bool type")
        if not isinstance(dynamic_discretization, bool):
            raise TypeError("dynamic_discretization input must be of bool type")
        if not isinstance(result_set_size, int) or result_set_size<1:
            raise RuntimeError("result_set_size input must be greater than 0")
        if not isinstance(depth, int):
            raise RuntimeError("depth input must be greater than 0")
        if not isinstance(min_support, int):
            raise RuntimeError("min_support input must be greater than 0")
        if not isinstance(min_support_ratio, float):
            raise RuntimeError("min_support_ratio input must be of float type")
        if min_quality>1 or min_quality<0:
            raise RuntimeError("min_quality input must be between 0 and 1")


class ResultSet:
    """
    This class is used to represent the subgroup set found by one of the
    subgroup discovery algorithms implemented in this package.
    """
    def __init__(self, descriptions_list, x_size):
        """
        :param descriptions_list: list of Description objects
        :param x_size: int, size of the dataset
        """
        self.descriptions_list=descriptions_list
        self.X_size = x_size

    def to_dataframe(self):
        """ This method convert the result set into a dataframe

        :return: pandas.Dataframe
        """
        lod=list()
        for d in self.descriptions_list:
            row = [d.quality, d.__repr__(), d.support, d.support/self.X_size]
            lod.append(row)
        columns = ['quality', 'description', 'size', 'proportion']
        index = [str(x) for x in range(len(self.descriptions_list))]
        return pd.DataFrame(lod, index=index, columns=columns)

    def get_description(self, sg_index):
        if (sg_index >= len(self.descriptions_list) or sg_index < 0):
            raise RuntimeError("The requested subgroup doesn't exists")
        return self.descriptions_list[sg_index]

    def sg_feature(self,sg_index, X):
        """
        This method generate and return the feature of the subgroup with index = sg_index in the current object.
        The result is indeed a boolean array of the same length of the dataset X. Each i-th element of this
        array is true iff the i-th tuple of X belong to the subgroup with index sg_index.

        :param sg_index: int, number of the subgroup in the current object
        :param X: pandas DataFrame or numpy array
        :return: boolean list
        """
        if(sg_index>=len(self.descriptions_list) or sg_index<0):
            raise RuntimeError("The requested subgroup doesn't exists")
        return pd.Series(self.descriptions_list[sg_index].to_boolean_array(X), name=str("sg"+str(sg_index)))

    def __repr__(self):
        res=""
        for desc in self.descriptions_list:
            res+= desc.__repr__() + "\n"
        return res

    def print(self):
        print(self.__repr__())

    def to_string(self):
        return self.__repr__()


class BeamSearch:
    """This class is used to execute the Beam Search Algorithm."""
    def __init__(self, beam_width=20):
        """
        :param beam_width : int
        """
        if beam_width<1:
            raise RuntimeError("beam_width must be greater than 0")
        self.beam_width = beam_width

    def execute(self, task):
        """
        This method execute the Beam Search

        :param task : SubgroupDiscoveryTask
        :return: ResultSet object

        Notes
        -----
        The list_of_beam variable is: a list of list of descriptions. The i-th element of list_of_beam, at the end,
        will contain the most interesting descriptions formed by i descriptors.
        """
        if self.beam_width < task.result_set_size:
            raise RuntimeError('Beam width is smaller than the result set size!')

        list_of_beam=list()
        list_of_beam.append(list())
        list_of_beam[0] =[Description()]

        depth = 0
        while depth < task.depth:
            list_of_beam.append(list())
            #print(depth)
            current_min_quality = 1
            for last_sg in list_of_beam[depth]:
                ss = task.search_space.extract_search_space(task.data, task.discretizer, current_description=last_sg)
                for sel in ss:
                    new_Descriptors = list(last_sg.Descriptors)
                    new_Descriptors.append(sel)
                    new_description=Description(new_Descriptors)

                    # check for duplicates
                    if new_description.is_present_in(list_of_beam[depth + 1]):
                        continue

                    sg_belonging_feature = new_description.to_boolean_array(task.data, set_attributes=True)
                    # check min support
                    if new_description.size(task.data)<task.min_support \
                        or new_description.size(task.data)<task.min_support_ratio*task.data.shape[0] \
                            or new_description.size(task.data)>task.max_support_ratio*task.data.shape[0]:
                        continue
                    # evaluate subgroup
                    if task.there_is_y_pred:
                        quality=task.qf(y_true = task.data['y_true'], y_pred = task.data['y_pred'],
                                        sensitive_features =sg_belonging_feature)
                    else:
                        quality = task.qf(y_true=task.data['y_true'], sensitive_features=sg_belonging_feature)
                    if quality < task.min_quality:
                        continue
                    new_description.set_quality(quality)

                    if len(list_of_beam[depth + 1]) < self.beam_width:
                        list_of_beam[depth + 1].append(new_description)
                        if current_min_quality > quality:
                            current_min_quality = quality
                    elif quality > current_min_quality:
                        i=0
                        while list_of_beam[depth + 1][i].quality != current_min_quality:
                            i = i + 1
                        list_of_beam[depth + 1][i] = new_description
                        current_min_quality = 1
                        for d in list_of_beam[depth + 1]:
                            if d.quality < current_min_quality:
                                current_min_quality = d.quality
            depth +=1

        subgroups=list()
        for l in list_of_beam[1:]:
            subgroups.extend(l)
        subgroups.sort(reverse=True)
        return ResultSet(subgroups[:task.result_set_size], task.data.shape[0])


class DSSD:
    """
    This class implements the Diverse Subgroup Set Discovery algorithm (DSSD).
    This algorithm is a variant of the Beam Search Algorithm that also take into account
    the redundancy of the generated subgroups.
    In this implementation a cover-based redundancy definition is used: roughly, the more tuples two subgroups
    have in common, the more they are considered redundant.
    This algorithm is described in details in the Van Leeuwen and Knobbe's paper "Diverse Subgroup Set Discovery".
    """
    def __init__(self, beam_width=20, a=0.9):
        """
        :param beam_width: int
        :param a: float
            this parameter correspond to the alpha parameter.
            the more a is high, the less the subgroups redundancy is taken into account.
        """
        if beam_width<1:
            raise RuntimeError("beam_width must be greater than 0")
        if a<0 or a>1:
            raise RuntimeError("a-parameter must be between 0 and 1")
        self.beam_width=beam_width
        self.a= 1-a # for future calculations it is more practical to memorize 1-a

    def execute(self, task):
        """
        :param task: SubgroupDiscoveryTask object
        :return: ResultSet object

        Notes
        -----
        The algorithm is divided in three phases:
        Phase 1: a modified beam search algorithm is performed to find a first non-redundant subset
        Phase 2: Dominance Pruning - the algorithm try to generalize the subgroups finded in the previous phase
        Phase 3: The subgroups to return are chosen among the sg-set resulting from the previous phase. Again, the
            subgroups to put in the result set are chosen by taking into account both quality and diversity.
        """
        if self.beam_width < task.result_set_size:
            raise RuntimeError('Beam width is smaller than the result set size!')

        # PHASE 1 - MODIFIED BEAM SEARCH
        list_of_beam = list()
        self.redundancy_aware_beam_search(list_of_beam, task)

        # PHASE 2 - DOMINANCE PRUNING
        subgroups = list()
        subgroups.extend(list_of_beam[1])
        if len(list_of_beam)>2:
            for l in list_of_beam[2:]:
                self.dominance_pruning(l, subgroups, task)
        if len(subgroups)<task.result_set_size:
            subgroups.sort(reverse=True)
            return ResultSet(subgroups, task.data.shape[0])

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
        """
        Parameters
        ----------
        list_of_beam : list
            This list is empty at the beginning and this method will fill it with the beams of each level.
            The beam of level i will be a list of Description objects where each description is composed of
            i Descriptors.
        task : SubgroupDiscoveryTask

        Notes
        -----
        Starting from the beam of the previous level, all the candidates subgroups (descriptions) are generated.
        After this, the beam of the current level is generated by calling the beam_creation method.
        """
        list_of_beam.append(list())
        list_of_beam[0] = [Description()]

        depth = 0
        while depth < task.depth:
            # Generation of the beam with number of descriptors = depth+1

            list_of_beam.append(list())
            logging.debug("DEPTH: "+str(depth+1))

            tuples_sg_matrix = [] # boolean matrix where rows are candidates subgroups and columns are tuples of the dataset
                                  # tuples_sg_matrix[i][j] == true iff subgroup i contain tuple j
            quality_array = []  # will contain the quality of each candidate subgroup
            support_array = []  # will contain the support of each candidate subgroup
            decriptions_list = list()  # will contain the description object of each candidate subgroup

            # generation of candidates subgroups
            for last_sg in list_of_beam[depth]:  # for each subgroup in the previous beam
                ss = task.search_space.extract_search_space(task.data, task.discretizer, current_description=last_sg)

                # generation of all the possible extensions of the description last_sg
                for sel in ss:
                    new_Descriptors = list(last_sg.Descriptors)
                    new_Descriptors.append(sel)
                    new_description = Description(new_Descriptors)

                    # check for duplicates
                    if new_description.is_present_in(decriptions_list):
                        continue

                    sg_belonging_feature = new_description.to_boolean_array(task.data, set_attributes=True)
                    support = new_description.support
                    # check min support
                    if support < task.min_support \
                        or support < task.min_support_ratio * task.data.shape[0] \
                            or support > task.max_support_ratio * task.data.shape[0]:
                        continue
                    # comparison with new descriptor alone
                    sel_feature = Description([sel]).to_boolean_array(task.data)
                    # evaluate subgroup
                    if task.there_is_y_pred:
                        quality = task.qf(y_true=task.data['y_true'], y_pred=task.data['y_pred'],
                                          sensitive_features=sg_belonging_feature)
                        ### to evaluate
                        sel_quality = task.qf(y_true=task.data['y_true'], y_pred=task.data['y_pred'],
                                          sensitive_features=sel_feature)
                    else:
                        quality = task.qf(y_true=task.data['y_true'], sensitive_features=sg_belonging_feature)
                        ### to evaluate
                        sel_quality = task.qf(y_true=task.data['y_true'], sensitive_features=sel_feature)

                    # if the quality of the new descriptor has deteriorated by merging the new descriptor
                    # with the current description, we do not add the new descriptor
                    ### to evaluate
                    if quality < task.min_quality or quality < sel_quality:
                        continue
                    '''
                    # This is for allowing descriptions with negative quality in the first beam
                    if depth>0 and (quality < task.min_quality or quality < sel_quality):
                        continue
                    '''

                    # This code is for apriori discard those descriptions dominated by another descriptions not containing sensitive features
                    pruned_des = []
                    new_description.set_quality(quality)
                    self.dominance_pruning([new_description], pruned_des, task)
                    pruned_des.sort(reverse=True)
                    pruned_attr = pruned_des[0].get_attributes()
                    if task.sensitive_features is not None:
                        any_in = any(i in task.sensitive_features for i in pruned_attr)
                        if not any_in:
                            continue


                    # insert current subgroup in the candidates
                    tuples_sg_matrix.append(sg_belonging_feature)
                    support_array.append(support)
                    quality_array.append(quality)
                    decriptions_list.append(new_description)

            if len(decriptions_list) == 0:
                break
            # CREATION OF THE BEAM
            support_array = np.array(support_array)
            quality_array = np.array(quality_array)
            tuples_sg_matrix = np.array(tuples_sg_matrix)
            self.beam_creation(tuples_sg_matrix, support_array, quality_array, decriptions_list,
                               list_of_beam[depth + 1], self.beam_width)


            for d in list_of_beam[depth + 1]:
                logging.debug(d)
            logging.debug(" ")
            depth += 1

    def dominance_pruning(self, subgroups, pruned_sgs, task):
        """
        Parameters
        ----------
        subgroups : list
                list of Description objects to try to generalize
        pruned_sgs : list
                the generalized subgroups (description objects) are inserted in this list
        task : SubgroupDiscoveryTask
        """

        for desc in subgroups:
            Descriptors = desc.get_Descriptors()

            generalizable = False
            for i in range(len(Descriptors)):

                #creation of a generalized description by excluding the i-th descriptor
                new_sel_list = []
                for j in range(len(Descriptors)):
                    if i != j:
                        new_sel_list.append(Descriptors[j])
                new_des = Description(new_sel_list)

                sg_belonging_feature = new_des.to_boolean_array(task.data, set_attributes=True)
                if task.there_is_y_pred:
                    quality = task.qf(y_true=task.data['y_true'], y_pred=task.data['y_pred'],
                                      sensitive_features=sg_belonging_feature)
                else:
                    quality = task.qf(y_true=task.data['y_true'], sensitive_features=sg_belonging_feature)

                if quality >= desc.get_quality():
                    generalizable=True
                    if new_des.is_present_in(pruned_sgs):
                        continue
                    new_des.set_quality(quality)
                    pruned_sgs.append(new_des)

            if generalizable == False:
                pruned_sgs.append(desc)

    def beam_creation(self, tuples_sg_matrix, support_array, quality_array, decriptions_list, beam, beam_width):
        """
        Parameters
        ----------
        tuples_sg_matrix : numpy array
            this is a boolean matrix. The rows are candidate subgroups and the columns are the istances of the dataset
        support_array :  numpy array
            contains the support of each candidate subgroup
        quality_array :  numpy array
            contains the quality of each candidate subgroup
        decriptions_list : list
            list of Description objects of the candidates subgroups
        beam : list
            list of Description objects. This parameter is empty at the beginning and this method will feel it with the
            selected subgroups
        beam_width : int

        Notes
        -----
        In the code there is a variable called a_tothe_c_array. This variable represents a vector of weights,
        each weight refers to a tuple of the dataset. Every time that a subgroup sg is selected (inserted to the beam),
        all the weights relatives to the tuples belonging to sg are updated. In particular, they are decreased by
        multiplying them by a.
        At each round of the for loop, the subgroup with the highest product between its quality and its weight is
        selected. The weight of a subgroup is obtained by averaging the weights of all the tuples it contains.
        """

        if len(decriptions_list) <= beam_width:
            for i in range(len(decriptions_list)):
                descr = decriptions_list[i]
                descr.set_quality(quality_array[i])
                # isertion of the description in the beam
                beam.append(descr)
            return

        # sort in a way that, in case of equal quality, groups with highter support are preferred
        sorted_index = np.argsort(support_array)[::-1]
        support_array = support_array[sorted_index]
        quality_array = quality_array[sorted_index]
        tuples_sg_matrix = tuples_sg_matrix[sorted_index]
        decriptions_list.sort(key=lambda x: x.support, reverse=True)

        # selection of the sg with highest quality
        index_of_max = np.argmax(quality_array)
        descr = decriptions_list[index_of_max]
        descr.set_quality(quality_array[index_of_max])
        beam.append(descr)
        quality_array[index_of_max] = 0 # the quality of the selected sg is set to 0 in the quality_array,
                                        # in this way this subgroup will never be choosen again

        a_tothe_c_array = np.ones(tuples_sg_matrix.shape[1])

        num_iterations = min(beam_width, support_array.size)
        for i in range(1, num_iterations):
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
