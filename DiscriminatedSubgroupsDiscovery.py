import numpy as np
from heapq import heappush, heappop
#from mdlp.discretization import MDLP
import Orange
from Orange.data.pandas_compat import table_from_frame
import pandas as pd
from abc import ABC, abstractmethod


class QualityFunction(ABC):
    @abstractmethod
    def evaluate(self, description, task):
        '''
        :param description: class Description - is the description of the subgroup to evaluate
        :param task: class SubgroupDiscoveryTask
        :return:quality of the subgroup - double type
        '''
        pass


class BinaryTarget:

    def __init__(self, target_attribute, dataset, predicted_target_attr=None, target_value=False):

        #check target attribute
        if target_attribute not in dataset.columns:
            raise RuntimeError('target_attribute is not present in the dataset')
        #check target attribute
        if target_value not in dataset[target_attribute].unique():
            raise RuntimeError('No tuple in the dataset has the selected target value')

        # check predicted target attribute
        if predicted_target_attr is not None and predicted_target_attr not in dataset.columns:
            raise RuntimeError('predicted_target_attr is not present in the dataset')

        self.target_attribute = target_attribute
        self.predicted_target_attr = predicted_target_attr
        self.target_value = target_value


class StandardQF:
    def __init__(self, a):
        self.a = a

    def evaluate(self, description, task):
        N = task.dataset_size
        p = task.dataset_positive_target
        data = task.data
        target_attr = task.target.target_attribute
        target_val = task.target.target_value
        subset = data[description.to_boolean_array(data)]
        N_sg = subset.shape[0]
        p_sg = subset[(subset[target_attr] == target_val)].shape[0]

        N_sg_correct = 1 if N_sg == 0 else N_sg

        return ((N_sg / N) ** self.a) * ((p_sg / N_sg_correct) - (p / N)), N_sg


class Selector:
    def __init__(self, attribute_name, attribute_value=None, up_bound=None, low_bound=None, to_discretize=False, is_numeric=False):
        self.attribute_name = attribute_name
        self.is_numeric = is_numeric ##########################forse non serve
        if is_numeric:
            self.up_bound = up_bound
            self.low_bound = low_bound
        else:
            self.attribute_value = attribute_value

        self.to_discretize = to_discretize ######################################### da eliminare

    def get_attribute_name(self):
        return self.attribute_name
    def is_to_discretize(self):
        return self.to_discretize
    def is_present_in(self, other_descriptors):
        for other in other_descriptors:
            if self.attribute_name == other.attribute_name:
                if self.is_numeric and self.up_bound==other.up_bound and self.low_bound==other.low_bound:
                    return True
                elif self.is_numeric == False and self.attribute_value==other.attribute_value:
                    return True
        return False


class Description:
    def __init__(self, selectors=None):
        if selectors == None:
            self.selectors=[]
        else:
            self.selectors=selectors
        self.support = None

    def set_support(self, support):
        self.support = support

    def __repr__(self):
        descr=""
        for s in self.selectors:
            if s.is_numeric:
                descr = descr + s.attribute_name + " = '(" + str(s.low_bound) +", "+ str(s.up_bound)+"]' AND "
            else:
                descr = descr+ s.attribute_name+" = '"+s.attribute_value+"' AND "
        if descr != "":
            descr = descr[:-4]
        return descr

    def __lt__(self, other): ################## da rivedere
        if self.support != other.support:
            return self.support < other.support
        else:
            return len(self.selectors) > len(other.selectors)

    def to_boolean_array(self, dataset):
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


    def size(self, dataset): ##############
        if self.support is None:
            self.support = dataset[self.to_boolean_array(dataset)].shape[0]
        return self.support
    def get_attributes(self):
        attributes = []
        for sel in self.selectors:
            attributes.append(sel.get_attribute_name())
        return attributes

    def is_present_in(self, beam):
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
    def __init__(self, dataset, ignore=None):
        if ignore is None:
            ignore = []
        self.nominal_selectors = []
        self.numeric_selectors = []
        # create nominal selectors
        dtypes_subs = dataset.select_dtypes(exclude=['number'])
        for col in dtypes_subs.columns:
            if col not in ignore:
                values = dataset[col].unique()
                for x in values:
                    self.nominal_selectors.append(Selector(col, attribute_value=x))
        # numerical selectors
        dtypes_subs = dataset.select_dtypes(include=['number'])
        for col in dtypes_subs.columns:
            if col not in ignore:
                self.numeric_selectors.append(Selector(col, to_discretize=True, is_numeric=True))

    def extract_search_space(self, dataset, discretizer, current_description=None):
        '''
        to_exclude: list attributes (string)
        :param dataset:
        :return:
        '''
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
    def __init__(self, discretization_type, target):
        if discretization_type !='mdlp':
            raise RuntimeError('discretization_type mus be "mdlp" OR...')
        self.discretization_type = discretization_type
        self.target = target

    def discretize(self, data, description, feature): ############# to test
        subset= data[description.to_boolean_array(data)]
        y = subset[[self.target]]
        x = subset[[feature]].copy()
        '''
        transformer = MDLP()
        transformer.fit(x, y)  # discretization
        cut_points = transformer.cut_points_[0]
        '''
        data_df = pd.concat([x, y], axis=1)
        orange_table = table_from_frame(data_df, class_name='y_true')

        disc = Orange.preprocess.Discretize()
        disc.method = Orange.preprocess.discretize.EntropyMDL(force=True)
        cut_points = disc(orange_table)

        selectors = []
        if len(cut_points) < 1:
            return selectors

        selectors.append(Selector(feature, is_numeric=True))
        for cp in cut_points:
            selectors[-1].up_bound=cp
            selectors.append(Selector(feature, low_bound=cp, is_numeric=True))
        return selectors


class SubgroupDiscoveryTask:
    '''
    Capsulates all parameters required to perform standard subgroup discovery
    '''

    def __init__(self, data, target, search_space, qf, discretizer, result_set_size=10, depth=3, min_quality=0, min_support=0):
        self.data = data
        self.target = target
        self.search_space = search_space
        self.discretizer = discretizer
        self.qf = qf
        self.result_set_size = result_set_size
        self.depth = depth
        self.min_quality = min_quality
        self.min_support = min_support
        self.dataset_size=data.shape[0] #to remove
        self.dataset_positive_target= data[(data[target.target_attribute]==target.target_value)].shape[0] # to remove


class BeamSearch:
    def __init__(self, beam_width=20):
        self.beam_width = beam_width

    def execute(self, task):
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
                    #controllo duplicati
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
