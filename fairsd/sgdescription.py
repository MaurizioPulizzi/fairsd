import numpy as np

# The following class is no more used in the current version
#class BinaryTarget:
#    """Contains the target for the subgroup discovery task.
#
#    The target can be boolean or Nominal. Only the value contained in the parameter "target_value" will be considered as
#    the true value. In this way also if the target is nominal, it will still be treated as a boolean.
#    """
#
#    def __init__(self, y_true, y_pred=None, target_value=False):
#        """
#        Parameters
#        ----------
#        y_true : string
#            Contains the label of the target.
#        dataset: pandas.DataFrame
#            The dataset is required because it will be checked if the others parameters are coherent
#            (present inside the dataset).
#        y_pred: String, optional
#            Contains the label of the predicted attribute.
#        target_value: bool or String, optional
#       """
#        self.y_true = y_true
#        self.y_pred = y_pred
#        self.target_value = target_value


class Descriptor:
    """
    Thi object is formed by an attribute name and an attribute value (or a lower bound plus an upper bound
    if the Descriptor is numeric).
    """

    def __init__(self, attribute_name, attribute_value=None, up_bound=None, low_bound=None, to_discretize=False, is_numeric=False):
        """
        Parameters
        ----------
        attribute_name : string
        attribute_value : string or bool, default None
            To set only if the Descriptor is not numeric.
        up_bound : double or int, default None
             To set iff the Descriptor is numeric and already discretized.
         low_bound : double or int, default None
             To set iff the Descriptor is numeric and already discretized.
       to_discretize : bool, default False
            To set at True iff the Descriptor is numeric and not still discretized. In this case
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
    """List of Descriptors plus other description attributes.

    Semantically it is to be interpreted as the conjunction of all the Descriptors contained in the list:
    a dataset record will match the description if each single Descriptor of the description will match with this record.
    """
    def __init__(self, Descriptors=None):
        """
        :param Descriptors : list of Descriptor
        """
        if Descriptors == None:
            self.Descriptors=[]
        else:
            self.Descriptors=Descriptors
        self.support = None

    def __repr__(self):
        """Represent the description as a string.

        :return : String
        """
        descr=""
        for s in self.Descriptors:
            if s.is_numeric:
                low = str(s.low_bound) if s.low_bound is not None else "-infinite"
                up = str(s.up_bound) if s.up_bound is not None else "+infinite"
                descr = descr + s.attribute_name + " = '(" + low +", "+ up +"]' AND "
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
            return len(self.Descriptors) > len(other.Descriptors)

    def to_boolean_array(self, dataset, set_attributes=False):
        """
        Parameters
        ----------
        dataset : pandas.DataFrame
        set_attributes : bol, default False
            if this input is True, this method will set also the support attribute

        Returns
        -------
        array of boolean:
            The array will have the length of the passed  dataset (number of rows).
            Each element of the array will be true iff the description (self) match the corresponding row of the dataset.
            If a description is empty, the returned array will have all elements equal to True.
        """
        s = np.full(dataset.shape[0], True)
        for i in range(0, len(self.Descriptors)):
            if self.Descriptors[i].is_numeric:
                if self.Descriptors[i].low_bound is not None:
                    s = s & (dataset[self.Descriptors[i].attribute_name] > self.Descriptors[i].low_bound)
                if self.Descriptors[i].up_bound is not None:
                    s = s & (dataset[self.Descriptors[i].attribute_name] <= self.Descriptors[i].up_bound)
            else:
                s =( (s) & (dataset[self.Descriptors[i].attribute_name] == self.Descriptors[i].attribute_value))

        if set_attributes:
            #set size, relative size and target share
            self.support=sum(s)
        return s

    def size(self, dataset):
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
        for sel in self.Descriptors:
            attributes.append(sel.get_attribute_name())
        return attributes

    def get_Descriptors(self):
        return self.Descriptors

    def is_present_in(self, beam):
        """
        :param beam : list of Description objects

        :return: bool
            True if the current description (self) is present in the list (beam).
            Return true iff the current object (Description) have the same parameters of at list another object
            present in the beam.

        """
        for descr in beam:
            equals=True
            for sel in self.Descriptors:
                if sel.is_present_in(descr.Descriptors) == False:
                    equals = False
                    break
            if equals:
                return True
        return False

    def set_quality(self, q):
        self.quality=q

    def get_quality(self):
        return self.quality
