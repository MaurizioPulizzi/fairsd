import fairsd.discretization as discr
from .sgdescription import Description
from .sgdescription import Descriptor


class SearchSpace:
    """Will contain the set of all the descriptors to take into consideration for creating a Description.

    Attributes
    ----------
    nominal_Descriptors : list of Descriptor(s)
        will contain all possible non numeric Descriptors.
    numeric_Descriptors : list of Descriptor(s)
        will contain all possible  numeric Descriptors.
    """

    def __init__(self, dataset, ignore=None, nominal_features=None, numeric_features=None, dynamic_discretization=False, discretizer=None, sensitiive_features=None):
        """
        :param dataset : pandas.DataFrame
        :param ignore : list of String(s)
            list the attributes to not take into consideration for creating the search space.
        :param nominal_features : list of Strings that contain a subgroup of nominal features
        :param numeric_features : list of Strings that contain a subgroup of numeric features
        :param dynamic_discretization: boolean
            if dinamic_discretization is true the numerical features will be discretizized in the extract_search_space()
            method, otherwise the numerical features will be discretizized here, during the inizialization.
        :param discretizer: Discretizer object
        :param sensitiive_features: List of str

        Notes
        -----
        The type of features not present in numeric_features and in nominal_features will be deduced based on the attributes

        """
        self.sensitiive_features = sensitiive_features
        if ignore is None:
            ignore = []
        if nominal_features is None:
            nominal_features = []
        if numeric_features is None:
            numeric_features = []
        self.nominal_Descriptors = []
        self.numeric_Descriptors = []
        # create nominal Descriptors
        #nominal features with explicit type phassed
        for col in nominal_features:
            if col not in ignore:
                values = dataset[col].unique()
                for x in values:
                    self.nominal_Descriptors.append(Descriptor(col, attribute_value=x))
        #nominal features without explicit type phassed
        dtypes_subs = dataset.select_dtypes(exclude=['number'])
        for col in dtypes_subs.columns:
            if col not in ignore + nominal_features + numeric_features:
                values = dataset[col].unique()
                for x in values:
                    self.nominal_Descriptors.append(Descriptor(col, attribute_value=x))

        # numerical Descriptors
        if dynamic_discretization:
            for col in numeric_features:
                if col not in ignore:
                    self.numeric_Descriptors.append(Descriptor(col, to_discretize=True, is_numeric=True))
            dtypes_subs = dataset.select_dtypes(include=['number'])
            for col in dtypes_subs.columns:
                if col not in ignore + nominal_features + numeric_features:
                    self.numeric_Descriptors.append(Descriptor(col, to_discretize=True, is_numeric=True))
        else:
            for col in numeric_features:
                if col not in ignore:
                    self.numeric_Descriptors.extend(discretizer.discretize(dataset, Description(), col))
            dtypes_subs = dataset.select_dtypes(include=['number'])
            for col in dtypes_subs.columns:
                if col not in ignore + nominal_features + numeric_features:
                    self.numeric_Descriptors.extend(discretizer.discretize(dataset, Description(), col))

    def extract_search_space(self, dataset, discretizer, current_description=None):
        """This method return the subset of the search space to explore
         for expanding the description "current_description".

        All descriptors containing attributes present in the current_description will be removed
        from the returned search space subset.

        Parameters
        ----------
        dataset : pandas.Dataframe
        discretizer : Discretizer
        current_description : Description

        Rreturns
        --------
        list of Descriptors
            the subset of the search space to explore

        Notes
        -----
        "Descriptors"  list will contain the subset of the search space to return. All the Descriptors not yet
        discretized in the original search space, will be discretized before being inserted in the "Descriptors" list.
        """
        if current_description is None:
            current_description = Description()
        to_exclude = current_description.get_attributes()
        if len(to_exclude) == 0:
            to_exclude = [i for i in list(dataset.columns) if i not in self.sensitiive_features]
        Descriptors = []
        for Descriptor in self.nominal_Descriptors:
            if Descriptor.attribute_name not in to_exclude:
                Descriptors.append(Descriptor)
        for Descriptor in self.numeric_Descriptors:
            if Descriptor.attribute_name not in to_exclude:
                if Descriptor.is_to_discretize():
                    Descriptors.extend(discretizer.discretize(dataset, current_description, Descriptor.get_attribute_name()))
                else :
                    Descriptors.append(Descriptor)
        return Descriptors


class Discretizer:
    """Class for the discretization of the numeric attributes."""

    def __init__(self, discretization_type, target=None, min_groupsize=1, num_bins = 6):
        """

        :param discretization_type : enumerated
            can be "mdlp" or "equalfreq" or "equalwidth"
        :param target: String, optional
            this parameter is needed only for supervised discretizations (mdlp)
        :param min_groupsize: int, optional
            discretize() method will create only subgroups with size >= min_groupsize.
        """
        self.discretization_type = discretization_type
        if discretization_type =='mdlp':
            self.supervised = True
            self.discretizer =discr.MDLP(min_groupsize, force=True)
            self.target = target
        elif discretization_type =='equalfreq':
            self.supervised = False
            self.discretizer = discr.EqualFrequency(min_groupsize, num_bins)
        elif discretization_type =='equalwidth':
            self.supervised = False
            self.discretizer = discr.EqualWidth(min_groupsize, num_bins)
        else:
            raise RuntimeError('discretization_type must be "mdlp" OR "equalfreq" OR "equalwidth"')

    def discretize(self, data, description, feature): #### to test
        """
        Parameters
        ----------
        data: pandas.DataFrame
            The dataset.
        description: Description
            The discretization will be based only on those tuples of the dataset that match the description.
        feature: String
            Is the name of the numeric attribute of the dataset to discretize.

        Returns
        -------
        list of Descriptor(s)
            Will be created and returned (in a list) one Descriptor for each bin created in the discretization phase.
        """
        subset= data[description.to_boolean_array(data)]
        x = subset[feature]
        if self.supervised:
            y = subset[self.target]
            cut_points=self.discretizer.findCutPoints(x, y)
        else:
            cut_points = self.discretizer.findCutPoints(x)

        Descriptors = []
        if len(cut_points) < 1:
            return Descriptors

        Descriptors.append(Descriptor(feature, low_bound=None, up_bound=None, is_numeric=True))
        for cp in cut_points:
            Descriptors[-1].up_bound=cp
            Descriptors.append(Descriptor(feature, low_bound=cp, up_bound=None, is_numeric=True))
        return Descriptors
