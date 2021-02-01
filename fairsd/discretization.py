import pandas as pd
import math
import numpy as np

class MDLP:
    """
    This class find the cut points to discretize a numerical feature using the Fayyad and Irani approach (MDL principle).

    In thi class the constructor allows to specify the minimum possible size for a group (min_groupSize).
    This parameter is interpret as an hard constraint: only cut points that produce buckets larger than
    min_groupSize will be returned.
    Another parameter that is possible to set is "force". If this parameter is setted to True the findCutPoints
    method will retur at list one cut point. This unless even a cut point can be found which produces two subgroups
    with a size greater than min_groupSize and with class partition entropy less than the entropy of the entire set.

    """
    def __init__(self, min_groupsize = 1, force=False):
        """
        :param min_groupsize: int
        :param force: boolean
        """
        self.min_groupsize = min_groupsize
        self.force = force

    def findCutPoints(self, x, y):
        """
        :param x: list, numpy array or pandas Series that contains the values of the numeric feature to discretize
        :param y: list, numpy array or pandas Series that contains binary values (0 or 1) representing the class label

        :return: list of (ascending) ordered cut points
            if the function returns the cut points [c1, c2, ..., cn], with c1<c2<...<cn, the feature can be discretized
            by creating n+1 buckets: (-infinite, c1], (c1, c2], ..., (cn-1, cn], (cn, +infinite)
        """
        df = pd.DataFrame({'x':x, 'y':y})
        total_size = df.shape[0]

        df=df.groupby('x')['y'].agg(['sum','count'])
        df.reset_index(inplace=True)
        total_sum=df['sum'].sum()
        df['prop'] = df['sum'] / df['count']

        cut_points = self.find_partitions(df, total_size, total_sum, self.force)
        cut_points.sort()
        return cut_points

    def find_partitions(self, df, total_size, total_sum, force=False):
        """
        This is a private class function. It works in a recursive manner.

        Parameters
        ----------
        df: pandas.DataFrame
            this dataframe contains, for each distinct values of the feature to discretize, the number of positive
            instances (called sum), the total number of instances (count) and the proportion
            positive_instances/total_num_of_instances (prop). See the findCutPoints method for details.
        total_size: int
            represent the total number of instances of the x feature in the current partition.
        total_sum: int
            represent the total number of positive instances of the x feature in the current partition.
        force: boolean
            force the method to return at list one cut point, exept for some exceptional cases described in the class description.

        Returns
        -------
        list of int:
            list of cut points
        """

        sum = 0
        count = 0
        min_cpe = total_size  # Class Partition Entropy (CPE)
        partition_index = 0
        partition_x = 0
        partition_sum = 0
        partition_count = 0

        #find best candidate cut point
        for i in range(0, df.shape[0] - 1):
            loc = df.iloc[i]
            sum += loc['sum']
            count += loc['count']
            if loc['prop'] == df.iloc[i + 1]['prop'] or count < self.min_groupsize or (
                    total_size - count) < self.min_groupsize:
                continue

            # cakculate CPE cut point
            pc1s0 = sum / count  # probability of class 1 in subgroup 0
            pc0s0 = 1 - pc1s0
            pc1s0_ = pc1s0 if pc1s0 != 0 else 1
            pc0s0_ = pc0s0 if pc0s0 != 0 else 1
            entS0 = -(pc1s0 * math.log2(pc1s0_) + pc0s0 * math.log2(pc0s0_))

            pc1s1 = (total_sum - sum) / (total_size - count)
            pc0s1 = 1 - pc1s1
            pc1s1_ = pc1s1 if pc1s1 != 0 else 1
            pc0s1_ = pc0s1 if pc0s1 != 0 else 1
            entS1 = -(pc1s1 * math.log2(pc1s1_) + pc0s1 * math.log2(pc0s1_))

            cpe = (count / total_size) * entS0 + ((total_size - count) / total_size) * entS1

            if cpe < min_cpe:
                min_cpe = cpe
                partition_index = i
                partition_x = loc['x']
                partition_sum = sum
                partition_count = count
                partition_entS0 = entS0
                partition_entS1 = entS1
        if min_cpe == total_size:
            return []

        #test MDLP condition
        pc1 = total_sum / total_size
        pc0 = 1 - pc1
        pc1_ = pc1 if pc1 != 0 else 1
        pc0_ = pc0 if pc0 != 0 else 1
        entS = -(pc1 * math.log2(pc1_) + pc0 * math.log2(pc0_))
        gain = entS - min_cpe

        remained_pos = total_sum - partition_sum
        total_rem = total_size - partition_count
        c0 = 2 if (partition_sum == 0 or partition_sum == partition_count) else 1
        c1 = 2 if (remained_pos == 0 or remained_pos == total_rem) else 1
        delta = math.log2(9) - c0 * partition_entS0 - c1 * partition_entS1

        delta = math.log2(7) -( 2*entS -c0 * partition_entS0 - c1 * partition_entS1)

        if (gain <= ((math.log2(total_size - 1) + delta) / total_size)):
            if force:
                return [partition_x]
            return []

        #recoursive splitting
        left_partitions = self.find_partitions(df.iloc[:(partition_index+1)], partition_count, partition_sum)
        right_partitions= self.find_partitions(df.iloc[(partition_index+1):], (total_size-partition_count), (total_sum-partition_sum))
        a= [partition_x]+ left_partitions + right_partitions
        return a


class EqualFrequency:
    """
    This class find the cut points to discretize a numerical feature using an approximate equal frequency discretization.
    """
    def __init__(self, min_bin_size=1, num_bins=0):
        """
        Parameters
        __________
        num_bins : int
            this number is to interpret as the maximum number of bins that will be generated.
            If this parameter is not specified (0 by deafault), it will be automatically determined.
        min_bin_size : int
            Represent the minimum size that a bin can have.

        Notes
        -----
        If the number of bins has to be automatically determined, this will be chosen so that each bin has an average
        size of 1.2 * min_bin_size. Let's call this number automatic_nbin
        ( automatic_nbin = int(x.size/(self.min_group_size*1.2)) )
        If instead the bin number is specified in the constructor (num_bins parameter), the number of bins that will
        actually be generated will be, at most, equal to the minimum between num_bins and automatic_nbin. This is
        to ensure that the constraint given by parameter min_bin_size is respected.
        """
        self.min_group_size = min_bin_size
        self.num_bins = num_bins

    def findCutPoints(self, x):
        """
        :param x: numpy array or pandas series
        :return: list of (ascending) ordered cut points
        """
        if self.num_bins >1:
            num_bins = min(self.num_bins, int(x.size/(self.min_group_size*1.2)))
        else:
            num_bins = int(x.size/(self.min_group_size*1.2))
        if num_bins < 2:
            return []

        avg_group_size = x.size / num_bins

        if isinstance(x, pd.Series):
            x = x.to_numpy()
        val, counts = np.unique(x, return_counts=True)

        quantiles = [] #actually this array will contains quantiles * x.size
        sum =0
        for c in counts:
            sum = sum+c
            quantiles.append(sum)

        cut_points = []
        bins_size = []
        next_expected_quantile = avg_group_size #again, is quantile * x.size
        binsize = 0
        for i in range(len(quantiles)):
            binsize = binsize + counts[i]
            if quantiles[i] >= next_expected_quantile:
                if i==0:
                    cut_points.append(val[i])
                    bins_size.append(binsize)
                    binsize = 0
                else:
                    up = quantiles[i] - next_expected_quantile
                    low = next_expected_quantile - quantiles[i-1]
                    if up <= low:
                        cut_points.append(val[i])
                        bins_size.append(binsize)
                        binsize = 0
                    else:
                        cut_points.append(val[i-1])
                        bins_size.append(binsize-counts[i])
                        binsize = counts[i]
                while next_expected_quantile <= quantiles[i]:
                    next_expected_quantile = next_expected_quantile + avg_group_size
                if next_expected_quantile >= x.size:
                    bins_size.append(x.size - quantiles[i] - binsize)
                    break
        if len(cut_points) == 0:
            return []

        #The bins with size <= min_group_size will be merged with one of the other adiacent bins
        self.mergeSmallBins(cut_points, bins_size)
        return cut_points


    def mergeSmallBins(self, cut_points, bins_size):
        """
        :param cut_points: list
        :param bins_size: list
        :return: void
        """
        min_smallbin = self.min_group_size
        min_index = 0
        for i in range(len(bins_size)):
            if bins_size[i]<min_smallbin:
                min_smallbin = bins_size[i]
                min_index = i
        if min_smallbin == self.min_group_size:
            return

        previous_size = 0
        next_size = 0
        if min_index > 0:
            previous_size = bins_size[min_index - 1]
        if min_index < (len(bins_size)-1):
            next_size = bins_size[min_index + 1]

        if previous_size == 0 and next_size ==0:
            return

        if (previous_size == 0) or (next_size > 0 and next_size < previous_size):
            bins_size[min_index+1] = bins_size[min_index+1]+bins_size[min_index]
            cut_points.pop(min_index)
            bins_size.pop(min_index)
        else:
            bins_size[min_index - 1] = bins_size[min_index - 1] + bins_size[min_index]
            cut_points.pop(min_index -1)
            bins_size.pop(min_index -1)

        self.mergeSmallBins(cut_points, bins_size)


class EqualWidth:
    """
        This class find the cut points to discretize a numerical feature using the equal width discretization.
        """
    def __init__(self, min_bins_size=1, num_bins=0):
        """
        Parameters
        __________
        num_bins : int
            this number is to interpret as the maximum number of bins that will be generated.
            If this parameter is not specified (0 by deafault), it will be automatically determined.
        min_bins_size : int

        Notes
        -----
        min_bins_size and num_bins parameters are to be interpreted as for the EqualFreq class
        """
        self.min_group_size = min_bins_size
        self.num_bins = num_bins

    def findCutPoints(self, x):
        """

        :param x: numpy array or pandas series
        :return: list of (ascending) ordered cut points
        """
        if self.num_bins > 1:
            num_bins = min(self.num_bins, int(x.size / (self.min_group_size * 1.2)))
        else:
            num_bins = int(x.size / (self.min_group_size * 1.2))
        if num_bins < 2:
            return []

        if isinstance(x, pd.Series):
            x = x.to_numpy()
        min = x.min()
        bin_width = (x.max() - min)/num_bins
        cut_points = []
        current_cut = min + bin_width
        for i in range(1, num_bins):
            cut_points.append(current_cut)
            current_cut = current_cut + bin_width

