import pandas as pd
import math

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


