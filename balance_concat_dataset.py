from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Dict
)

import logging
import bisect
from mmengine.utils import is_seq_of
from mmengine.registry import DATASETS
from mmengine.dataset import ConcatDataset as MMENGINE_ConcatDataset
from mmengine.dataset import force_full_init
from mmengine.logging import print_log

@DATASETS.register_module()
class BalanceConcatDataset(MMENGINE_ConcatDataset):
    def __init__(self, datasets=[], repeat_factors=[], lazy_init = False, ignore_keys = None):
        super().__init__(datasets, lazy_init, ignore_keys)
        self.origin_len = [len(s_dataset) for s_dataset in self.datasets]
        if repeat_factors and is_seq_of(repeat_factors, (int,float)):
            assert len(repeat_factors) == len(self.datasets), f"len(repeat_factors)" \
                f"need to be the same length of concated datasets number"
            self.repeat_len = list(map(int, [self.origin_len[i]*repeat_factors[i] \
                                            for i in range(len(repeat_factors))]))
            print_log(f"[{self.__class__.__name__}]origin_len={self.origin_len}," \
                      f"repeat_len={self.repeat_len}",'current')

        #overwrite self.cumulative_sizes
        self.cumulative_sizes = self.cumsum_len(self.repeat_len)

    @staticmethod
    def cumsum_len(lst):
        res = []
        s = 0
        for v in lst:
            s += v
            res.append(s)

        return res
    
    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to '
                'accelerate the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx][sample_idx]
    
    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> Tuple[int, int]:
        """Convert global idx to local index.

        Args:
            idx (int): Global index of ``RepeatDataset``.

        Returns:
            Tuple[int, int]: The index of ``self.datasets`` and the local
            index of data.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    f'absolute value of index({idx}) should not exceed dataset'
                    f'length({len(self)}).')
            idx = len(self) + idx
        # Get `dataset_idx` to tell idx belongs to which dataset.
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        # Get the inner index of single dataset.
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        sample_idx = sample_idx % self.origin_len[dataset_idx]
        return dataset_idx, sample_idx

    def __len__(self):
        return sum(self.repeat_len)
    


