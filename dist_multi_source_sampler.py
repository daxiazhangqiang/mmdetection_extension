# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import Iterator, List, Optional, Sized, Union
import math
import bisect
import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class DistributedMultiSourceSampler(Sampler):
    """Distributed multi-source finite sampler for EpochBasedTrainLoop.

    The key difference from :class:`MultiSourceSampler` is that this sampler
    yields a fixed number of samples from the concatenated dataset without
    resampling, avoiding an effectively infinite dataset length.

    For each iteration, all ranks draw samples from the same sub-dataset.
    Each rank receives ``batch_size`` samples per iteration, and the total
    samples consumed per iteration across all ranks is
    ``world_size * batch_size == iter_sample_num``.

    Args:
        dataset (Sized): A concatenated dataset (ConcatDataset).
        batch_size (int): Number of samples per rank per iteration. Must match
            the ``batch_sampler`` setting of the DataLoader.
        shuffle (bool): Whether to shuffle the order of batches (sub-dataset
            blocks) across iterations. Defaults to False.
        seed (int, optional): Random seed for reproducibility. If None, a
            random seed is synchronized across distributed processes.
            Defaults to None.
        drop_last (bool, optional): Whether to drop incomplete tail blocks.
            For this sampler, drop_last is fixed to True to ensure consistent
            behavior across all ranks. Defaults to True.
        rank (int, optional): Rank of the current process. If None, inferred
            from the distributed environment.
        world_size (int, optional): Total number of processes. If None,
            inferred from the distributed environment.
    """

    def __init__(self,
                 dataset: Sized,
                 batch_size: int,
                 shuffle: bool = False,
                 drop_last: bool = True,
                 seed: Optional[int] = None,
                 rank: int = None,
                 world_size: int = None,
                 ) -> None:

        assert hasattr(dataset, 'datasets') and hasattr(dataset, 'cumulative_sizes'),\
            f'The dataset must be ConcatDataset, but get {dataset}'
        assert isinstance(batch_size, int) and batch_size > 0, \
            'batch_size must be a positive integer value, ' \
            f'but got batch_size={batch_size}'
        if rank is None or world_size is None:
            rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.iter_sample_num = world_size * batch_size  
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = sync_random_seed() if seed is None else seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.cumulative_sizes = [0] + dataset.cumulative_sizes
        # safety check
        self.safety_check()
        # num_batch
        self.num_batch, self.num_batch_index= self.get_num_batch()
        # total_num_sample
        self.num_samples = self.num_batch * self.batch_size 
        # init epoch
        self.epoch = 0
    def get_num_batch(self):
        num_batch = 0
        num_batch_index = []
        for _, _dataset in enumerate(self.dataset.datasets):
            num_batch_index.append(num_batch)
            num_batch += len(_dataset) // self.iter_sample_num
        return num_batch, num_batch_index
    
    def safety_check(self):
        assert self.drop_last == True, 'drop last should be set to true for ' \
            f'DistributedMultiSourceSampler'
        for i in range(len(self.dataset.datasets)):
            assert hasattr(self.dataset.datasets[i], 'test_mode'), "each dataset " \
                f"should has attribute 'test_mode'"
            assert self.dataset.datasets[i].test_mode == self.dataset.datasets[0].test_mode, \
                f"'test_mode' attribute should be same for all concated datasets"
        if self.dataset.datasets[0].test_mode and self.shuffle:
            raise RuntimeError("shuffle should be true for test_mode")

    def get_cumulative_size(self, idx):
        dataset_idx = bisect.bisect_right(self.num_batch_index, idx) - 1
        concat_dataset_idx = self.cumulative_sizes[dataset_idx] + \
            (idx - self.num_batch_index[dataset_idx])*self.iter_sample_num
        return concat_dataset_idx

    def __iter__(self):
        iter_seqs = []
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            iter_seqs = torch.randperm(self.num_batch, generator=g).tolist()  # type: ignore[arg-type]
        else:
            iter_seqs = list(range(self.num_batch))  # type: ignore[arg-type]

        # remove tail of data to make it evenly divisible.
        indices = []
        for i in iter_seqs:
            temp_index = self.get_cumulative_size(i) + np.arange(self.iter_sample_num)
            indices.extend(list(temp_index.astype(np.int64)))
        # subsample
        rank_indices = indices[self.rank:len(indices):self.world_size]
        assert len(rank_indices) == self.num_samples, f"{len(rank_indices)} != {self.num_samples}"

        return iter(rank_indices)
    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch