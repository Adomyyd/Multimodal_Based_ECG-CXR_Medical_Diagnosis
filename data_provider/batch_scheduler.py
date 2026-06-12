import math
import random
from torch.utils.data import RandomSampler, Sampler
import torch
import torch.distributed as dist


class BatchSchedulerSampler_ever2(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        # [] max len
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            cnt = 0  # cnt=0获得第一个batch, cnt=1时获得第二个同一个数据集batch
            i = 0
            while i != self.number_of_datasets:
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                i += 1
                cnt += 1
                if cnt & 1 == 0:
                    cnt = 0
                else:
                    i -= 1
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


class SeqBatchSchedulerSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        # [] max len
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


class WeightBatchSchedulerSampler(Sampler):
    def __init__(self, dataset, batch_size, train_size, weights):
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_size = train_size
        self.weights = weights
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        # [] all len
        # return self.batch_size * math.ceil(self.train_size / self.batch_size) * len(self.dataset.datasets)
        return self.train_size

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []

        # 获得每个数据集的sample list和iter list
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size
        samples_to_grab = self.batch_size

        epoch_samples = self.train_size
        print(epoch_samples)

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            # random choose one dataset by weight
            i = random.choices(range(self.number_of_datasets), weights=self.weights)
            i = i[0] # [dataset_idx] -> dataset_idx
            
            cur_batch_sampler = sampler_iterators[i]
            cur_samples = []

            for _ in range(samples_to_grab):
                try:
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org + push_index_val[i]
                    cur_samples.append(cur_sample)
                except StopIteration:
                    # got to the end of iterator - restart the iterator and continue to get samples
                    # until reaching "epoch_samples"
                    sampler_iterators[i] = samplers_list[i].__iter__()
                    cur_batch_sampler = sampler_iterators[i]
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org + push_index_val[i]
                    cur_samples.append(cur_sample)

            final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


class DistributedWeightBatchSchedulerSampler(Sampler):
    def __init__(self, dataset, batch_size, train_size, weights, num_replicas=None, rank=None, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.train_size = train_size
        self.weights = weights
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.number_of_datasets = len(dataset.datasets)

        # Number of full batches we want to generate in one epoch
        self.num_batches = math.ceil(self.train_size / self.batch_size)
        # Total number of batches must be divisible by num_replicas to assign evenly
        self.padded_num_batches = math.ceil(self.num_batches / self.num_replicas) * self.num_replicas

        # Each rank will get: padded_num_batches // num_replicas batches
        self.batches_per_rank = self.padded_num_batches // self.num_replicas
        self.num_samples = self.batches_per_rank * self.batch_size  # samples per rank

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch if hasattr(self, 'epoch') else self.seed)

        # Initialize samplers for each sub-dataset
        samplers_list = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = torch.utils.data.RandomSampler(
                cur_dataset,
                generator=torch.Generator().manual_seed(self.seed + dataset_idx)  # optional: unique seed per dataset
            )
            samplers_list.append(iter(sampler))

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]

        # Step 1: Generate list of complete batches (each is a list of global indices)
        batches = []  # List[List[int]]
        for _ in range(self.num_batches):
            # Choose a dataset by weight
            i = torch.multinomial(torch.tensor(self.weights, dtype=torch.float), 1, generator=g).item()

            cur_batch = []
            for _ in range(self.batch_size):
                try:
                    idx = next(samplers_list[i])
                except StopIteration:
                    # Recreate iterator with same seed for reproducibility
                    samplers_list[i] = iter(torch.utils.data.RandomSampler(
                        self.dataset.datasets[i],
                        generator=torch.Generator().manual_seed(self.seed + i)
                    ))
                    idx = next(samplers_list[i])
                global_idx = idx + push_index_val[i]
                cur_batch.append(global_idx)
            batches.append(cur_batch)

        # Step 2: Pad with copies of the last batch to make total batches divisible by num_replicas
        while len(batches) < self.padded_num_batches:
            batches.append(batches[-1][:])  # copy last batch

        # Step 3: Assign complete batches to this rank
        my_batches = batches[self.rank :: self.num_replicas]
        assert len(my_batches) == self.batches_per_rank

        # Step 4: Flatten into sample indices for this rank
        indices = [idx for batch in my_batches for idx in batch]
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch