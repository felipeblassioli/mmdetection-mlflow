# custom_sampler.py
from mmengine.logging import MMLogger, print_log
from mmengine.dataset import DefaultSampler
from mmdet.registry import DATA_SAMPLERS
import random


@DATA_SAMPLERS.register_module()
class SubsetSampler(DefaultSampler):
    def __init__(self, subset_ratio=0.1, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.subset_ratio = subset_ratio
        self.seed = seed

        logger = MMLogger.get_current_instance()
        logger.info(f"SubsetSampler initialized with subset_ratio: {subset_ratio}, seed: {seed}")

    def __iter__(self):
        # Get the full list of indices
        indices = list(range(len(self.dataset)))
        
        # Set random seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)
        
        # Subsample indices
        subset_size = int(len(indices) * self.subset_ratio)
        indices = random.sample(indices, subset_size)
        
        # Shuffle if required
        if self.shuffle:
            random.shuffle(indices)
            
        return iter(indices)