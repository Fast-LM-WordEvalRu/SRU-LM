#   Author: Artem Skiba
#   Created: 20/01/2020

from .dataset import FastDataset
from .dataloader import get_dataloader

__all__ = [
    'FastDataset', 'get_dataloader'
]
