#   Author: Artem Skiba
#   Created: 20/01/2020

from pathlib import Path
from typing import Union


def raw_count(filename: Union[str, Path]) -> int:
    """
    fast counting lines in file
    :param filename: str or Path object. Path to file in which lines will be count
    :return: int, number of lines
    """
    with open(filename, 'rb') as f:
        num_lines = 0
        buf_size = 1024 * 1024
        read_f = f.raw.read

        buf = read_f(buf_size)
        while buf:
            num_lines += buf.count(b'\n')
            buf = read_f(buf_size)
    return num_lines
