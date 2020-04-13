from typing import Sequence, Tuple, Callable
import random
from collections import Counter
import itertools
import logging
import numpy as np
import sys

# Assume all fields are categorical with some number of possible values (0, 1, 2, etc),
# and fields don't have names, just their index.

# Type-check this using:
#     mypy noise/ --ignore-missing-imports --strict-optional


def get_random_row(field_sizes: Sequence[int]) -> Tuple[int, ...]:
    """
    Given the number of possible values n for each field,
    and assuming these values can be simply referenced by their index 0..n-1,
    return a random index for each field.
    Eg. random_row([5, 7, 3]) = (4, 0, 2).
    for a field of value 5 - the possible values are 0 , 1, 2, 3, 4

    """

    try:
        isinstance(field_sizes, (list, tuple))
    except:
        logging.debug('TypeError: The input is not of sequence type')
        raise TypeError
        sys.exit(1)

    if(len(field_sizes) == 0):
        logging.debug("Error: Input sequence is empty")
        sys.exit(1)

    row = tuple(random.randint(0, r_num-1) for r_num in field_sizes)

    return row



def make_raw_data(field_sizes: Sequence[int],
                  num_rows: int,
                  get_random_row:Callable[[Sequence[int]], Tuple[int, ...]]
                  ) -> Sequence[Tuple[int, ...]]:
    """
    Create num_rows rows of data, using the supplied get_random_row function.
    """

    try:
        isinstance(field_sizes, (list, tuple))
    except:
        logging.debug('TypeError: The input is not of sequence type')
        raise TypeError
        sys.exit(1)

    if (len(field_sizes) == 0):
        logging.debug("Error: Input sequence is empty")
        sys.exit(1)

    if num_rows <= 0:
        raise ValueError

    return [get_random_row(field_sizes) for i in range(num_rows)]


def get_counts(raw_data: Sequence[Tuple[int, ...]],
               field_sizes: Sequence[int],
               ) -> np.array:
    """
    Count the number of times each possible tuple of data occurs,
    and return the counts as a numpy array, with the same number of dimensions as the number of fields in the data.
    Ie. count[i][j][k] = the count of tuples (i, j, k) in the raw data.
    """

    try:
        isinstance(raw_data, (list, tuple))
    except:
        logging.debug('TypeError: The input is not of sequence type')
        raise TypeError
        sys.exit(1)

    try:
        all(isinstance(item, tuple) for item in raw_data)
    except:
        logging.debug('TypeError: The input is not of tuple type')
        raise TypeError
        sys.exit(1)

    try:
        isinstance(field_sizes, (list, tuple))
    except:
        logging.debug('TypeError: The input is not of sequence type')
        raise TypeError
        sys.exit(1)

    if (len(field_sizes) == 0):
        logging.debug("Error: Input sequence is empty")
        sys.exit(1)

    shape = tuple(field_size for field_size in field_sizes)
    possibilities = tuple(range(field_size) for field_size in shape)
    combinations = itertools.product(*possibilities)
    counts = Counter(tuple(x for x in row) for row in raw_data)
    return np.array(tuple(counts[combination] for combination in combinations)).reshape(*shape)


def calculate_subtotals(counts: np.array) -> Sequence[np.array]:
    """
    Given a numpy array of counts (of any dimension), return a tuple of subtotals.
    Eg. if count[i][j][k] = the count of tuples (i, j, k), then return
    (sum count[j][k] over i, sum count[i][k] over j, sum count[i][j] over k).
    >>> calculate_subtotals(np.array([1, 2, 3]))
    (6,)
    >>> calculate_subtotals(np.array([[1, 2, 3], [4, 5, 6]]))
    (array([5, 7, 9]), array([ 6, 15]))
    >>> calculate_subtotals(np.array([[[1, 2, 3],[4, 5, 6]],[[-10, -10, -10],[50, 50, 50]]]))
    (array([[-9, -8, -7],
           [54, 55, 56]]), array([[ 5,  7,  9],
           [40, 40, 40]]), array([[  6,  15],
           [-30, 150]]))
    """
    try:
        isinstance(counts, np.ndarray)
    except:
        logging.debug('TypeError: Input paramter is not ndarray type.')
        raise TypeError
        sys.exit(1)

    final_result = ()
    count = 0
    for x in range(counts.ndim):
        result = np.cumsum(counts, axis=x)
        if count > 2:
            count = 0
        else:
            if count == 0:
                final_result += (result[len(result) - 1],)
                count += 1
            elif count == 1:
                val = ()
                for i in range(len(result)):
                    for j in range(len(result[i])):
                        if j == (len(result[i]) - 1):
                            val += (result[i][j],)
                final_result += (np.asarray(val),)
                count += 1
            elif count == 2:
                val = ()
                for i in range(len(result)):
                    tempVal = ()
                    for j in range(len(result[i])):
                        for k in range(len(result[i][j])):
                            if k == (len(result[i][j]) - 1):
                                tempVal += (result[i][j][k],)
                    val += (np.asarray(tempVal),)
                final_result += (np.asarray(val),)
                count += 1
    return final_result


def draw_laplace(scale: float,
                 field_sizes: Sequence[int],
                 ) -> np.array:
    """
    Return a numpy array of Laplace-distributed noise, suitable to add to the counts returned by get_counts.
    """
    return np.random.laplace(0, scale, field_sizes)


if __name__ == '__main__':
    field_sizes = (3, 6)  # Also check it with tuples of different lengths.
    num_rows = 100
    np.random.seed(7)
    random.seed(7)
    raw_data = make_raw_data(field_sizes=field_sizes, num_rows=num_rows, get_random_row=get_random_row)

    true_counts = get_counts(raw_data=raw_data, field_sizes=field_sizes)
    print('true counts\n', true_counts)

    perturbed_counts = true_counts + draw_laplace(scale=1, field_sizes=field_sizes)
    print('perturbed counts\n', np.round(perturbed_counts, 1))

    true_totals = calculate_subtotals(counts=true_counts)
    print('totals of true counts\n', true_totals)

    # Please complete this last line - the totals of the perturbed counts.
    print(f'totals of perturbed counts:\n {perturbed_counts}')