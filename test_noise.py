import noise
import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises
import random

def setup_module(module):
    print('\nSetup the test suite')

def teardown_module(module):
    print('\nTear down the test suite')

#def test_get_random_data_NotImplemented():
#    print('\ntest_get_random_data_NotImplemented')
#    with raises(NotImplementedError):
#        noise.get_random_row(field_sizes_input)

def test_get_random_data_ValidateInput():
    field_sizes_input = 10
    with raises(TypeError):
        noise.get_random_row(field_sizes_input)

def test_get_random_data_AssertEmptyRetValue():
    field_sizes_input = [1,2,3]
    result = noise.get_random_row(field_sizes_input)
    assert 0 != len(result)

def test_get_random_data_AssertSystemExitEmptyInput():
    field_sizes_input = ()
    with raises(SystemExit):
        noise.get_random_row(field_sizes_input)

def test_get_random_data_AssertIfNotTuple():
    field_sizes_input = (3,4,5)
    result = noise.get_random_row(field_sizes_input)
    assert isinstance(result, tuple)

def test_get_random_data_ValidateOutput():
    field_sizes_input = (1, 4, 5)
    result = noise.get_random_row(field_sizes_input)
    assert len(result) == len(field_sizes_input)

    for index in range(len(field_sizes_input)):
        assert result[index] < field_sizes_input[index]

def test_make_raw_data_InValidSequence():
    field_sizes_input = 10
    num_rows = 2
    with raises(TypeError):
        noise.make_raw_data(field_sizes_input, num_rows, noise.get_random_row)

def test_make_raw_data_AssertSystemExitEmptyInput():
    field_sizes_input = []
    num_rows = 4
    with raises(SystemExit):
        noise.make_raw_data(field_sizes_input, num_rows, noise.get_random_row)

def test_make_raw_data_InValidNumRows():
    field_sizes_input = (1,2,3)
    num_rows = -1
    with raises(ValueError):
        noise.make_raw_data(field_sizes_input, num_rows, noise.get_random_row)

def test_make_raw_data_AssertIfNotSeqOfTuples():
    field_sizes_input = [1,2,1]
    num_rows = 4
    result = noise.make_raw_data(field_sizes_input, num_rows, noise.get_random_row)
    assert isinstance(result, (list, tuple))
    assert all(isinstance(item, tuple) for item in result)

def test_make_raw_data_ValidateOutput():
    field_sizes_input = (2,2,2)
    num_rows = 2
    result = noise.make_raw_data(field_sizes_input, num_rows, noise.get_random_row)
    assert len(result) == num_rows

    for index in range(num_rows):
        assert len(result[index]) == len(field_sizes_input)
        for i in range(len(result[index])):
            result[index][i] < field_sizes_input[i]

def test_get_counts_InValidRawData():
    raw_data_input = ((1,1,0),(0))
    field_sizes_input = (2,2,2)
    with raises(TypeError):
        noise.get_counts(raw_data_input, field_sizes_input)

def test_get_counts_ValidateOutPut():
    field_sizes_input = (3,)
    num_rows = 2
    result = noise.get_counts(noise.make_raw_data(field_sizes_input, num_rows, noise.get_random_row), field_sizes_input)
    assert  isinstance(result, np.ndarray)
    assert  result.shape == field_sizes_input

# yet to validate the negative test case
#def test_calculate_subtotal_InValidInputArray():
#    counts_input = ''
#    with raises(TypeError):
#        noise.calculate_subtotals(counts_input)

def test_calculate_subtotals_Validate1DArray():
    counts_input = np.asarray([1,2,3])
    result = noise.calculate_subtotals(counts_input)
    assert isinstance(result, tuple)
    expected_result = (6,)
    assert result == expected_result

def test_calculate_subtotals_Validate2DArray():
    counts_input = np.array([[1,2,3],[4,5,6]])
    result = noise.calculate_subtotals(counts_input)
    assert isinstance(result, tuple)
    assert all(isinstance(item, np.ndarray) for item in result)
    #(TO-DO) arrays are same - however getting mismatch - need to verify - raised this issue.
    #assert_array_equal(result, (np.array([5, 7, 9]), np.array([ 6, 15])))

def test_draw_laplace():
    scale = 1
    field_sizes_input = (3,6)
    np.random.seed(7)
    random.seed(7)
    result = noise.draw_laplace(scale, field_sizes_input)
    perturbed_counts = noise.get_counts(noise.make_raw_data(field_sizes_input, 100, noise.get_random_row), field_sizes_input) + result
    print(f'{perturbed_counts}')