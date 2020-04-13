# Laplace-distributed-noise
Technique to design private mechanisms for real-valued functions on sensitive data.

Adding controlled noise from predetermined distributions - a common technique to maintain privacy is to add noise to the results of queries. Some commonly used distributions for adding noise include Laplace and Gaussian distributions.
I have used laplace distribution here in this example.

Install Scikit-learn:
I have followed the following doc and it works.
https://scikit-learn.org/stable/install.html

Alternatively you can install Anaconda which offers scikit-learn:
Follow this :  https://teamtreehouse.com/library/installing-scikitlearn-using-anaconda

Either of these should be fine.
I developed and ran this module in Ubuntu and Fedora.

Knowledge of Python and numpy library is must.

How it works:

```
    Given a numpy array of counts (of any dimension), return a tuple of subtotals.
    Eg. if count[i][j][k] = the count of tuples (i, j, k), then return
    (sum count[j][k] over i, sum count[i][k] over j, sum count[i][j] over k).
    >>> calculate_subtotals(np.array([1, 2, 3]))
    (6,)
    >>> calculate_subtotals(np.array([[1, 2, 3], [4, 5, 6]]))
    (array([5, 7, 9]), array([6, 15]))
    >>> calculate_subtotals(np.array([
          [[1, 2, 3],
           [4, 5, 6]],
          [[-10, -10, -10],
           [50, 50, 50]]
        ]))
    (array([[-9, -8, -7], [54, 55, 56]]),
     array([[ 5,  7,  9], [40, 40, 40]]),
     array([[  6,  15], [-30, 150]]))
     ```
     
