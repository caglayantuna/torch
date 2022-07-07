from tensorly.decomposition._base_decomposition import DecompositionMixin
import torch
from torch.optim import Adam
import tensorly as tl
tl.set_backend('pytorch')
from tensorly.random import random_cp
from tensorly.cp_tensor import CPTensor, validate_cp_rank
import math


def initialize_generalized_parafac(tensor, rank, init='random', svd='numpy_svd', non_negative=False, random_state=None):
    r"""Initialize factors used in `generalized parafac`.

    Parameters
    ----------
    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices with uniform distribution using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor. If init is a previously initialized `cp tensor`, all
    the weights are pulled in the last factor and then the weights are set to "1" for the output tensor.

    Parameters
    ----------
    tensor : ndarray
    rank : int, number of components in the CP decomposition
    init : {'svd', 'random', cptensor}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    loss : {'gaussian', 'gamma', 'rayleigh', 'poisson_count', 'poisson_log', 'bernoulli_odds', 'bernoulli_log'}
        Some loss functions require positive factors, which is enforced by clipping
    random_state : {None, int, np.random.RandomState}
    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    rng = tl.check_random_state(random_state)
    if init == 'random':
        kt = random_cp(tl.shape(tensor), rank, random_state=rng, normalise_factors=False, **tl.context(tensor))

    elif init == 'svd':
        try:
            svd_fun = tl.SVD_FUNS[svd]
        except KeyError:
            message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                      svd, tl.get_backend(), tl.SVD_FUNS)
            raise ValueError(message)

        factors = []
        for mode in range(tl.ndim(tensor)):
            U, S, _ = svd_fun(tl.unfold(tensor, mode), n_eigenvecs=rank)

            # Put SVD initialization on the same scaling as the tensor in case normalize_factors=False
            if mode == 0:
                idx = min(rank, tl.shape(S)[0])
                U = tl.index_update(U, tl.index[:, :idx], U[:, :idx] * S[:idx])

            if tensor.shape[mode] < rank:
                random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                U = tl.concatenate([U, random_part], axis=1)

            factors.append(U[:, :rank])
        kt = CPTensor((None, factors))
    elif isinstance(init, (tuple, list, CPTensor)):
        try:
            weights, factors = CPTensor(init)

            if tl.all(weights == 1):
                weights, factors = CPTensor((None, factors))
            else:
                weights_avg = tl.prod(weights) ** (1.0 / tl.shape(weights)[0])
                for i in range(len(factors)):
                    factors[i] = factors[i] * weights_avg
            kt = CPTensor((None, factors))
            return kt
        except ValueError:
            raise ValueError(
                'If initialization method is a mapping, then it must '
                'be possible to convert it to a CPTensor instance'
            )
    else:
        raise ValueError('Initialization method "{}" not recognized'.format(init))
    if non_negative:
        kt.factors = [tl.abs(f) for f in kt[1]]
    return kt


def vectorize_factors(factors):
    """
    Vectorizes each factor in factors, then concatenates them to return one vector.

    Parameters
    ----------
    factors : list of ndarray

    Returns
    -------
    vectorized_factors: vector
    """
    vectorized_factors = []
    for i in range(len(factors)):
        vectorized_factors.append(tl.tensor_to_vec(factors[i]))
    vectorized_factors = tl.concatenate(vectorized_factors, axis=0)
    return vectorized_factors


def vectorized_factors_to_tensor(vectorized_factors, shape, rank, mask=None, return_factors=False):
    """
    Transforms vectorized factors of a CP decomposition into a reconstructed full tensor.

    Parameters
    ----------
    vectorized_factors : 1d array, a vector of length :math:`\prod(shape) * rank^{len(shape)}`
    shape : tuple, contains the row dimensions of the factors
    rank : int, number of components in the CP decomposition
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else.
    return_factors : bool, if True returns factors list instead of full tensor
        Default: False

    Returns
    -------
    tensor: ndarray or list of ndarrays
    """
    n_factors = len(shape)
    factors = []

    cursor = 0
    for i in range(n_factors):
        factors.append(tl.reshape(vectorized_factors[cursor: cursor + shape[i] * rank], [shape[i], rank]))
        cursor += shape[i] * rank

    if return_factors:
        return CPTensor((None, factors))
    else:
        if mask is not None:
            return tl.cp_to_tensor((None, factors)) * mask
        else:
            return tl.cp_to_tensor((None, factors))


def loss_operator_func(tensor, rank, loss, batch_size=None, mask=None):
    """
    Various loss functions for generalized parafac decomposition, see [1] for more details.
    The returned function maps a vectorized factors input x to the loss :math:`1/len(x) * L(T,x)`
    where L is the maximum likelihood estimator when tensor is generated from x using one of the following distributions:

    * Gaussian
    * Gamma
    * Rayleigh
    * Poisson (count or log)
    * Bernoulli (odds or log)

    Parameters
    ----------
    tensor : ndarray, input tensor data
    rank : int, number of components in the CP decomposition
    loss : string, choices are {'gaussian', 'gamma', 'rayleigh', 'poisson_count', 'poisson_log', 'bernoulli_odds', 'bernoulli_log'}
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else.

    Returns
    -------
    function to compute loss
        Size based normalized loss for each entry

    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    """
    shape = tl.shape(tensor)
    if batch_size is None:
        batch_size = tl.prod(tl.tensor(shape, **tl.context(tensor)))
    epsilon = 1e-8
    rng = tl.check_random_state(None)

    if loss == 'gaussian':
        def func(x):
            indices_tuple = tuple([rng.randint(0, shape[i], size=batch_size, dtype=int) for i in range(len(shape))])
            est = vectorized_factors_to_tensor(x, shape, rank, mask)
            return tl.sum((tensor[indices_tuple] - est[indices_tuple]) ** 2) / batch_size
        return func
    elif loss == 'bernoulli_odds':
        def func(x):
            indices_tuple = tuple([rng.randint(0, shape[i], size=batch_size, dtype=int) for i in range(len(shape))])
            est = vectorized_factors_to_tensor(x, shape, rank, mask)
            return tl.sum(tl.log(est[indices_tuple] + 1) - (tensor[indices_tuple] * tl.log(est[indices_tuple] + epsilon))) / batch_size
        return func
    elif loss == 'bernoulli_logit':
        def func(x):
            indices_tuple = tuple([rng.randint(0, shape[i], size=batch_size, dtype=int) for i in range(len(shape))])
            est = vectorized_factors_to_tensor(x, shape, rank, mask)
            return tl.sum(tl.log(tl.exp(est[indices_tuple]) + 1) - (tensor[indices_tuple] * est[indices_tuple])) / batch_size
        return func
    elif loss == 'rayleigh':
        def func(x):
            indices_tuple = tuple([rng.randint(0, shape[i], size=batch_size, dtype=int) for i in range(len(shape))])
            est = vectorized_factors_to_tensor(x, shape, rank, mask)
            return tl.sum(2 * tl.log(est[indices_tuple] + epsilon) + (math.pi / 4) * ((tensor[indices_tuple] / (est[indices_tuple] + epsilon)) ** 2)) / batch_size
        return func
    elif loss == 'poisson_count':
        def func(x):
            indices_tuple = tuple([rng.randint(0, shape[i], size=batch_size, dtype=int) for i in range(len(shape))])
            est = vectorized_factors_to_tensor(x, shape, rank, mask)
            return tl.sum(est[indices_tuple] - tensor[indices_tuple] * tl.log(est[indices_tuple] + epsilon)) / batch_size
        return func
    elif loss == 'poisson_log':
        def func(x):
            indices_tuple = tuple([rng.randint(0, shape[i], size=batch_size, dtype=int) for i in range(len(shape))])
            est = vectorized_factors_to_tensor(x, shape, rank, mask)
            return tl.sum(tl.exp(est[indices_tuple]) - (tensor[indices_tuple] * est[indices_tuple])) / batch_size
        return func
    elif loss == 'gamma':
        def func(x):
            indices_tuple = tuple([rng.randint(0, shape[i], size=batch_size, dtype=int) for i in range(len(shape))])
            est = vectorized_factors_to_tensor(x, shape, rank, mask)
            return tl.sum(tensor[indices_tuple] / (est[indices_tuple] + epsilon) + tl.log(est[indices_tuple] + epsilon)) / batch_size
        return func
    else:
        raise ValueError('Loss "{}" not recognized'.format(loss))


def stochastic_generalized_parafac(tensor, rank, n_iter_max=1000, init='random', return_errors=False,
                                   loss='gaussian', batch_size=200, lr=0.001, beta_1=0.9, beta_2=0.999,
                                   mask=None, random_state=None):
    """ Generalized PARAFAC decomposition by using ADAM optimization.
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::

        tensor ~ D([|weights; factors[0], ..., factors[-1] |]) 

    where D is a parametric distribution such as Gaussian, Poisson, Rayleigh, Gamma or Bernoulli.

    Generalized parafac essentially performs the same kind of decomposition as the parafac function,
    but using a more diverse set of user-chosen loss functions. Under the hood, it relies on stochastic
    optimization using a home-made implementation of ADAM.

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initialization.
        See `initialize_factors`.
    return_errors : bool, optional
        Activate return of iteration errors
    loss : {'gaussian', 'bernoulli_odds', 'bernoulli_logit', 'rayleigh', 'poisson_count', 'poisson_log', 'gamma'}
        Default : 'gaussian'
    batch_size : int
        Default : 200
    lr : float
        Default : 0.001
    beta_1 : float
        ADAM optimization parameter.
        Default : 0.9
    beta_2 : float
        ADAM optimization parameter.
        Default : 0.999
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else.
    random_state : {None, int, np.random.RandomState}

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    .. [2] Kolda, T. G., & Hong, D. (2020). Stochastic gradients for large-scale tensor decomposition.
           SIAM Journal on Mathematics of Data Science, 2(4), 1066-1095.
    """
    rank = validate_cp_rank(tl.shape(tensor), rank=rank)
    rng = tl.check_random_state(random_state)

    if loss == 'gamma' or loss == 'rayleigh' or loss == 'poisson_count' or loss == 'bernoulli_odds':
        non_negative = True
    else:
        non_negative = False

    # initial tensor
    _, factors = initialize_generalized_parafac(tensor, rank, init=init, non_negative=non_negative, random_state=rng)

    if loss is not None:
        loss = loss_operator_func(tensor, rank, loss=loss, batch_size=batch_size, mask=mask)

    vectorized_factors = vectorize_factors(factors)
    norm = tl.norm(tensor, 2)
    x0 = tl.copy(vectorized_factors)
    x0.requires_grad = True
    optimizer = Adam([x0], lr=lr, betas=(beta_1, beta_2))
    error = []
    for i in range(n_iter_max):
        optimizer.zero_grad()
        objective = loss(x0)
        objective.backward()
        optimizer.step()
        if non_negative:
            with torch.no_grad():
                x0.data = x0.data.clamp(min=0)
        error.append(objective.item() / norm)

    _, factors = vectorized_factors_to_tensor(vectorized_factors, tl.shape(tensor), rank, return_factors=True)

    cp_tensor = CPTensor((None, factors))
    if return_errors:
        return cp_tensor, error
    else:
        return cp_tensor


class Stochastic_GCP(DecompositionMixin):
    """ Stochastic Generalized PARAFAC decomposition by using ADAM optimization.
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that::

        tensor ~ [|weights; factors[0], ..., factors[-1] |].

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initialization.
        See `initialize_factors`.
    lr : float
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else.
    return_errors : bool, optional
        Activate return of iteration errors
    loss : {'gaussian', 'bernoulli_odds', 'bernoulli_logit', 'rayleigh', 'poisson_count', 'poisson_log', 'gamma'}

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
        * sparse_component : nD array of shape tensor.shape. Returns only if `sparsity` is not None.
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] Hong, D., Kolda, T. G., & Duersch, J. A. (2020).
           Generalized canonical polyadic tensor decomposition. SIAM Review, 62(1), 133-163.
    .. [2] Kolda, T. G., & Hong, D. (2020). Stochastic gradients for large-scale tensor decomposition.
           SIAM Journal on Mathematics of Data Science, 2(4), 1066-1095.
    """

    def __init__(self, rank, n_iter_max=100, init='random', loss='gaussian', batch_size=100, lr=0.01,
                 beta_1=0.9, beta_2=0.999, return_errors=False, random_state=None, mask=None):
        self.rank = rank
        self.n_iter_max = n_iter_max
        self.init = init
        self.batch_size = batch_size
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.return_errors = return_errors
        self.loss = loss
        self.random_state = random_state
        self.lr = lr
        self.mask = mask

    def fit_transform(self, tensor):
        """Decompose an input tensor

        Parameters
        ----------
        tensor : tensorly tensor
            input tensor to decompose

        Returns
        -------
        CPTensor
            decomposed tensor
        """
        cp_tensor, errors = stochastic_generalized_parafac(
            tensor,
            rank=self.rank,
            n_iter_max=self.n_iter_max,
            init=self.init,
            batch_size=self.batch_size,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            loss=self.loss,
            lr=self.lr,
            random_state=self.random_state,
            mask=self.mask,
            return_errors=self.return_errors
        )
        self.decomposition_ = cp_tensor
        self.errors_ = errors
        return self.decomposition_
