#implementing a standard NN in a "nice", object oriented way to kinda learn that or smth
from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
import pickle
import matplotlib.pyplot as plt
import numba
from skimage.util.shape import view_as_windows
import types 

#We generally define a neural network
class neural_network(ABC):
    @abstractmethod
    def forward_propagation(self):
        raise NotImplementedError
    @abstractmethod
    def backward_propagation(self):
        raise NotImplementedError

class Computational_Graph(neural_network):
    def __init__(self) -> None:
        super().__init__()

class tensor():
    def __init__(self) -> None:
        pass
    
#Actual current project
class Node:
    """
    A node in our computational graph by an operation of two further notes
    
    value: Value of the forward pass of this node
    fun: Function that the node was evaluated by
    parents: List of other Node objects to which the function was applied
    has_gradient: If the function requieres a gradient w.r.t. some inherent parameter
    is_leaf: If the tensor is either created independently or a result of has_gradient = False operation
    grad_fun: Reference to the function that calculates the gradient for the fun - Inputs from the fun - Output gradient
    """
    def __init__(self, value, fun, parents, has_gradient = False, is_leaf = True, grad_fun = None) -> None:
        self.parents = parents
        self.value = value
        self.fun = fun
        self.has_gradient = has_gradient
    """
    Computes the gradient of the current node w.r.t. graph leafs during a backwards-pass
    Procedure:
    1. Take incoming gradient as input
    2. Compute local gradient at particular node
    3. Multiply local gradient with incoming gradient
    4. Forward the computed gradient to the parent nodes by calling grad_fun 
    """
    
    def backwards(self, grad_vec):
        self.jacob = None
        if self.has_gradient:
            self.grad = self.jacob @ self.grad_vec

def start_node(value = None):
    return Node(value, lambda x: x, [])

#Code taken from: https://github.com/puzzler10/autograd_simpler/blob/master/numpy_autograd/np_wrapping.py
def primitive(f, has_gradient = True):
    @wraps(f)
    def inner(*args, **kwargs):
        # Add to graph, where we have to differentiate node and primitive parents
        def getval(o):
            if type(o) == Node:
                return o.value
            else:
                return o
        if len(args): 
            argvals = [getval(o) for o in args]
        else:
            argvals = args
        if len(kwargs):
            kwargvals = dict([(k, getval(o)) for k,o in kwargs.items()])
        else:
            kwargvals = kwargs
        
        value = f(*args, **kwargs)
        parents = [o for o in list(args) + list(kwargs.values()) if type(o) == Node]
        return Node(value, f, parents, has_gradient)
    return inner

def wrap_namespace(old, new):
    nograd_functions = [
        np.ndim, np.shape, np.iscomplexobj, np.result_type, np.zeros_like,
        np.ones_like, np.floor, np.ceil, np.round, np.rint, np.around,
        np.fix, np.trunc, np.all, np.any, np.argmax, np.argmin,
        np.argpartition, np.argsort, np.argwhere, np.nonzero, np.flatnonzero,
        np.count_nonzero, np.searchsorted, np.sign, np.ndim, np.shape,
        np.floor_divide, np.logical_and, np.logical_or, np.logical_not,
        np.logical_xor, np.isfinite, np.isinf, np.isnan, np.isneginf,
        np.isposinf, np.allclose, np.isclose, np.array_equal, np.array_equiv,
        np.greater, np.greater_equal, np.less, np.less_equal, np.equal,
        np.not_equal, np.iscomplexobj, np.iscomplex, np.size, np.isscalar,
        np.isreal, np.zeros_like, np.ones_like, np.result_type
    ]
    function_types = {np.ufunc, types.FunctionType, types.BuiltinFunctionType}

    for name,obj in old.items(): 
        if obj in nograd_functions:  
            # non-differentiable functions 
            new[name] = primitive(obj, keepgrad=False)
        elif type(obj) in function_types:  # functions with gradients 
            # differentiable functions
            new[name] = primitive(obj)
        else: 
            # just copy over 
            new[name] = obj

anp = globals()
wrap_namespace(np.__dict__, anp)

## Definitions taken from here:  
## https://github.com/mattjj/autodidact/blob/b3b6e0c16863e6c7750b0fc067076c51f34fe271/autograd/numpy/numpy_boxes.py#L8
setattr(Node, 'ndim', property(lambda self: self.value.ndim))
setattr(Node, 'size', property(lambda self: self.value.size))
setattr(Node, 'dtype',property(lambda self: self.value.dtype))
setattr(Node, 'T', property(lambda self: anp['transpose'](self)))
setattr(Node, 'shape', property(lambda self: self.value.shape))
setattr(Node,'__len__', lambda self, other: len(self._value))
setattr(Node,'astype', lambda self,*args,**kwargs: anp['_astype'](self, *args, **kwargs))
setattr(Node,'__neg__', lambda self: anp['negative'](self))
setattr(Node,'__add__', lambda self, other: anp['add'](     self, other))
setattr(Node,'__sub__', lambda self, other: anp['subtract'](self, other))
setattr(Node,'__mul__', lambda self, other: anp['multiply'](self, other))
setattr(Node,'__pow__', lambda self, other: anp['power'](self, other))
setattr(Node,'__div__', lambda self, other: anp['divide'](  self, other))
setattr(Node,'__mod__', lambda self, other: anp['mod'](     self, other))
setattr(Node,'__truediv__', lambda self, other: anp['true_divide'](self, other))
setattr(Node,'__matmul__', lambda self, other: anp['matmul'](self, other))
setattr(Node,'__radd__', lambda self, other: anp['add'](     other, self))
setattr(Node,'__rsub__', lambda self, other: anp['subtract'](other, self))
setattr(Node,'__rmul__', lambda self, other: anp['multiply'](other, self))
setattr(Node,'__rpow__', lambda self, other: anp['power'](   other, self))
setattr(Node,'__rdiv__', lambda self, other: anp['divide'](  other, self))
setattr(Node,'__rmod__', lambda self, other: anp['mod'](     other, self))
setattr(Node,'__rtruediv__', lambda self, other: anp['true_divide'](other, self))
setattr(Node,'__rmatmul__', lambda self, other: anp['matmul'](other, self))
setattr(Node,'__eq__', lambda self, other: anp['equal'](self, other))
setattr(Node,'__ne__', lambda self, other: anp['not_equal'](self, other))
setattr(Node,'__gt__', lambda self, other: anp['greater'](self, other))
setattr(Node,'__ge__', lambda self, other: anp['greater_equal'](self, other))
setattr(Node,'__lt__', lambda self, other: anp['less'](self, other))
setattr(Node,'__le__', lambda self, other: anp['less_equal'](self, other))
setattr(Node,'__abs__', lambda self: anp['abs'](self))
setattr(Node,'__hash__', lambda self: id(self))