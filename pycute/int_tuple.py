#################################################################################################
#
# Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Functions for manipulating IntTuples
"""

from abc import ABC
from functools import reduce
from itertools import chain
from typing import Union


class Integer(ABC):
    '''
    Interger class for type checking.
    '''
    @classmethod
    def __subclasshook__(cls, c):
        if c in [bool, float]:
            return False

        return issubclass(c, int)


def is_int(x) -> bool:
    '''
    Check whether variable is an integer.
    '''
    return isinstance(x, Integer)


def is_tuple(x) -> bool:
    '''
    Check whether variable is a tuple.
    '''
    if isinstance(x, tuple):
        return all(is_int(e) or is_tuple(e) for e in x)
    else:
        return False


def flatten(t) -> tuple[Integer]:
    '''
    Flatten a nested tuple.
    '''
    if is_tuple(t):
        if len(t) == 0:
            return ()
        else:
            return tuple(i for a in t for i in flatten(a))
    elif is_int(t):
        return (t,)
    else:
        raise TypeError(f"Unsupported type {type(t)}.")


def signum(a: Integer) -> Integer:
    '''
    Check the signal of integer.
    '''
    if is_int(a):
        return bool(a > 0) - bool(a < 0)
    else:
        raise TypeError(f"Unsupported type {type(a)}")


def product(a) -> Integer:
    '''
    Return the product of a tuple of Integer.
    '''
    if is_tuple(a):
        return reduce(lambda val, elem: val * product(elem), a, 1)
    elif is_int(a):
        return a
    else:
        raise TypeError(f"Unsupported type {type(a)}")


def inner_product(a, b) -> Integer:
    '''
    Return the inner product of the 2 tuples.
    '''
    if is_tuple(a) and is_tuple(b):
        assert len(a) == len(b), \
            f"The inner product operations requires two tuples of same length, " \
            f"but got a of {len(a)} and b of {len(b)}."
        return sum(inner_product(x, y) for x, y in zip(a, b))
    elif is_int(a) and is_int(b):  # "int"-"int"
        return a * b
    else:
        raise TypeError(f"Unsupported operation type: a{type(a)} - b{type(b)}")


def tuple_max(a) -> Integer:
    '''
    Return the mas element in a tuple.
    '''
    if is_tuple(a):
        return max(tuple_max(x) for x in a)
    elif is_int(a):
        return a
    else:
        raise TypeError(f"Unsupported type {type(a)}")


def elem_scale(a, b) -> tuple:
    '''
    Scale element in tuple a with corresponding element in tuple b.
    '''
    if is_tuple(a) and is_tuple(b):
        assert len(a) == len(b), \
            f"The element scale operations requires two tuples of same length, " \
            f"but got a of length {len(a)} and b of length {len(b)}."
        return tuple(elem_scale(x, y) for x, y in zip(a, b))
    elif is_int(a) and is_tuple(b):
        return elem_scale(a, product(b))
    elif is_int(a) and is_int(b):
        return a * b
    else:
        raise TypeError(f"Unsupported type for operation element scale: a {type(a)} - b {type(b)}")


def shape_div(a, b) -> tuple:
    '''
    Inclusive prefix ceil div with output congruent to input a.
    '''
    if is_tuple(a) and is_tuple(b):
        assert len(a) == len(b), \
            f"The shape_div requires two tuples of same length, " \
            f"but got a of length {len(a)} and b of length {len(b)}."
        return tuple(shape_div(x, y) for x, y in zip(a, b))
    elif is_tuple(a) and is_int(b):
        # r = [shape_div(a[0],b)] +
        # [shape_div(a[i],b := shape_div(b, product(a[i-1]))) for i in range(1,len(a))]
        r = []
        for v in a:
            r.append(shape_div(v, b))
            b = shape_div(b, product(v))
        return tuple(r)
    elif is_int(a) and is_tuple(b):
        return shape_div(a, product(b))
    elif is_int(a) and is_int(b):
        assert a % b == 0 or b % a == 0
        # return -(-a // b)      # Python exclusive impl: "//" is always floor div
        if a % b == 0:
            return a // b
        else:
            return signum(a * b)
    else:
        raise TypeError(f"Unsupported type for operation shape division: a {type(a)} - b {type(b)}")


def prefix_product(a, init=1) -> tuple:
    '''
    Exclusive prefix product with output congruent to input a.
    '''
    if is_tuple(a) and is_tuple(init):
        assert len(a) == len(init), \
            f"The prefix_product requires two tuples of same length, " \
            f"but got a of length {len(a)} and init of length {len(init)}."
        return tuple(prefix_product(x, i) for x, i in zip(a, init))
    elif is_tuple(a) and is_int(init):
        # r = [prefix_product(a[0],init)] +
        # [prefix_product(a[i],init := init * product(a[i-1])) for i in range(1,len(a))]
        r = []
        for v in a:
            r.append(prefix_product(v, init))
            init = init * product(v)
        return tuple(r)
    elif is_int(a) and is_int(init):
        return init
    else:
        raise TypeError(f"Unsupported type for operation prefix product: "
                        f"a {type(a)} - init {type(init)}")


def suffix_product(a, init=1) -> tuple:
    '''
    Exclusive suffix product with output congruent to input a.
    '''
    if is_tuple(a) and is_tuple(init):
        assert len(a) == len(init), \
            f"The suffix_product requires two tuples of same length, " \
            f"but got a of length {len(a)} and init of length {len(init)}."
        return tuple(prefix_product(x, i) for x, i in zip(a[::-1], init[::-1]))
    elif is_tuple(a) and is_int(init):
        # r = [prefix_product(a[0],init)] +
        # [prefix_product(a[i],init := init * product(a[i-1])) for i in range(1,len(a))]
        r = []
        for v in a[::-1]:
            r.append(prefix_product(v, init))
            init = init * product(v)
        return tuple(reversed(r))
    elif is_int(a) and is_int(init):
        return init
    else:
        raise TypeError(f"Unsupported type for operation suffix product: "
                        f"a {type(a)} - init {type(init)}")


def idx2crd(idx, shape, stride=None):
    '''
    Transform index to coordinate.
    '''
    if stride is None:
        stride = prefix_product(shape)

    if is_tuple(idx) and is_tuple(shape) and is_tuple(stride):
        assert len(idx) == len(shape) and len(idx) == len(stride), \
            f"The idx2crd operations requires three tuples of same length, " \
            f"but got idx of length {len(idx)} and shape of length {len(shape)} " \
            f"and stride of length {len(stride)}."
        return tuple(idx2crd(i, s, d) for i, s, d in zip(idx, shape, stride))
    elif is_int(idx) and is_tuple(shape) and is_tuple(stride):
        assert len(shape) == len(stride), \
            f"The idx2crd operations requires two tuples of same length, " \
            f"but got shape of length {len(shape)} " \
            f"and stride of length {len(stride)}."
        return tuple(idx2crd(idx, s, d) for s, d in zip(shape, stride))
    elif is_int(idx) and is_int(shape) and is_int(stride):  # "int" "int" "int"
        return (idx // stride) % shape
    else:
        raise TypeError(f"Unsupported type "
                        f"combinatioin({type(idx)}, {type(shape)}, {type(stride)})")


def crd2idx(crd, shape, stride=None):
    '''
    Transform coordinate to index.
    '''
    if stride is None:
        stride = prefix_product(shape)

    if is_tuple(crd) and is_tuple(shape) and is_tuple(stride):
        assert len(crd) == len(shape) and len(crd) == len(stride), \
            f"The crd2idx operations requires three tuples of same length, " \
            f"but got crd of length {len(crd)} and shape of length {len(shape)} " \
            f"and stride of length {len(stride)}."
        return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
    else:
        if crd is None:
            crd = 0

        if is_int(crd) and is_tuple(shape) and is_tuple(stride):  # "int" tuple tuple
            assert len(shape) == len(stride), \
                f"The crd2idx operations requires two tuples of same length, " \
                f"but got shape of length {len(shape)} " \
                f"and stride of length {len(stride)}."
            result = 0
            for i in range(len(shape) - 1):
                result += crd2idx(crd % product(shape[i]), shape[i], stride[i])
                crd = crd // product(shape[i])
            return result + crd2idx(crd, shape[-1], stride[-1])
        elif is_int(crd) and is_int(shape) and is_int(stride):  # "int" "int" "int"
            return crd * stride
        else:
            raise TypeError(f"Unsupported type "
                            f"combinatioin({type(crd)}, {type(shape)}, {type(stride)})")


def crd2crd(crd, dst_shape, src_shape=None):
    '''
    Transform crd into the dst_shape's iteration space.
    '''
    if is_tuple(crd) and is_tuple(dst_shape):
        assert len(crd) == len(dst_shape), \
                f"The crd2idx operations requires two tuples of same length, " \
                f"but got crd of length {len(crd)} " \
                f"and dst_shape of length {len(dst_shape)}."
        return tuple(crd2crd(x, y) for x, y in zip(crd, dst_shape))
    elif is_tuple(crd) and is_int(dst_shape):
        # Ambiguous unless we have src_shape
        assert src_shape is not None, "Ambiguous transform unless src_shape is not None."
        return crd2idx(crd, src_shape)
    elif is_int(crd) and is_tuple(dst_shape):
        return idx2crd(crd, dst_shape)
    elif is_int(crd) and is_int(dst_shape):  # "int" "int"
        assert crd < dst_shape
        return crd
    else:
        raise TypeError(f"Unsupported type combinatioin({type(crd)}, {type(dst_shape)})")


def slice_(crd: Union[None, tuple, int], trg: Union[tuple, int]):
    '''
    Filter trg according to crd: keep only elements of trg that are paired with None
    '''
    if is_tuple(crd) and is_tuple(trg):
        assert len(crd) == len(trg), \
                f"The slice_ operations requires two tuples of same length, " \
                f"but got crd of length {len(crd)} " \
                f"and trg of length {len(trg)}."
        # match C++ behavior of `filter_tuple` using `tuple_cat(...)`
        return tuple(
            chain(
                *filter(lambda x: x != (), [slice_(c, s) for c, s in zip(crd, trg)])
            )
        )
    elif crd is None:
        # match C++ behavior `return cute::tuple<B>{b};`
        return (trg,)
    else:
        return ()


def has_none(a: Union[None, tuple, int]):
    '''
    Determine if None appears at any of an int_tuples' terminals.
    '''
    if is_tuple(a):
        return any(has_none(v) for v in a)
    else:
        return a is None
