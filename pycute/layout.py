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
Definition of CuTe Layouts and functions to manipulate them
"""

from enum import Enum
from itertools import chain
from .int_tuple import (prefix_product, suffix_product, is_tuple, is_int, has_none, slice_,
                        crd2idx, flatten, product, shape_div)


class LayoutBase:
    '''
    LayoutBase class
    '''


def is_layout(x):
    '''
    Check whether variable is Layout
    '''
    return isinstance(x, LayoutBase)


class StrideType(Enum):
    '''
    Stride type.
    '''
    ROW_MAJOR = 1
    COL_MAJOR = 2


class Layout(LayoutBase):
    '''
    Layout which describes the mapping of tensor logical address to physical address
    '''
    def __init__(self, _shape, _stride=None):
        if is_int(_shape) or is_tuple(_shape):
            self.shape = _shape
        else:
            raise TypeError(f"Unsupported type of shape {type(_shape)}")

        if _stride is None:
            self.stride = prefix_product(self.shape)
        elif _stride == StrideType.COL_MAJOR:
            self.stride = prefix_product(self.shape)
        elif _stride == StrideType.ROW_MAJOR:
            self.stride = suffix_product(self.shape)
        elif is_tuple(_stride) or is_int(_stride):
            self.stride = _stride
        else:
            raise TypeError(f"Unsupported type of stride {type(_stride)}")

    # operator ==
    def __eq__(self, other):
        return self.shape == other.shape and self.stride == other.stride

    # operator len(L)  (len [rank] like tuples)
    def __len__(self):
        if is_tuple(self.shape):
            return len(self.shape)
        else:
            return 1

    # operator ()    (map coord to idx)
    def __call__(self, *args):
        """
        Map a logical coordinate to a linear index (Coord has no Underscore slice operators)
        OR
        Slice the layout and return the sublayout (Coord has an Underscore slice op)

        Follow the same behavior of `Layout::operator(Coord const&)` in cute C++
        """
        if has_none(args):
            if len(args) == 1:
                return Layout(slice_(args[0], self.shape), slice_(args[0], self.stride))
            else:
                return Layout(slice_(args, self.shape), slice_(args, self.stride))
        else:
            if len(args) == 1:
                return crd2idx(args[0], self.shape, self.stride)
            else:
                return crd2idx(args, self.shape, self.stride)

    # operator []    (get-i like tuples)
    def __getitem__(self, i):
        if is_tuple(self.shape):
            return Layout(self.shape[i], self.stride[i])
        else:
            assert i == 0
            return Layout(self.shape, self.stride)

    def size(self):
        '''
        size(layout)   Size of the domain
        '''
        return product(self.shape)

    def cosize(self):
        '''
        cosize(layout)   Size of the codomain
        '''
        return self(self.size() - 1) + 1

    # print and str
    def __str__(self):
        return f"{self.shape}:{self.stride}"

    # error msgs and representation
    def __repr__(self):
        return f"Layout({self.shape},{self.stride})"


def make_layout(*layouts):
    '''
    Make Layout from a list of layouts (each layout it's own mode in the result)
    '''
    if len(layouts) == 1 and not is_layout(layouts[0]):
        layouts = layouts[0]

    shape, stride = zip(*((a.shape, a.stride) for a in layouts))
    return Layout(shape, stride)


def size(layout):
    '''
    Size of the domain
    '''
    if is_layout(layout):
        return layout.size()
    return product(layout)


def cosize(layout):
    '''
    Size of the codomain
    '''
    return layout.cosize()


def coalesce(layout, profile=None):
    '''
    # Layout coalesce -- flatten and combine as many modes as possible 
    #                    while preserving the int-to-int function
    '''
    if is_tuple(profile):
        assert len(layout) >= len(profile)
        return make_layout(
            chain(
                (coalesce(layout[i], profile[i]) for i in range(0, len(profile))),
                (layout[i] for i in range(len(profile), len(layout))),
            )
        )

    result_shape = [1]
    result_stride = [0]
    for shape, stride in zip(flatten(layout.shape), flatten(layout.stride)):
        # skip their shape-1s
        if shape == 1:
            continue
        # replace our shape-1 with anything
        elif result_shape[-1] == 1:
            result_shape[-1] = shape
            result_stride[-1] = stride
        # merge modes if the shape*stride match
        elif result_shape[-1] * result_stride[-1] == stride:
            result_shape[-1] = result_shape[-1] * shape
        # append a new mode
        else:
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 1:
        return Layout(result_shape[0], result_stride[0])
    else:
        return Layout(tuple(result_shape), tuple(result_stride))


def filter_(layout, profile=None):
    '''
    Layout filter -- replace all stride-0 modes with size-1 and then coalesce to remove them
    '''
    if is_tuple(profile):
        assert len(layout) >= len(profile)
        return make_layout(
            chain(
                (filter_(layout[i], profile[i]) for i in range(0, len(profile))),
                (layout[i] for i in range(len(profile), len(layout))),
            )
        )

    result_shape = []
    result_stride = []
    for shape, stride in zip(flatten(layout.shape), flatten(layout.stride)):
        # skip their shape-1s and stride-0s
        if not (shape == 1 or stride == 0):
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 0:
        return Layout(1, 0)
    else:
        return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


def composition(layout_a, layout_b):
    '''
    Layout composition
    Use tuples-of-layouts to perform this operation by-mode and None as no-op
    '''
    if layout_b is None:
        return layout_a
    elif is_int(layout_b):
        return composition(layout_a, Layout(layout_b))
    elif is_tuple(layout_b):
        assert len(layout_a) >= len(layout_b)
        return make_layout(
            chain(
                (composition(layout_a[i], layout_b[i]) for i in range(0, len(layout_b))),
                (layout_a[i] for i in range(len(layout_b), len(layout_a))),
            )
        )
    elif is_tuple(layout_b.shape):
        return make_layout(composition(layout_a, layoutB_i) for layoutB_i in layout_b)

    if layout_b.stride == 0:
        return Layout(layout_b.shape, 0)
    else:
        result_shape = []
        result_stride = []
        rest_shape = layout_b.shape
        rest_stride = layout_b.stride
        for s, d in zip(flatten(layout_a.shape)[:-1], flatten(layout_a.stride)[:-1]):
            s1 = shape_div(s, rest_stride)
            result_shape.append(min(s1, rest_shape))
            result_stride.append(rest_stride * d)
            rest_shape = shape_div(rest_shape, abs(s1))
            rest_stride = shape_div(rest_stride, s)

        result_shape.append(rest_shape)
        result_stride.append(rest_stride * flatten(layout_a.stride)[-1])

        return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


def complement(layout, max_idx=1):
    '''
    Layout complement
    '''
    if is_int(layout):
        return complement(Layout(layout))

    result_shape = []
    result_stride = []
    current_idx = 1

    sorted_ds = sorted(zip(flatten(layout.stride), flatten(layout.shape)))
    for stride, shape in sorted_ds:
        if stride == 0 or shape == 1:
            continue

        in_bound = current_idx <= shape * stride
        # To support symbolic value which can't be evaluated now
        assert not isinstance(in_bound, bool) or in_bound

        result_shape.append(stride // current_idx)
        result_stride.append(current_idx)
        current_idx = shape * stride

    result_shape.append((max_idx + current_idx - 1) // current_idx)  # ceil_div
    result_stride.append(current_idx)

    return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


def right_inverse(layout):
    '''
    Layout right inverse
    '''
    if layout is None:
        return None
    elif is_int(layout):
        return Layout(layout)

    result_shape = []
    result_stride = []
    current_idx = 1

    flat_shape = flatten(layout.shape)
    flat_stride = flatten(layout.stride)
    sorted_dsa = sorted(zip(flat_stride, flat_shape, prefix_product(flat_shape)))
    for stride, shape, rstride in sorted_dsa:
        if shape == 1:
            continue
        if current_idx != stride:
            break

        result_shape.append(shape)
        result_stride.append(rstride)
        current_idx = shape * stride

    return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


def left_inverse(layout):
    '''
    Layout left inverse
    '''
    if layout is None:
        return None
    elif is_int(layout):
        return Layout(layout)
    return right_inverse(make_layout(layout, complement(layout)))


def logical_divide(layout_a, layout_b):
    '''
    Split a layout by the composition of B and the "rest"
    Use tuples-of-layouts to perform this operation by-mode and None as no-op
    '''
    if layout_b is None:
        return layout_a
    elif is_int(layout_b):
        return logical_divide(layout_a, Layout(layout_b))
    elif is_tuple(layout_b):
        assert len(layout_a) >= len(layout_b)
        return make_layout(
            chain(
                (
                    logical_divide(layout_a[i], layout_b[i])
                    for i in range(0, len(layout_b))
                ),
                (layout_a[i] for i in range(len(layout_b), len(layout_a))),
            )
        )

    return composition(
        layout_a, make_layout(layout_b, complement(layout_b, size(layout_a)))
    )


def logical_product(layout_a, layout_b):
    '''
    Reproduce a layout_a over a layout_b
    Use tuples-of-layouts to perform this operation by-mode and None as no-op
    '''
    if layout_b is None:
        return layout_a
    elif is_int(layout_b):
        return logical_divide(layout_a, Layout(layout_b))
    elif is_tuple(layout_b):
        assert len(layout_a) >= len(layout_b)
        return make_layout(
            chain(
                (
                    logical_product(layout_a[i], layout_b[i])
                    for i in range(0, len(layout_b))
                ),
                (layout_a[i] for i in range(len(layout_b), len(layout_a))),
            )
        )

    return make_layout(
        layout_a,
        composition(complement(layout_a, size(layout_a) * cosize(layout_b)), layout_b),
    )


def hier_unzip(splitter, layout_a, layout_b):
    '''
    Gather the modes from a hierarchical logical_divide or logical_product
    '''
    if layout_b is None:
        return make_layout(Layout(1, 0), layout_a)
    elif is_tuple(layout_b):
        assert len(layout_a) >= len(layout_b)
        # A layout with shape ((A,a),(B,b),(C,c))
        split = make_layout(
            hier_unzip(splitter, layout_a[i], layout_b[i]) for i in range(0, len(layout_b))
        )
        # Gather to shape ((A,B,C,...),(a,b,c,...,y,z))
        return make_layout(
            make_layout(split[i][0] for i in range(0, len(layout_b))),
            make_layout(
                chain(
                    (split[i][1] for i in range(0, len(layout_b))),
                    (layout_a[i] for i in range(len(layout_b), len(layout_a))),
                )
            ),
        )

    # splitter must return a rank-2 layout
    return splitter(layout_a, layout_b)


def zipped_divide(layout_a, layout_b):
    '''
    Apply logical divide hierarchically and gather the split modes into two modes
    '''
    return hier_unzip(logical_divide, layout_a, layout_b)


def tiled_divide(layout_a, layout_b):
    '''
    Perform logical divide hierarchically and gather tiles (B-layouts) into a new mode
    '''
    result = zipped_divide(layout_a, layout_b)
    return make_layout([result[0]] + [result[1][i] for i in range(len(result[1]))])


def zipped_product(layout_a, layout_b):
    '''
    Apply logical product hierarchically and gather the split modes into two modes
    '''
    return hier_unzip(logical_product, layout_a, layout_b)


def tiled_product(layout_a, layout_b):
    '''
    Perform logical product hierarchically and gather tiles (B-layouts) into a new mode
    '''
    result = zipped_product(layout_a, layout_b)
    return make_layout([result[0]] + [result[1][i] for i in range(len(result[1]))])


def slice_and_offset(crd: tuple, layout: Layout):
    '''
    Perform Layout slice and compute corresponding offset
    '''
    return (
        Layout(slice_(crd, layout.shape), slice_(crd, layout.stride)),
        crd2idx(crd, layout.shape, layout.stride),
    )
