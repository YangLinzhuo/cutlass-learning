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
Methods for layout swizzling
"""

from .layout import LayoutBase, size, cosize


def shiftr(a, s):
    '''
    Integer bit right shift.
    '''
    return a >> s if s > 0 else shiftl(a, -s)


def shiftl(a, s):
    '''
    Integer bit left shift.
    '''
    return a << s if s > 0 else shiftr(a, -s)


class Swizzle:
    '''
    ## A generic Swizzle functor
    # 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
    #                               ^--^  Base is the number of least-sig bits to keep constant
    #                  ^-^       ^-^      Bits is the number of bits in the mask
    #                    ^---------^      Shift is the distance to shift the YYY mask
    #                                     (pos shifts YYY to the right, neg shifts YYY to the left)
    #
    # e.g. Given
    # 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
    # the result is
    # 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
    #
    '''
    def __init__(self, bits, base, shift):
        assert bits >= 0
        assert base >= 0
        assert abs(shift) >= bits
        self.bits = bits
        self.base = base
        self.shift = shift
        bit_msk = (1 << bits) - 1
        self.yyy_msk = bit_msk << (base + max(0, shift))
        self.zzz_msk = bit_msk << (base - min(0, shift))

    # operator ()    (transform integer)
    def __call__(self, offset):
        return offset ^ shiftr(offset & self.yyy_msk, self.shift)

    def size(self):
        '''
        Size of the domain
        '''
        return 1 << (self.bits + self.base + abs(self.shift))

    def cosize(self):
        '''
        Size of the codomain
        '''
        return self.size()

    # print and str
    def __str__(self):
        return f"SW_{self.bits}_{self.base}_{self.shift}"

    # error msgs and representation
    def __repr__(self):
        return f"Swizzle({self.bits},{self.base},{self.shift})"


class ComposedLayout(LayoutBase):
    '''
    Composed Layout
    '''
    def __init__(self, layout_b, offset, layout_a):
        self.layout_b = layout_b
        self.offset = offset
        self.layout_a = layout_a

    # operator ==
    def __eq__(self, other):
        return (
            self.layout_b == other.layoutB
            and self.offset == other.offset
            and self.layout_a == other.layoutA
        )

    # operator len(L)  (len [rank] like tuples)
    def __len__(self):
        return len(self.layout_a)

    # operator ()    (map coord to idx)
    def __call__(self, *args):
        return self.layout_b(self.offset + self.layout_a(*args))

    # operator []    (get-i like tuples)
    def __getitem__(self, i):
        return ComposedLayout(self.layout_b, self.offset, self.layout_a[i])

    def size(self):
        '''
        size(layout)   Size of the domain
        '''
        return size(self.layout_a)

    def cosize(self):
        '''
        cosize(layout)   Size of the codomain
        '''
        return cosize(self.layout_b)

    # print and str
    def __str__(self):
        return f"{self.layout_b} o {self.offset} o {self.layout_a}"

    # error msgs and representation
    def __repr__(self):
        return f"ComposedLayout({repr(self.layout_b)},{repr(self.offset)},{repr(self.layout_a)})"
