"""Second-order streaming moment calculator translated from sOSM.cpp.

This is a direct Python translation of the original branching and formula logic.

Conventions:
    phase = 0 for cosine, 1 for sine.
    dim1, dim2 = 0, 1, or 2 for the two spatial direction indices.
"""

from math import sqrt
import numpy as np


class sOSM:
    """Compute second-order real-spherical-harmonic streaming moments."""

    def __init__(
        self,
        dim1: int = 0,
        dim2: int = 0,
        l1: int = 0,
        l2: int = 0,
        m1: int = 0,
        m2: int = 0,
        phase1: int = 0,
        phase2: int = 0,
    ) -> None:
        self.Set(dim1, dim2, l1, l2, m1, m2, phase1, phase2)

    def Set(
        self,
        dim1: int,
        dim2: int,
        l1: int,
        l2: int,
        m1: int,
        m2: int,
        phase1: int,
        phase2: int,
    ) -> None:
        """Set the moment indices, preserving the C++ argument order."""
        self.dim1 = dim1
        self.dim2 = dim2
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.phase1 = phase1
        self.phase2 = phase2

    def GetMom(self) -> float:
        """Return the second-order streaming moment."""
        eps = 1.0
        lr = float(self.l1)
        mr = float(self.m1)
        r = 0.0
        if self.phase1 == 0:
            if self.phase2 == 0:
                if self.dim1 == 0:
                    if self.dim2 == 0:
                        if self.l2 == self.l1 - 2:
                            if self.m2 == self.m1 - 2:
                                if self.m1 == 0:
                                    eps = 1. / sqrt(2.)
                                elif self.m1 == 2:
                                    eps = sqrt(2.)
                                r = eps * sqrt((lr + mr - 3) * (lr + mr - 2) * (lr + mr - 1) * (lr + mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                            elif self.m2 == self.m1:
                                r = -sqrt((lr + mr - 1) * (lr + mr) * (lr - mr - 1) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                                if self.m1 == 1:
                                    r = 1.5 * r
                            elif self.m2 == self.m1 + 2:
                                if self.m1 == 0:
                                    eps = sqrt(2.)
                                r = eps * sqrt((lr - mr - 3) * (lr - mr - 2) * (lr - mr - 1) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                        elif self.l2 == self.l1:
                            if self.m2 == self.m1 - 2:
                                if self.m1 == 0:
                                    eps = 1. / sqrt(2.)
                                elif self.m1 == 2:
                                    eps = sqrt(2.)
                                r = -eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr + mr - 1) * (lr + mr)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                            elif self.m2 == self.m1:
                                r = (pow(lr, 2) + lr - 1 + pow(mr, 2)) / ((2 * lr - 1) * (2 * lr + 3))
                                if self.m1 == 1:
                                    r = 1.5 * r
                            elif self.m2 == self.m1 + 2:
                                if self.m1 == 0:
                                    eps = sqrt(2.)
                                r = -eps * sqrt((lr - mr - 1) * (lr - mr) * (lr + mr + 1) * (lr + mr + 2)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                        elif self.l2 == self.l1 + 2:
                            if self.m2 == self.m1 - 2:
                                if self.m1 == 0:
                                    eps = 1. / sqrt(2.)
                                elif self.m1 == 2:
                                    eps = sqrt(2.)
                                r = eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) * (lr - mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                            elif self.m2 == self.m1:
                                r = -sqrt((lr + mr + 1) * (lr + mr + 2) * (lr - mr + 1) * (lr - mr + 2) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                                if self.m1 == 1:
                                    r = 1.5 * r
                            elif self.m2 == self.m1 + 2:
                                if self.m1 == 0:
                                    eps = sqrt(2.)
                                r = eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr + mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                    elif self.dim2 == 2:
                        if self.l2 == self.l1 - 2:
                            if self.m2 == self.m1 - 1:
                                if self.m1 == 0:
                                    eps = 1. / sqrt(2.)
                                elif self.m1 == 1:
                                    eps = sqrt(2.)
                                r = -eps * sqrt((lr + mr - 2) * (lr + mr - 1) * (lr + mr) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                            elif self.m2 == self.m1 + 1:
                                if self.m1 == 0:
                                    eps = sqrt(2.)
                                r = eps * sqrt((lr + mr) * (lr - mr - 2) * (lr - mr - 1) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                        elif self.l2 == self.l1:
                            if self.m2 == self.m1 - 1:
                                if self.m1 == 0:
                                    eps = 1. / sqrt(2.)
                                elif self.m1 == 1:
                                    eps = sqrt(2.)
                                r = eps * (-sqrt((lr + mr) * (lr - mr + 1)) * (lr + mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) + sqrt((lr + mr) * (lr - mr + 1)) * (lr - mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                            elif self.m2 == self.m1 + 1:
                                if self.m1 == 0:
                                    eps = sqrt(2.)
                                r = eps * (sqrt((lr + mr + 1) * (lr - mr)) * (lr - mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) - sqrt((lr + mr + 1) * (lr - mr)) * (lr + mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                        elif self.l2 == self.l1 + 2:
                            if self.m2 == self.m1 - 1:
                                if self.m1 == 0:
                                    eps = 1. / sqrt(2.)
                                elif self.m1 == 1:
                                    eps = sqrt(2.)
                                r = eps * sqrt((lr + mr + 1) * (lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                            elif self.m2 == self.m1 + 1:
                                if self.m1 == 0:
                                    eps = sqrt(2.)
                                r = -eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr - mr + 1) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                elif self.dim1 == 1:
                    if self.dim2 == 1:
                        if self.l2 == self.l1 - 2:
                            if self.m2 == self.m1 - 2:
                                if self.m1 == 0:
                                    eps = 1. / sqrt(2.)
                                elif self.m1 == 2:
                                    eps = sqrt(2.)
                                r = -eps * sqrt((lr + mr - 3) * (lr + mr - 2) * (lr + mr - 1) * (lr + mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                            elif self.m2 == self.m1:
                                r = -sqrt((lr - mr - 1) * (lr - mr) * (lr + mr - 1) * (lr + mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                                if self.m1 == 1:
                                    r = 0.5 * r
                            elif self.m2 == self.m1 + 2:
                                if self.m1 == 0:
                                    eps = sqrt(2.)
                                r = -eps * sqrt((lr - mr) * (lr - mr - 1) * (lr - mr - 2) * (lr - mr - 3) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                        elif self.l2 == self.l1:
                            if self.m2 == self.m1 - 2:
                                if self.m1 == 0:
                                    eps = 1. / sqrt(2.)
                                elif self.m1 == 2:
                                    eps = sqrt(2.)
                                r = eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr + mr - 1) * (lr + mr)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                            elif self.m2 == self.m1:
                                r = (pow(lr, 2) + lr - 1 + pow(mr, 2)) / ((2 * lr - 1) * (2 * lr + 3))
                                if self.m1 == 1:
                                    r = 0.5 * r
                            elif self.m2 == self.m1 + 2:
                                if self.m1 == 0:
                                    eps = sqrt(2.)
                                r = eps * sqrt((lr - mr - 1) * (lr - mr) * (lr + mr + 1) * (lr + mr + 2)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                        elif self.l2 == self.l1 + 2:
                            if self.m2 == self.m1 - 2:
                                if self.m1 == 0:
                                    eps = 1. / sqrt(2.)
                                elif self.m1 == 2:
                                    eps = sqrt(2.)
                                r = -eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) * (lr - mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                            elif self.m2 == self.m1:
                                r = -sqrt((lr + mr + 1) * (lr + mr + 2) * (lr - mr + 1) * (lr - mr + 2) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                                if self.m1 == 1:
                                    r = 0.5 * r
                            elif self.m2 == self.m1 + 2:
                                if self.m1 == 0:
                                    eps = sqrt(2.)
                                r = -eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr + mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                elif self.dim1 == 2:
                    if self.dim2 == 0:
                        if self.l2 == self.l1 - 2:
                            if self.m2 == self.m1 - 1:
                                if self.m1 == 0:
                                    eps = 1. / sqrt(2.)
                                elif self.m1 == 1:
                                    eps = sqrt(2.)
                                r = -eps * sqrt((lr + mr - 2) * (lr + mr - 1) * (lr + mr) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                            elif self.m2 == self.m1 + 1:
                                if self.m1 == 0:
                                    eps = sqrt(2.)
                                r = eps * sqrt((lr + mr) * (lr - mr - 2) * (lr - mr - 1) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                        elif self.l2 == self.l1:
                            if self.m2 == self.m1 - 1:
                                if self.m1 == 0:
                                    eps = 1. / sqrt(2.)
                                elif self.m1 == 1:
                                    eps = sqrt(2.)
                                r = eps * (-sqrt((lr + mr) * (lr - mr + 1)) * (lr + mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) + sqrt((lr + mr) * (lr - mr + 1)) * (lr - mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                            elif self.m2 == self.m1 + 1:
                                if self.m1 == 0:
                                    eps = sqrt(2.)
                                r = eps * (sqrt((lr + mr + 1) * (lr - mr)) * (lr - mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) - sqrt((lr + mr + 1) * (lr - mr)) * (lr + mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                        elif self.l2 == self.l1 + 2:
                            if self.m2 == self.m1 - 1:
                                if self.m1 == 0:
                                    eps = 1. / sqrt(2.)
                                elif self.m1 == 1:
                                    eps = sqrt(2.)
                                r = eps * sqrt((lr + mr + 1) * (lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                            elif self.m2 == self.m1 + 1:
                                if self.m1 == 0:
                                    eps = sqrt(2.)
                                r = -eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr - mr + 1) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                    elif self.dim2 == 2:
                        if self.l2 == self.l1 - 2:
                            if self.m2 == self.m1:
                                r = sqrt((lr - mr - 1) * (lr - mr) * (lr + mr - 1) * (lr + mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * lr - 1)
                        elif self.l2 == self.l1:
                            if self.m2 == self.m1:
                                r = (2 * pow(lr, 2) + 2 * lr - 1 - 2 * pow(mr, 2)) / ((2 * lr - 1) * (2 * lr + 3))
                        elif self.l2 == self.l1 + 2:
                            if self.m2 == self.m1:
                                r = sqrt((lr - mr + 1) * (lr - mr + 2) * (lr + mr + 1) * (lr + mr + 2) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * lr + 3)
            elif self.phase2 == 1:
                if self.m2 != 0:
                    if self.dim1 == 0:
                        if self.dim2 == 1:
                            if self.l2 == self.l1 - 2:
                                if self.m2 == self.m1 - 2:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 2:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr + mr - 3) * (lr + mr - 2) * (lr + mr - 1) * (lr + mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                                elif self.m2 == self.m1:
                                    if self.m1 == 1:
                                        r = -sqrt((lr - mr) * (lr - mr - 1) * (lr + mr) * (lr + mr - 1) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                                elif self.m2 == self.m1 + 2:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr - mr - 3) * (lr - mr - 2) * (lr - mr - 1) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                            elif self.l2 == self.l1:
                                if self.m2 == self.m1 - 2:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 2:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr + mr - 1) * (lr + mr)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                                elif self.m2 == self.m1:
                                    if self.m1 == 1:
                                        r = ((lr - mr) * (lr - mr - 1) / ((2 * lr - 1) * (2 * lr + 1)) + (lr + mr + 1) * (lr + mr + 2) / ((2 * lr + 1) * (2 * lr + 3))) / 4.
                                elif self.m2 == self.m1 + 2:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr - mr - 1) * (lr - mr) * (lr + mr + 1) * (lr + mr + 2)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                            elif self.l2 == self.l1 + 2:
                                if self.m2 == self.m1 - 2:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 2:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) * (lr - mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                                elif self.m2 == self.m1:
                                    if self.m1 == 1:
                                        r = -sqrt((lr + mr + 1) * (lr + mr + 2) * (lr - mr + 1) * (lr - mr + 2) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                                elif self.m2 == self.m1 + 2:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr + mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                    elif self.dim1 == 1:
                        if self.dim2 == 0:
                            if self.l2 == self.l1 - 2:
                                if self.m2 == self.m1 - 2:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 2:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr + mr - 3) * (lr + mr - 2) * (lr + mr - 1) * (lr + mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                                elif self.m2 == self.m1:
                                    if self.m1 == 1:
                                        r = -sqrt((lr - mr) * (lr - mr - 1) * (lr + mr) * (lr + mr - 1) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                                elif self.m2 == self.m1 + 2:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr - mr - 3) * (lr - mr - 2) * (lr - mr - 1) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                            elif self.l2 == self.l1:
                                if self.m2 == self.m1 - 2:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 2:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr + mr - 1) * (lr + mr)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                                elif self.m2 == self.m1:
                                    if self.m1 == 1:
                                        r = ((lr - mr) * (lr - mr - 1) / ((2 * lr - 1) * (2 * lr + 1)) + (lr + mr + 1) * (lr + mr + 2) / ((2 * lr + 1) * (2 * lr + 3))) / 4.
                                elif self.m2 == self.m1 + 2:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr - mr - 1) * (lr - mr) * (lr + mr + 1) * (lr + mr + 2)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                            elif self.l2 == self.l1 + 2:
                                if self.m2 == self.m1 - 2:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 2:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) * (lr - mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                                elif self.m2 == self.m1:
                                    if self.m1 == 1:
                                        r = -sqrt((lr + mr + 1) * (lr + mr + 2) * (lr - mr + 1) * (lr - mr + 2) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                                elif self.m2 == self.m1 + 2:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr + mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                        elif self.dim2 == 2:
                            if self.l2 == self.l1 - 2:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr + mr - 2) * (lr + mr - 1) * (lr + mr) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr + mr) * (lr - mr - 1) * (lr - mr) * (lr - mr - 2) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                            elif self.l2 == self.l1:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.)
                                    r = eps * (sqrt((lr - mr + 1) * (lr + mr)) * (lr + mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) - sqrt((lr - mr + 1) * (lr + mr)) * (lr - mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * (sqrt((lr - mr) * (lr + mr + 1)) * (lr - mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) - sqrt((lr - mr) * (lr + mr + 1)) * (lr + mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                            elif self.l2 == self.l1 + 2:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr + mr + 1) * (lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr - mr + 1) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                    elif self.dim1 == 2:
                        if self.dim2 == 1:
                            if self.l2 == self.l1 - 2:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr + mr - 2) * (lr + mr - 1) * (lr + mr) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr + mr) * (lr - mr - 1) * (lr - mr) * (lr - mr - 2) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                            elif self.l2 == self.l1:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.)
                                    r = eps * (sqrt((lr - mr + 1) * (lr + mr)) * (lr + mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) - sqrt((lr - mr + 1) * (lr + mr)) * (lr - mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * (sqrt((lr - mr) * (lr + mr + 1)) * (lr - mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) - sqrt((lr - mr) * (lr + mr + 1)) * (lr + mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                            elif self.l2 == self.l1 + 2:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr + mr + 1) * (lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr - mr + 1) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
        elif self.phase1 == 1:
            if self.m1 != 0:
                if self.phase2 == 0:
                    if self.dim1 == 0:
                        if self.dim2 == 1:
                            if self.l2 == self.l1 - 2:
                                if self.m2 == self.m1 - 2:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 2:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr + mr - 3) * (lr + mr - 2) * (lr + mr - 1) * (lr + mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                                elif self.m2 == self.m1:
                                    if self.m1 == 1:
                                        r = -sqrt((lr - mr) * (lr - mr - 1) * (lr + mr) * (lr + mr - 1) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                                elif self.m2 == self.m1 + 2:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr - mr - 3) * (lr - mr - 2) * (lr - mr - 1) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                            elif self.l2 == self.l1:
                                if self.m2 == self.m1 - 2:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 2:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr + mr - 1) * (lr + mr)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                                elif self.m2 == self.m1:
                                    if self.m1 == 1:
                                        r = ((lr - mr) * (lr - mr - 1) / ((2 * lr - 1) * (2 * lr + 1)) + (lr + mr + 1) * (lr + mr + 2) / ((2 * lr + 1) * (2 * lr + 3))) / 4.
                                elif self.m2 == self.m1 + 2:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr - mr - 1) * (lr - mr) * (lr + mr + 1) * (lr + mr + 2)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                            elif self.l2 == self.l1 + 2:
                                if self.m2 == self.m1 - 2:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 2:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) * (lr - mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                                elif self.m2 == self.m1:
                                    if self.m1 == 1:
                                        r = -sqrt((lr + mr + 1) * (lr + mr + 2) * (lr - mr + 1) * (lr - mr + 2) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                                elif self.m2 == self.m1 + 2:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = - eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr + mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                    elif self.dim1 == 1:
                        if self.dim2 == 0:
                            if self.l2 == self.l1 - 2:
                                if self.m2 == self.m1 - 2:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 2:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr + mr - 3) * (lr + mr - 2) * (lr + mr - 1) * (lr + mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                                elif self.m2 == self.m1:
                                    if self.m1 == 1:
                                        r = -sqrt((lr - mr) * (lr - mr - 1) * (lr + mr) * (lr + mr - 1) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                                elif self.m2 == self.m1 + 2:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr - mr - 3) * (lr - mr - 2) * (lr - mr - 1) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                            elif self.l2 == self.l1:
                                if self.m2 == self.m1 - 2:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 2:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr + mr - 1) * (lr + mr)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                                elif self.m2 == self.m1:
                                    if self.m1 == 1:
                                        r = ((lr - mr) * (lr - mr - 1) / ((2 * lr - 1) * (2 * lr + 1)) + (lr + mr + 1) * (lr + mr + 2) / ((2 * lr + 1) * (2 * lr + 3))) / 4.
                                elif self.m2 == self.m1 + 2:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr - mr - 1) * (lr - mr) * (lr + mr + 1) * (lr + mr + 2)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                            elif self.l2 == self.l1 + 2:
                                if self.m2 == self.m1 - 2:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 2:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) * (lr - mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                                elif self.m2 == self.m1:
                                    if self.m1 == 1:
                                        r = -sqrt((lr + mr + 1) * (lr + mr + 2) * (lr - mr + 1) * (lr - mr + 2) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                                elif self.m2 == self.m1 + 2:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = - eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr + mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                        elif self.dim2 == 2:
                            if self.l2 == self.l1 - 2:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr + mr - 2) * (lr + mr - 1) * (lr + mr) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr + mr) * (lr - mr - 1) * (lr - mr) * (lr - mr - 2) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                            elif self.l2 == self.l1:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.)
                                    r = eps * (-sqrt((lr - mr + 1) * (lr + mr)) * (lr + mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) + sqrt((lr - mr + 1) * (lr + mr)) * (lr - mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * (-sqrt((lr - mr) * (lr + mr + 1)) * (lr - mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) + sqrt((lr - mr) * (lr + mr + 1)) * (lr + mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                            elif self.l2 == self.l1 + 2:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr + mr + 1) * (lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr - mr + 1) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                    elif self.dim1 == 2:
                        if self.dim2 == 1:
                            if self.l2 == self.l1 - 2:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr + mr - 2) * (lr + mr - 1) * (lr + mr) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = -eps * sqrt((lr + mr) * (lr - mr - 1) * (lr - mr) * (lr - mr - 2) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                            elif self.l2 == self.l1:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.)
                                    r = eps * (-sqrt((lr - mr + 1) * (lr + mr)) * (lr + mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) + sqrt((lr - mr + 1) * (lr + mr)) * (lr - mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * (-sqrt((lr - mr) * (lr + mr + 1)) * (lr - mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) + sqrt((lr - mr) * (lr + mr + 1)) * (lr + mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                            elif self.l2 == self.l1 + 2:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1. / sqrt(2.)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr + mr + 1) * (lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.)
                                    r = eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr - mr + 1) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                elif self.phase2 == 1:
                    if self.m2 != 0:
                        if self.dim1 == 0:
                            if self.dim2 == 0:
                                if self.l2 == self.l1 - 2:
                                    if self.m2 == self.m1 - 2:
                                        if self.m1 == 0:
                                            eps = 1. / sqrt(2.)
                                        elif self.m1 == 2:
                                            eps = sqrt(2.)
                                        r = eps * sqrt((lr + mr - 3) * (lr + mr - 2) * (lr + mr - 1) * (lr + mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                                    elif self.m2 == self.m1:
                                        r = -sqrt((lr + mr - 1) * (lr + mr) * (lr - mr - 1) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                                        if self.m1 == 1:
                                            r = 0.5 * r
                                    elif self.m2 == self.m1 + 2:
                                        if self.m1 == 0:
                                            eps = sqrt(2.)
                                        r = eps * sqrt((lr - mr - 3) * (lr - mr - 2) * (lr - mr - 1) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                                elif self.l2 == self.l1:
                                    if self.m2 == self.m1 - 2:
                                        if self.m1 == 0:
                                            eps = 1. / sqrt(2.)
                                        elif self.m1 == 2:
                                            eps = sqrt(2.)
                                        r = -eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr + mr - 1) * (lr + mr)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                                    elif self.m2 == self.m1:
                                        r = (pow(lr, 2) + lr - 1 + pow(mr, 2)) / ((2 * lr - 1) * (2 * lr + 3))
                                        if self.m1 == 1:
                                            r = 0.5 * r
                                    elif self.m2 == self.m1 + 2:
                                        if self.m1 == 0:
                                            eps = sqrt(2.)
                                        r = -eps * sqrt((lr - mr - 1) * (lr - mr) * (lr + mr + 1) * (lr + mr + 2)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                                elif self.l2 == self.l1 + 2:
                                    if self.m2 == self.m1 - 2:
                                        if self.m1 == 0:
                                            eps = 1. / sqrt(2.)
                                        elif self.m1 == 2:
                                            eps = sqrt(2.)
                                        r = eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) * (lr - mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                                    elif self.m2 == self.m1:
                                        r = -sqrt((lr + mr + 1) * (lr + mr + 2) * (lr - mr + 1) * (lr - mr + 2) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                                        if self.m1 == 1:
                                            r = 0.5 * r
                                    elif self.m2 == self.m1 + 2:
                                        if self.m1 == 0:
                                            eps = sqrt(2.)
                                        r = eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr + mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                            elif self.dim2 == 2:
                                if self.l2 == self.l1 - 2:
                                    if self.m2 == self.m1 - 1:
                                        if self.m1 == 0:
                                            eps = 1. / sqrt(2.)
                                        elif self.m1 == 1:
                                            eps = sqrt(2.)
                                        r = -eps * sqrt((lr + mr - 2) * (lr + mr - 1) * (lr + mr) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                                    elif self.m2 == self.m1 + 1:
                                        if self.m1 == 0:
                                            eps = sqrt(2.)
                                        r = eps * sqrt((lr + mr) * (lr - mr - 2) * (lr - mr - 1) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                                elif self.l2 == self.l1:
                                    if self.m2 == self.m1 - 1:
                                        if self.m1 == 0:
                                            eps = 1. / sqrt(2.)
                                        elif self.m1 == 1:
                                            eps = sqrt(2.)
                                        r = eps * (-sqrt((lr + mr) * (lr - mr + 1)) * (lr + mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) + sqrt((lr + mr) * (lr - mr + 1)) * (lr - mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                                    elif self.m2 == self.m1 + 1:
                                        if self.m1 == 0:
                                            eps = sqrt(2.)
                                        r = eps * (sqrt((lr + mr + 1) * (lr - mr)) * (lr - mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) - sqrt((lr + mr + 1) * (lr - mr)) * (lr + mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                                elif self.l2 == self.l1 + 2:
                                    if self.m2 == self.m1 - 1:
                                        if self.m1 == 0:
                                            eps = 1. / sqrt(2.)
                                        elif self.m1 == 1:
                                            eps = sqrt(2.)
                                        r = eps * sqrt((lr + mr + 1) * (lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                                    elif self.m2 == self.m1 + 1:
                                        if self.m1 == 0:
                                            eps = sqrt(2.)
                                        r = -eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr - mr + 1) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                        elif self.dim1 == 1:
                            if self.dim2 == 1:
                                if self.l2 == self.l1 - 2:
                                    if self.m2 == self.m1 - 2:
                                        if self.m1 == 0:
                                            eps = 1. / sqrt(2.)
                                        elif self.m1 == 2:
                                            eps = sqrt(2.)
                                        r = -eps * sqrt((lr + mr - 3) * (lr + mr - 2) * (lr + mr - 1) * (lr + mr) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                                    elif self.m2 == self.m1:
                                        r = -sqrt((lr - mr - 1) * (lr - mr) * (lr + mr - 1) * (lr + mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                                        if self.m1 == 1:
                                            r = 1.5 * r
                                    elif self.m2 == self.m1 + 2:
                                        if self.m1 == 0:
                                            eps = sqrt(2.)
                                        r = -eps * sqrt((lr - mr) * (lr - mr - 1) * (lr - mr - 2) * (lr - mr - 3) / ((2 * lr - 3) * (2 * lr + 1))) / (4 * (2 * lr - 1))
                                elif self.l2 == self.l1:
                                    if self.m2 == self.m1 - 2:
                                        if self.m1 == 0:
                                            eps = 1. / sqrt(2.)
                                        elif self.m1 == 2:
                                            eps = sqrt(2.)
                                        r = eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr + mr - 1) * (lr + mr)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                                    elif self.m2 == self.m1:
                                        r = (pow(lr, 2) + lr - 1 + pow(mr, 2)) / ((2 * lr - 1) * (2 * lr + 3))
                                        if self.m1 == 1:
                                            r = 1.5 * r
                                    elif self.m2 == self.m1 + 2:
                                        if self.m1 == 0:
                                            eps = sqrt(2.)
                                        r = eps * sqrt((lr - mr - 1) * (lr - mr) * (lr + mr + 1) * (lr + mr + 2)) / (2 * (2 * lr - 1) * (2 * lr + 3))
                                elif self.l2 == self.l1 + 2:
                                    if self.m2 == self.m1 - 2:
                                        if self.m1 == 0:
                                            eps = 1. / sqrt(2.)
                                        elif self.m1 == 2:
                                            eps = sqrt(2.)
                                        r = -eps * sqrt((lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) * (lr - mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                                    elif self.m2 == self.m1:
                                        r = - sqrt((lr + mr + 1) * (lr + mr + 2) * (lr - mr + 1) * (lr - mr + 2) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                                        if self.m1 == 1:
                                            r = 1.5 * r
                                    elif self.m2 == self.m1 + 2:
                                        if self.m1 == 0:
                                            eps = sqrt(2.)
                                        r = - eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr + mr + 4) / ((2 * lr + 1) * (2 * lr + 5))) / (4 * (2 * lr + 3))
                        elif self.dim1 == 2:
                            if self.dim2 == 0:
                                if self.l2 == self.l1 - 2:
                                    if self.m2 == self.m1 - 1:
                                        if self.m1 == 0:
                                            eps = 1. / sqrt(2.)
                                        elif self.m1 == 1:
                                            eps = sqrt(2.)
                                        r = -eps * sqrt((lr + mr - 2) * (lr + mr - 1) * (lr + mr) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                                    elif self.m2 == self.m1 + 1:
                                        if self.m1 == 0:
                                            eps = sqrt(2.)
                                        r = eps * sqrt((lr + mr) * (lr - mr - 2) * (lr - mr - 1) * (lr - mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * (2 * lr - 1))
                                elif self.l2 == self.l1:
                                    if self.m2 == self.m1 - 1:
                                        if self.m1 == 0:
                                            eps = 1. / sqrt(2.)
                                        elif self.m1 == 1:
                                            eps = sqrt(2.)
                                        r = eps * (-sqrt((lr + mr) * (lr - mr + 1)) * (lr + mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) + sqrt((lr + mr) * (lr - mr + 1)) * (lr - mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                                    elif self.m2 == self.m1 + 1:
                                        if self.m1 == 0:
                                            eps = sqrt(2.)
                                        r = eps * (sqrt((lr + mr + 1) * (lr - mr)) * (lr - mr - 1) / (2 * (2 * lr - 1) * (2 * lr + 1)) - sqrt((lr + mr + 1) * (lr - mr)) * (lr + mr + 2) / (2 * (2 * lr + 1) * (2 * lr + 3)))
                                elif self.l2 == self.l1 + 2:
                                    if self.m2 == self.m1 - 1:
                                        if self.m1 == 0:
                                            eps = 1. / sqrt(2.)
                                        elif self.m1 == 1:
                                            eps = sqrt(2.)
                                        r = eps * sqrt((lr + mr + 1) * (lr - mr + 1) * (lr - mr + 2) * (lr - mr + 3) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                                    elif self.m2 == self.m1 + 1:
                                        if self.m1 == 0:
                                            eps = sqrt(2.)
                                        r = -eps * sqrt((lr + mr + 1) * (lr + mr + 2) * (lr + mr + 3) * (lr - mr + 1) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * (2 * lr + 3))
                            elif self.dim2 == 2:
                                if self.l2 == self.l1 - 2:
                                    if self.m2 == self.m1:
                                        r = sqrt((lr - mr - 1) * (lr - mr) * (lr + mr - 1) * (lr + mr) / ((2 * lr - 3) * (2 * lr + 1))) / (2 * lr - 1)
                                elif self.l2 == self.l1:
                                    if self.m2 == self.m1:
                                        r = (2 * pow(lr, 2) + 2 * lr - 1 - 2 * pow(mr, 2)) / ((2 * lr - 1) * (2 * lr + 3))
                                elif self.l2 == self.l1 + 2:
                                    if self.m2 == self.m1:
                                        r = sqrt((lr - mr + 1) * (lr - mr + 2) * (lr + mr + 1) * (lr + mr + 2) / ((2 * lr + 1) * (2 * lr + 5))) / (2 * lr + 3)
        return r

    # Python-style aliases; the original C++-style API remains available.
    def set(
        self,
        dim1: int,
        dim2: int,
        l1: int,
        l2: int,
        m1: int,
        m2: int,
        phase1: int,
        phase2: int,
    ) -> None:
        self.Set(dim1, dim2, l1, l2, m1, m2, phase1, phase2)

    def get_moment(self) -> float:
        return self.GetMom()


if __name__ == "__main__":

    # to make the system of equations, we need to iterate through every l',m',alpha' index (l' < L_max) for the 
    # SH function being multiplied onto the equation, and then find all of the values of the
    # coefficients for the phi^{\alpha}_{l,m} functions (which will only be non-zero when l and m are close to l' and m' and l < L_max).
    # Each row will be a different combo of l',m', and alpha' and the columns are the combo of l, m, and alpha

    L_max = 4
    dim1 = 0 # 0 is for x, 1 is for y, 2 is for z
    dim2 = 0 # 0 is for x, 1 is for y, 2 is for z
    N_c = int(0.5 * (L_max + 1) * (L_max + 2))  # total term in the sum of l and m indices is 0.5 * (L_max + 1) * (L_max + 2) for the cosine terms
    N_s = int(0.5 * (L_max) * (L_max + 1))  # total term in the sum of l and m indices is 0.5 * (L_max) * (L_max + 1) for the sine terms, fewer because m=0 is not included
    M = np.zeros((N_c + N_s,N_c + N_s))
    moment = sOSM(dim1=0, dim2=0, l1=0, l2=0, m1=0, m2=0, phase1=0, phase2=0)

    row_i = 0  # ranges from 0 to N_c+N_s-1 for each of the basis functions multiplied and integrated over
    for alpha_p in range(2): # 0 is for cosine, 1 is for sine
        for l_p in range(L_max + 1):
            for m_p in range(alpha_p, l_p + 1):
                col_c_i = 0 # ranges from 0 to N_c-1 for each l and m combination (sine and cosine forms set at the same time)
                col_s_i = 0 # ranges from 0 to N_s-1 for each l and m combination (sine and cosine forms set at the same time)
                for l in range(L_max  + 1):
                    for m in range(l + 1):
                        # cosine moment term
                        moment.set(dim1=dim1, dim2=dim2, l1=l_p, l2=l, m1=m_p, m2=m, phase1=alpha_p, phase2=0) # alpha=cosine
                        M[row_i, col_c_i] += moment.get_moment()

                        # sine moment term
                        if m != 0:
                            moment.set(dim1=dim1, dim2=dim2, l1=l_p, l2=l, m1=m_p, m2=m, phase1=alpha_p, phase2=1) # alpha=sine
                            M[row_i, col_s_i + N_c] += moment.get_moment()
                            col_s_i += 1

                        col_c_i+=1
                    
                row_i += 1
    
    M_sing_vals = np.linalg.svd(M, compute_uv=False)
    M_sing_max = M_sing_vals[0]
    M_sing_min = M_sing_vals[-1]
    M_cond = M_sing_max / M_sing_min

                                    



    #moment = sOSM(dim1=2, dim2=2, l1=0, l2=0, m1=0, m2=0, phase1=0, phase2=0)
    #print("Second-order streaming moment =", moment.GetMom())

    print("Finished")