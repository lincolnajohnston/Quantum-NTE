"""Python translation of the C++ fOSM class.

This module computes a first-order streaming moment for real spherical
harmonics using the same branching logic as the original fOSM.cc file.

Conventions from the original implementation:
    phase = 0: cosine harmonic
    phase = 1: sine harmonic
    dim = 0, 1, 2: spatial direction indices used by the original code
"""

from math import sqrt
import numpy as np


class fOSM:
    """First-order streaming moment calculator."""

    def __init__(self) -> None:
        self.dim = 0
        self.l1 = 0
        self.l2 = 0
        self.m1 = 0
        self.m2 = 0
        self.phase1 = 0
        self.phase2 = 0

    def Set(
        self,
        dim: int,
        l1: int,
        l2: int,
        m1: int,
        m2: int,
        phase1: int,
        phase2: int,
    ) -> None:
        """Set dimension and harmonic indices.

        Args:
            dim: Spatial dimension/direction index.
            l1, l2: Spherical harmonic degree indices.
            m1, m2: Spherical harmonic order indices.
            phase1, phase2: 0 for cosine and 1 for sine.
        """
        self.dim = dim
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.phase1 = phase1
        self.phase2 = phase2

    def GetMom(self) -> float:
        """Return the first-order streaming moment."""
        lr = float(self.l1)
        mr = float(self.m1)
        r = 0.0
        eps = 1.0

        if self.phase1 == 0:
            if self.phase2 == 0:
                if self.dim == 0:
                    if self.l2 == self.l1 - 1:
                        if self.m2 == self.m1 - 1:
                            if self.m1 == 0:
                                eps = 1.0 / sqrt(2.0)
                            elif self.m1 == 1:
                                eps = sqrt(2.0)
                            r = -sqrt(
                                (lr + mr - 1) * (lr + mr)
                                / ((2 * lr - 1) * (2 * lr + 1))
                            ) / 2
                        elif self.m2 == self.m1 + 1:
                            if self.m1 == 0:
                                eps = sqrt(2.0)
                            r = sqrt(
                                (lr - mr - 1) * (lr - mr)
                                / ((2 * lr - 1) * (2 * lr + 1))
                            ) / 2
                    elif self.l2 == self.l1 + 1:
                        if self.m2 == self.m1 - 1:
                            if self.m1 == 0:
                                eps = 1.0 / sqrt(2.0)
                            elif self.m1 == 1:
                                eps = sqrt(2.0)
                            r = sqrt(
                                (lr - mr + 1) * (lr - mr + 2)
                                / ((2 * lr + 1) * (2 * lr + 3))
                            ) / 2
                        elif self.m2 == self.m1 + 1:
                            if self.m1 == 0:
                                eps = sqrt(2.0)
                            r = -sqrt(
                                (lr + mr + 1) * (lr + mr + 2)
                                / ((2 * lr + 1) * (2 * lr + 3))
                            ) / 2
                elif self.dim == 2:
                    if self.l2 == self.l1 - 1:
                        if self.m2 == self.m1:
                            r = sqrt(
                                (lr - mr) * (lr + mr)
                                / ((2 * lr - 1) * (2 * lr + 1))
                            )
                    elif self.l2 == self.l1 + 1:
                        if self.m2 == self.m1:
                            r = sqrt(
                                (lr - mr + 1) * (lr + mr + 1)
                                / ((2 * lr + 1) * (2 * lr + 3))
                            )

            elif self.phase2 == 1:
                if self.dim == 1 and self.m2 != 0:
                    if self.l2 == self.l1 - 1:
                        if self.m2 == self.m1 - 1:
                            if self.m1 == 0:
                                eps = 1.0 / sqrt(2.0)
                            elif self.m1 == 1:
                                eps = sqrt(2.0)
                            r = sqrt(
                                (lr + mr - 1) * (lr + mr)
                                / ((2 * lr - 1) * (2 * lr + 1))
                            ) / 2
                        elif self.m2 == self.m1 + 1:
                            if self.m1 == 0:
                                eps = sqrt(2.0)
                            r = sqrt(
                                (lr - mr - 1) * (lr - mr)
                                / ((2 * lr - 1) * (2 * lr + 1))
                            ) / 2
                    elif self.l2 == self.l1 + 1:
                        if self.m2 == self.m1 - 1:
                            if self.m1 == 0:
                                eps = 1.0 / sqrt(2.0)
                            elif self.m1 == 1:
                                eps = sqrt(2.0)
                            r = -sqrt(
                                (lr - mr + 1) * (lr - mr + 2)
                                / ((2 * lr + 1) * (2 * lr + 3))
                            ) / 2
                        elif self.m2 == self.m1 + 1:
                            if self.m1 == 0:
                                eps = sqrt(2.0)
                            r = -sqrt(
                                (lr + mr + 1) * (lr + mr + 2)
                                / ((2 * lr + 1) * (2 * lr + 3))
                            ) / 2

        elif self.phase1 == 1:
            if self.m1 != 0:
                if self.phase2 == 0:
                    if self.dim == 1:
                        if self.l2 == self.l1 - 1:
                            if self.m2 == self.m1 - 1:
                                if self.m1 == 0:
                                    eps = 1.0 / sqrt(2.0)
                                elif self.m1 == 1:
                                    eps = sqrt(2.0)
                                r = -sqrt(
                                    (lr + mr - 1) * (lr + mr)
                                    / ((2 * lr - 1) * (2 * lr + 1))
                                ) / 2
                            elif self.m2 == self.m1 + 1:
                                if self.m1 == 0:
                                    eps = sqrt(2.0)
                                r = -sqrt(
                                    (lr - mr - 1) * (lr - mr)
                                    / ((2 * lr - 1) * (2 * lr + 1))
                                ) / 2
                        elif self.l2 == self.l1 + 1:
                            if self.m2 == self.m1 - 1:
                                if self.m1 == 0:
                                    eps = 1.0 / sqrt(2.0)
                                elif self.m1 == 1:
                                    eps = sqrt(2.0)
                                r = sqrt(
                                    (lr - mr + 1) * (lr - mr + 2)
                                    / ((2 * lr + 1) * (2 * lr + 3))
                                ) / 2
                            elif self.m2 == self.m1 + 1:
                                if self.m1 == 0:
                                    eps = sqrt(2.0)
                                r = sqrt(
                                    (lr + mr + 1) * (lr + mr + 2)
                                    / ((2 * lr + 1) * (2 * lr + 3))
                                ) / 2

                elif self.phase2 == 1:
                    if self.m2 != 0:
                        if self.dim == 0:
                            if self.l2 == self.l1 - 1:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1.0 / sqrt(2.0)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.0)
                                    r = -sqrt(
                                        (lr + mr - 1) * (lr + mr)
                                        / ((2 * lr - 1) * (2 * lr + 1))
                                    ) / 2
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.0)
                                    r = sqrt(
                                        (lr - mr - 1) * (lr - mr)
                                        / ((2 * lr - 1) * (2 * lr + 1))
                                    ) / 2
                            elif self.l2 == self.l1 + 1:
                                if self.m2 == self.m1 - 1:
                                    if self.m1 == 0:
                                        eps = 1.0 / sqrt(2.0)
                                    elif self.m1 == 1:
                                        eps = sqrt(2.0)
                                    r = sqrt(
                                        (lr - mr + 1) * (lr - mr + 2)
                                        / ((2 * lr + 1) * (2 * lr + 3))
                                    ) / 2
                                elif self.m2 == self.m1 + 1:
                                    if self.m1 == 0:
                                        eps = sqrt(2.0)
                                    r = -sqrt(
                                        (lr + mr + 1) * (lr + mr + 2)
                                        / ((2 * lr + 1) * (2 * lr + 3))
                                    ) / 2
                        elif self.dim == 2:
                            if self.l2 == self.l1 - 1:
                                if self.m2 == self.m1:
                                    r = sqrt(
                                        (lr - mr) * (lr + mr)
                                        / ((2 * lr - 1) * (2 * lr + 1))
                                    )
                            elif self.l2 == self.l1 + 1:
                                if self.m2 == self.m1:
                                    r = sqrt(
                                        (lr - mr + 1) * (lr + mr + 1)
                                        / ((2 * lr + 1) * (2 * lr + 3))
                                    )

        return eps * r

    # Python-style aliases, while preserving the original C++ method names above.
    def set(
        self,
        dim: int,
        l1: int,
        l2: int,
        m1: int,
        m2: int,
        phase1: int,
        phase2: int,
    ) -> None:
        self.Set(dim, l1, l2, m1, m2, phase1, phase2)

    def get_moment(self) -> float:
        return self.GetMom()


if __name__ == "__main__":
    moment = fOSM()

    # to make the system of equations, we need to iterate through every l',m',alpha' index (l' < L_max) for the 
    # SH function being multiplied onto the equation, and then find all of the values of the
    # coefficients for the phi^{\alpha}_{l,m} functions (which will only be non-zero when l and m are close to l' and m' and l < L_max)

    # We will first do the second term on the LHS of the equation with only first order
    '''L_max = 1
    for lp in range(L_max):
        for mp in range()
    phase_1_vec = [0,1]
    phase_2_vec = [0,1]
    l1_vec = np.arange(0,L_max)'''


    moment.Set(dim=2, l1=1, l2=0, m1=0, m2=0, phase1=0, phase2=0)
    print("First-order streaming moment =", moment.GetMom())
