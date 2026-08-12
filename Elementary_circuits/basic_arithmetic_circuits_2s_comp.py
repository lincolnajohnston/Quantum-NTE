import Elementary_circuits.basic_arithmetic_circuits as basic_arithmetic_circuits
import Elementary_circuits.subarithmetic_circuits as subarithmetic_circuits

# TODO: make sure every arithmetic operation has an inplace and outofplace version, and create tests to show that all of these work


# |a>^n|0>^n -> |(-a) mod 2**n>^n|0>^n
def Twos_complement_inplace(qc, a, b):
    """Negate a little-endian register in place using a zeroed work register.

    For ``n = len(a)`` this performs ``|a>|0> -> |-a mod 2**n>|0>``.
    ``b`` must contain n zeroed work qubits.  The circuit first complements
    every bit of ``a`` and then adds one, which is the definition of two's
    complement negation.  ``Addition_gate`` preserves ``b`` so it can be
    returned to zero after supplying the constant one.

    Gate Counts:
    - (n + 2) X gates
    - (4n - 5) CNOT gates
    - (n - 1) CCX gates
    - (n - 1) Peres gates
    ----------------
    - (5n - 3) Clifford gates before decomposing CCX and Peres gates
    - 0 T gates before decomposing CCX and Peres gates
    """
    # Form the one's complement, 2**n - 1 - a.
    basic_arithmetic_circuits.Ones_complement_inplace(qc, a)
    # Store the constant 1 as little-endian 00...001 in the work register.
    qc.x(b[0])  # prepare the constant 1 in the work register
    # a <- a + 1 (mod 2**n), while the constant register b is preserved.
    basic_arithmetic_circuits.Addition_gate_inplace(
        qc, b, a, modular=True, useElementaryGates=False
    )
    qc.x(b[0])  # Addition_gate preserves b, so return it to |0...0>


# |a>^n|0>^n|0>^n -> |a>^n|0>^n|(-a) mod 2**n>^n
def Twos_complement_outofplace(qc, a, b, result):
    """Write the two's complement of ``a`` to ``result``, preserving ``a``.

    All three little-endian registers must have the same nonzero width.
    ``b`` is a work register and ``result`` is the output register; both must
    start in |0>.  The mapping is
    ``|a>|0>|0> -> |a>|0>|-a mod 2**n>``.

    CNOTs first copy ``a`` into ``result``.  The existing in-place
    :func:`Twos_complement` circuit negates that copy and restores ``b`` to
    zero, leaving the original ``a`` unchanged.

    Gate Counts:
    - (n + 2) X gates
    - (5n - 5) CNOT gates
    - (n - 1) CCX gates
    - (n - 1) Peres gates
    ----------------
    - (6n - 3) Clifford gates before decomposing CCX and Peres gates
    - 0 T gates before decomposing CCX and Peres gates
    """
    n = len(a)
    if n == 0 or len(b) != n or len(result) != n:
        raise ValueError("a, b, and result must have the same nonzero length")

    # Prepare result = a without changing the input register.
    for i in range(n):
        qc.cx(a[i], result[i])

    Twos_complement_inplace(qc, result, b)


# |c>^1|r>^n -> |c>^1|(-r) mod 2**n>^n if c=1, otherwise |c>^1|r>^n
def controlled_twos_complement_inplace(qc, ctrl, register, useElementaryGates=True):
    """Negate a little-endian register modulo its width when ``ctrl`` is one.

    The mapping is ``|c>|r> -> |c>|r>`` for c=0 and
    ``|1>|r> -> |1>|-r mod 2**n>`` for c=1.  ``ctrl`` is preserved.  Controlled
    X gates first form the one's complement only on the enabled branch.  A
    descending ladder of multi-controlled X gates then increments that branch
    by one; descending order ensures lower controls still hold their original
    complemented values.

    Gate Counts:
    - 6 H gates
    - 4 S-dagger gates
    - 9 T gates
    - (n + 9) CNOT gates
    - (n - 2) MCX gates
    ----------------
    - (n + 19) Clifford gates before decomposing MCX gates
    - 9 T gates before decomposing MCX gates
    """
    # Negation is one's complement followed by an increment.  Apply the
    # increment from most to least significant bit so its controls still
    # see the pre-increment values of the lower bits.
    for qubit in register:
        qc.cx(ctrl, qubit)

    for i in range(len(register) - 1, 0, -1):
        # Flip bit i iff the operation is enabled and every lower bit is one,
        # which is precisely the carry condition for incrementing register[i].
        controls = [ctrl] + list(register[:i])
        if len(controls) == 2:
            subarithmetic_circuits.Toffoli_gate(
                qc,
                [register[i], controls[0], controls[1]],
                [0, 0],
                useElementaryGates=useElementaryGates,
            )
        else:
            qc.mcx(controls, register[i])
    # The least-significant increment bit needs only the external control.
    qc.cx(ctrl, register[0])


# |a>^n|b>^n|0>^(2*n+1)|0>^2 -> |a>^n|b>^n|a*b>^(2*n)|0>^1|0>^2 (two's-complement a, b, and product)
def Twos_complement_integer_multiplication_gate(qc, a, b, p, sign, useElementaryGates=True):
    """Multiply equal-width two's-complement integers into a zeroed product register.

    For two n-qubit, little-endian input registers, this implements

        |a>|b>|0>^(2*n + 1)|00> -> |a>|b>|a*b>|0>|00>.

    The 2*n-bit two's-complement product occupies ``p[0:2*n]``.  ``p[2*n]``
    is the work qubit used by :func:`Integer_multiplication_gate`.  The two
    qubits in ``sign`` must start in |00> and are restored to |00>.

    The inputs are conditionally converted to their unsigned magnitudes, the
    existing unsigned multiplier is applied, and the product is conditionally
    negated when the input signs differ.  CNOTs copy the original sign bits to
    ``sign``; controlled two's-complement circuits perform each conversion and
    final sign correction.  Both inputs and both sign-work qubits are restored.

    Gate Counts:
    - (18n**2 - 6n + 30) H gates
    - (12n**2 - 4n + 20) S-dagger gates
    - (27n**2 - 9n + 45) T gates
    - (28n**2 - 10n + 56) CNOT gates
    - (8n - 12) MCX gates
    ----------------
    - (58n**2 - 20n + 106) Clifford gates before decomposing MCX gates
    - (27n**2 - 9n + 45) T gates before decomposing MCX gates
    """
    n = len(a)
    if n == 0 or len(b) != n:
        raise ValueError("a and b must have the same nonzero length")
    if len(p) != 2 * n + 1:
        raise ValueError("p must have length 2*n + 1")
    if len(sign) != 2:
        raise ValueError("sign must contain two work qubits")


    # Preserve the original signs while a and b are temporarily magnitudes.
    qc.cx(a[n - 1], sign[0])
    qc.cx(b[n - 1], sign[1])
    controlled_twos_complement_inplace(
        qc, sign[0], a, useElementaryGates=useElementaryGates
    )
    controlled_twos_complement_inplace(
        qc, sign[1], b, useElementaryGates=useElementaryGates
    )

    basic_arithmetic_circuits.Integer_multiplication_gate(
        qc, a, b, p, useElementaryGates=useElementaryGates
    )

    # Applying negation once per negative operand computes the sign XOR: two
    # negative operands cause two negations, which cancel.
    controlled_twos_complement_inplace(
        qc, sign[0], p[:2 * n], useElementaryGates=useElementaryGates
    )
    controlled_twos_complement_inplace(
        qc, sign[1], p[:2 * n], useElementaryGates=useElementaryGates
    )

    # Restore the input registers before clearing their saved sign bits.
    controlled_twos_complement_inplace(
        qc, sign[1], b, useElementaryGates=useElementaryGates
    )
    controlled_twos_complement_inplace(
        qc, sign[0], a, useElementaryGates=useElementaryGates
    )
    qc.cx(b[n - 1], sign[1])
    qc.cx(a[n - 1], sign[0])
