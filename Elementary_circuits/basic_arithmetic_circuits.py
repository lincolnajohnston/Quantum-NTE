from qiskit import QuantumCircuit

import numpy as np
import Elementary_circuits.subarithmetic_circuits as subarithmetic_circuits

# |a> -> |(2**n - 1) - a>
def Ones_complement(qc, a):
    """Replace an n-qubit bit string with its one's complement in place.

    An X gate is applied independently to every qubit, so the basis-state
    mapping is ``|a> -> |(2**n - 1) - a>``.  The register is little-endian,
    although the operation itself is independent of qubit order.
    """
    # X maps each stored bit a[i] to a[i] XOR 1.
    for i in range(len(a)):
        qc.x(a[i])


# From section 4 of https://arxiv.org/pdf/1712.02630
# modular: |a>|b> -> |a>|(a + b) mod 2**n>; full: |a>|b> -> |a>|a + b>
def Addition_gate(qc, a, b, modular=False, useElementaryGates=True):
    """Add little-endian register ``a`` into ``b`` while preserving ``a``.

    In modular mode, both registers contain n qubits and the mapping is
    ``|a>|b> -> |a>|(a + b) mod 2**n>``.  In full-addition mode, ``b`` has
    n + 1 qubits; when ``b[n]`` starts at zero it receives the carry-out and
    the complete n + 1 bit sum is stored in ``b``.

    The six stages below use CNOTs to encode and later uncompute carry
    dependencies, Toffoli gates to generate carries, and Peres gates to write
    sum bits while propagating carries.  ``useElementaryGates`` selects the
    Clifford+T implementations of the Toffoli and Peres gates when true.
    """
    n = len(a)
    if n + int(not modular) != len(b):
        raise Exception("a and b registers are not the same length for modular addition")
    
    # Step 1: XOR the upper a bits into b to begin encoding propagate bits.
    for i in range(1, n):
        qc.cx(a[i], b[i])

    qc.barrier(label="Step 2")

    # Step 2: expose the full-mode carry target and propagate information
    # toward the most-significant end of a.
    if not modular:
        qc.cx(a[n-1], b[n])
    for i in range(n-2,0,-1):
        qc.cx(a[i], a[i+1])

    qc.barrier(label="Step 3")

    # Step 3: Toffoli gates generate the carry dependencies in a[i + 1].
    for i in range(n-1):
        qc.barrier(label="Toffoli Gate")
        subarithmetic_circuits.Toffoli_gate(qc, [a[i+1], a[i], b[i]], useElementaryGates = useElementaryGates) # using Clifford + T gates

    qc.barrier(label="Step 4")

    # Step 4: Peres gates propagate carries back toward the low end while
    # writing the corresponding sum bits into b.
    if not modular:
        subarithmetic_circuits.Peres_gate(qc, [b[n], b[n-1], a[n-1]], useElementaryGates = useElementaryGates)
    else:
        qc.cx(a[n-1], b[n-1])
    for i in range(n-2, -1,-1):
        qc.barrier(label="Peres Gate")
        subarithmetic_circuits.Peres_gate(qc, [a[i+1], b[i], a[i]], useElementaryGates = useElementaryGates)

    qc.barrier(label="Step 5")

    # Step 5: undo the temporary carry-propagation correlations within a.
    for i in range(1, n-1):
        qc.cx(a[i], a[i+1])

    qc.barrier(label="Step 6")  

    # Step 6: finish the sum XORs and restore a to its input value.
    for i in range(1, n):
        qc.cx(a[i], b[i])


# modular: |a>|b>|0> -> |a>|b>|(a + b) mod 2**n>; full: |a>|b>|0> -> |a>|b>|a + b>
def Addition_gate_outofplace(qc, a, b, result, modular=False, useElementaryGates=True):
    """Add ``a`` and ``b`` into a zeroed result register, preserving both inputs.

    This is the out-of-place counterpart of :func:`Addition_gate`.  ``result``
    must start in |0> and have the same width as ``b``.  In modular mode all
    three registers have n qubits and the mapping is
    ``|a>|b>|0> -> |a>|b>|(a + b) mod 2**n>``.  In full-addition mode ``b``
    and ``result`` have n + 1 qubits; with ``b[n]`` initially zero, ``result``
    receives the complete n + 1 bit sum.

    CNOTs first copy ``b`` into ``result``.  The existing in-place adder then
    adds ``a`` to that copy, so neither source register is changed.
    """
    n = len(a)
    expected_width = n + int(not modular)
    if n <= 1:
        raise ValueError("Addition_gate_outofplace requires at least two operand bits")
    if len(b) != expected_width or len(result) != expected_width:
        raise ValueError(
            "b and result must have length n for modular addition or n + 1 "
            "for full addition"
        )

    # Prepare result = b without modifying b.
    for i in range(len(b)):
        qc.cx(b[i], result[i])

    # result <- result + a; Addition_gate preserves its first operand a.
    Addition_gate(
        qc,
        a,
        result,
        modular=modular,
        useElementaryGates=useElementaryGates,
    )


# |a>|0> -> |(-a) mod 2**n>|0>
def Twos_complement(qc, a, b):
    """Negate a little-endian register in place using a zeroed work register.

    For ``n = len(a)`` this performs ``|a>|0> -> |-a mod 2**n>|0>``.
    ``b`` must contain n zeroed work qubits.  The circuit first complements
    every bit of ``a`` and then adds one, which is the definition of two's
    complement negation.  ``Addition_gate`` preserves ``b`` so it can be
    returned to zero after supplying the constant one.
    """
    # Form the one's complement, 2**n - 1 - a.
    Ones_complement(qc, a)
    # Store the constant 1 as little-endian 00...001 in the work register.
    qc.x(b[0])  # prepare the constant 1 in the work register
    # a <- a + 1 (mod 2**n), while the constant register b is preserved.
    Addition_gate(qc, b, a, modular=True, useElementaryGates=False)
    qc.x(b[0])  # Addition_gate preserves b, so return it to |0...0>


# |a>|0>|0> -> |a>|0>|(-a) mod 2**n>
def Twos_complement_outofplace(qc, a, b, result):
    """Write the two's complement of ``a`` to ``result``, preserving ``a``.

    All three little-endian registers must have the same nonzero width.
    ``b`` is a work register and ``result`` is the output register; both must
    start in |0>.  The mapping is
    ``|a>|0>|0> -> |a>|0>|-a mod 2**n>``.

    CNOTs first copy ``a`` into ``result``.  The existing in-place
    :func:`Twos_complement` circuit negates that copy and restores ``b`` to
    zero, leaving the original ``a`` unchanged.
    """
    n = len(a)
    if n == 0 or len(b) != n or len(result) != n:
        raise ValueError("a, b, and result must have the same nonzero length")

    # Prepare result = a without changing the input register.
    for i in range(n):
        qc.cx(a[i], result[i])

    Twos_complement(qc, result, b)


# |c>|a>|b>|0>|0> -> |c>|a>|(b + c*a) mod 2**n>|floor((b + c*a)/2**n)>|0>
def Conditional_addition_gate(qc, ctrl, a, b, carry, work, useElementaryGates=True):
    """Conditionally add ``a`` into ``b`` and expose the unsigned carry-out.

    This is the seven-step Ctrl-Add circuit from Section III of
    arXiv:1706.05113.  With zeroed ``carry`` and ``work`` qubits it maps
    ``|c>|a>|b>|0>|0>`` to ``|c>|a>|b + c*(a mod 2**n)>|carry>|0>``.
    The control and ``a`` are preserved.  ``carry`` is one exactly when the
    enabled unsigned addition overflows; ``work`` is always restored to zero.

    Registers are little-endian and must have the same width of at least two qubits.
    """
    n = len(a)
    if n < 2 or len(b) != n:
        raise ValueError("a and b must have the same length of at least two")

    # |x>|y>|t> -> |x>|y>|t XOR (x AND y)>
    def apply_toffoli(control_1, control_2, target):
        subarithmetic_circuits.Toffoli_gate(qc, [target, control_1, control_2], useElementaryGates=useElementaryGates)

    # Step 1: encode the initial propagate information in b[1:].
    for i in range(1, n):
        qc.cx(a[i], b[i])

    # Step 2: condition the top carry on ctrl and ripple dependencies up a.
    apply_toffoli(ctrl, a[n - 1], carry)
    for i in range(n - 2, 0, -1):
        qc.cx(a[i], a[i + 1])

    # Step 3: generate the internal carries in a[1:].
    for i in range(n - 1):
        apply_toffoli(b[i], a[i], a[i + 1])

    # Step 4: use work to condition the final carry without leaving garbage,
    # then write the controlled most-significant sum bit.
    apply_toffoli(b[n - 1], a[n - 1], work)
    apply_toffoli(ctrl, work, carry)
    apply_toffoli(b[n - 1], a[n - 1], work)
    apply_toffoli(ctrl, a[n - 1], b[n - 1])

    # Step 5: uncompute internal carries and write lower controlled sum bits.
    for i in range(n - 2, -1, -1):
        apply_toffoli(b[i], a[i], a[i + 1])
        apply_toffoli(ctrl, a[i], b[i])

    # Step 6: restore the temporary ripple encoding within a.
    for i in range(1, n - 1):
        qc.cx(a[i], a[i + 1])

    # Step 7: finish restoring a and the propagate encoding in b.
    for i in range(1, n):
        qc.cx(a[i], b[i])


# |c>|a>|b>|0>^(n+1)|0> -> |c>|a>|b>|b + c*a>|0>
def Conditional_addition_gate_outofplace(
    qc, ctrl, a, b, result, work, useElementaryGates=True
):
    """Conditionally add ``a`` and ``b`` into ``result``, preserving inputs.

    For equal-width n-qubit, little-endian inputs, ``result`` must be a zeroed
    n + 1 qubit register and ``work`` must be a zeroed qubit.  This maps
    ``|c>|a>|b>|0>^(n+1)|0>`` to
    ``|c>|a>|b>|b + c*a>|0>``.  Thus ``result[0:n]`` contains the n low sum
    bits and ``result[n]`` contains the carry-out.  ``ctrl``, ``a``, and ``b``
    are preserved, and ``work`` is restored to zero.

    CNOTs copy ``b`` into the low result bits.  The existing conditional adder
    then targets that copy, using ``result[n]`` as its carry-output qubit.
    """
    n = len(a)
    if n < 2 or len(b) != n:
        raise ValueError("a and b must have the same length of at least two")
    if len(result) != n + 1:
        raise ValueError("result must have length n + 1")

    # Prepare the low result bits as a copy of b; result[n] remains zero.
    for i in range(n):
        qc.cx(b[i], result[i])

    Conditional_addition_gate(
        qc,
        ctrl,
        a,
        result[:n],
        result[n],
        work,
        useElementaryGates=useElementaryGates,
    )


# |a>|b>|0> -> |a>|b>|(b - a) mod 2**n>
def Integer_subtraction_gate(qc, a, b, p, useElementaryGates=True):
    """Compute ``(b - a) mod 2**n`` in a zeroed unsigned register p.

    The registers use little-endian standard binary representation.  Both
    input registers are preserved, and underflow wraps modulo ``2**n``::

        |a>|b>|0> -> |a>|b>|(b - a) mod 2**n>

    ``p`` must start in |0> so that copying ``b`` into it prepares the minuend.
    """
    n = len(a)
    if n == 0 or len(b) != n or len(p) != n:
        raise ValueError("a, b, and p must have the same nonzero length")

    # Prepare p = b without modifying either input register.
    for i in range(n):
        qc.cx(b[i], p[i])

    # Addition_gate implements p <- p + a modulo 2**n.  Its inverse therefore
    # implements p <- p - a while preserving a.  Constructing and composing
    # the inverse also keeps useElementaryGates consistent with Addition_gate.
    addition = QuantumCircuit(2 * n, name="modular_addition")
    Addition_gate(
        addition,
        addition.qubits[:n],
        addition.qubits[n:],
        modular=True,
        useElementaryGates=useElementaryGates,
    )
    qc.compose(
        addition.inverse(),
        qubits=list(a) + list(p),
        inplace=True,
    )


# |a>|b>|0>^(2*n+1) -> |a>|b>|a*b>|0>
def Integer_multiplication_gate(qc, a, b, p, useElementaryGates=True):
    """Multiply two equal-width unsigned integers into a zeroed register.

    For n-qubit, little-endian inputs this implements
    ``|a>|b>|0>^(2*n + 1) -> |a>|b>|a*b>|0>``.  The 2*n product bits occupy
    ``p[0:2*n]``; ``p[2*n]`` is temporary carry workspace and is restored to
    zero.  Both inputs are preserved.

    The circuit is a shift-and-add multiplier.  Toffoli gates form the partial
    product selected by ``b[0]``.  Each higher bit ``b[j]`` then controls an
    addition of ``a`` into the shifted slice ``p[j:j+n]``; the next two p
    qubits supply that addition's carry and clean work qubit.
    """
    n = len(a)
    if n != len(b) or len(p) != 2 * n + 1:
        raise ValueError("a and b must have length n and p must have length 2*n + 1")

    # Paper Step 1: p[i] ^= b[0] AND a[i], forming a*b[0].
    for i in range(n):
        subarithmetic_circuits.Toffoli_gate(qc, [p[i], b[0], a[i]], useElementaryGates=useElementaryGates)

    # Paper Steps 2 and 3: when b[j] is one, add a*2**j into p.
    for j in range(1, n):
        Conditional_addition_gate(
            qc,
            b[j],
            a,
            p[j:j + n],
            p[j + n],
            p[j + n + 1],
            useElementaryGates=useElementaryGates,
        )

# |c>|r> -> |c>|(-r) mod 2**n> if c=1, otherwise |c>|r>
def controlled_twos_complement(qc, ctrl, register, useElementaryGates=True):
    """Negate a little-endian register modulo its width when ``ctrl`` is one.

    The mapping is ``|c>|r> -> |c>|r>`` for c=0 and
    ``|1>|r> -> |1>|-r mod 2**n>`` for c=1.  ``ctrl`` is preserved.  Controlled
    X gates first form the one's complement only on the enabled branch.  A
    descending ladder of multi-controlled X gates then increments that branch
    by one; descending order ensures lower controls still hold their original
    complemented values.
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
                useElementaryGates=useElementaryGates,
            )
        else:
            qc.mcx(controls, register[i])
    # The least-significant increment bit needs only the external control.
    qc.cx(ctrl, register[0])


# |a>|b>|0>^(2*n+1)|00> -> |a>|b>|a*b>|0>|00> (two's-complement a, b, and product)
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
    controlled_twos_complement(qc, sign[0], a, useElementaryGates=useElementaryGates)
    controlled_twos_complement(qc, sign[1], b, useElementaryGates=useElementaryGates)

    Integer_multiplication_gate(qc, a, b, p, useElementaryGates=useElementaryGates)

    # Applying negation once per negative operand computes the sign XOR: two
    # negative operands cause two negations, which cancel.
    controlled_twos_complement(qc, sign[0], p[:2 * n], useElementaryGates=useElementaryGates)
    controlled_twos_complement(qc, sign[1], p[:2 * n], useElementaryGates=useElementaryGates)

    # Restore the input registers before clearing their saved sign bits.
    controlled_twos_complement(qc, sign[1], b, useElementaryGates=useElementaryGates)
    controlled_twos_complement(qc, sign[0], a, useElementaryGates=useElementaryGates)
    qc.cx(b[n - 1], sign[1])
    qc.cx(a[n - 1], sign[0])
