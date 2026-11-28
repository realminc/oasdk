#!/usr/bin/env python3
"""Record, combine, and multiply matrices on the default runtime.

Mirrors sdk/rs/tutorials/core/tu_core_basics/tu_core_basics.rs.
"""

import oa

# Record on the default runtime and checkpoint it: checkpoint submits all
# pending eager work without blocking the host. read_f32 flushes on its own.
runtime = oa.Runtime.current()
ones64 = oa.matrix.ones([64, 128], None)
twos64 = oa.matrix.full([64, 128], 2.0, None)
added = oa.matrix.add(ones64, twos64)
runtime.checkpoint().wait()

added_values = added.read_f32()
assert added.shape == [64, 128] and added_values[0] == 3.0
print("add(ones, full(2))[0] =", added_values[0])

a = oa.Matrix.new_from_f32([2, 3], [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], None)
b = oa.matrix.ones([2, 3], None)

# Element-wise arithmetic via the stateless module API.
operator_result = oa.matrix.clamp_min(
	oa.matrix.add_scalar(
		oa.matrix.scale(
			oa.matrix.add(a, b),
			0.5,
		),
		0.5,
	),
	0.0,
)

# Operator overloads express the same computation more concisely.
# Both paths share the same GPU kernels.
operator_alt = (a + b)
operator_alt = oa.matrix.scale(operator_alt, 0.5)
operator_alt = oa.matrix.add_scalar(operator_alt, 0.5)
operator_alt = oa.matrix.clamp_min(operator_alt, 0.0)

# reshape is a zero-cost semantic view; read_f32 is the explicit
# synchronization and device-to-host boundary.
reshaped = operator_result.reshape([3, 2])
values_a = operator_result.read_f32()
values_b = operator_alt.read_f32()

assert operator_result.shape == [2, 3]
assert reshaped.shape == [3, 2]
assert values_a == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
assert values_b == values_a

print(operator_result.shape, values_a)

# Matmul: C = A @ B^T  (FP32 [M,K] × [N,K] -> [M,N])
one = oa.matrix.ones([2, 3], None)
two = oa.matrix.full([2, 3], 2.0, None)
product = oa.matrix.matmul(one, two)
prod_vals = product.read_f32()

assert product.shape == [2, 2]
assert prod_vals == [6.0] * 4

print(product.shape, prod_vals)
