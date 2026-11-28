# Record, combine, and multiply matrices in Python

The first OA compute tutorial in Python: obtain the default `oa.Runtime`,
record domain operations with no submit or wait ceremony, compose matrix
expressions with module functions and operator overloads, read results
back, and run the same tiled F32 matmul the Rust walkthrough benchmarks.
The output below is from the Python run with OA 0.18.0 on Intel Iris Xe
with Mesa 26.3.1.

## Run in Python

From the OASDK root, with the sibling `oa-dev` checkout present, install its
native package into an activated virtual environment with `maturin`.

```sh
(cd ../oa-dev && maturin develop --release)

python src/py/tutorials/core/tu_core_basics/tu_core_basics.py
```

## 1. Record on the default runtime

`oa.Runtime.current()` returns the calling thread's default runtime, lazily
initializing the Vulkan instance, logical device, allocator, and queues on
first use. `oa.matrix.ones`, `oa.matrix.full`, and `oa.matrix.add` record
into its private eager session; the `None` runtime argument on the
constructors resolves the same default. `checkpoint` submits all pending
eager work without blocking the host, and `read_f32` is the explicit device
to host boundary.

```python
runtime = oa.Runtime.current()
ones64 = oa.matrix.ones([64, 128], None)
twos64 = oa.matrix.full([64, 128], 2.0, None)
added = oa.matrix.add(ones64, twos64)
runtime.checkpoint().wait()
```

```text
add(ones, full(2))[0] = 3.0
```

## 2. Compose the same expression two ways

`oa.Matrix.new_from_f32` uploads explicit values. The module functions
record each stage into the eager session: `add`, `scale`, `add_scalar`,
then `clamp_min`. Operator overloads such as `a + b` spell the same kernels
more concisely, and both paths produce identical results because they share
the same GPU kernels.

```python
a = oa.Matrix.new_from_f32([2, 3], [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], None)
b = oa.matrix.ones([2, 3], None)

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

operator_alt = (a + b)
operator_alt = oa.matrix.scale(operator_alt, 0.5)
operator_alt = oa.matrix.add_scalar(operator_alt, 0.5)
operator_alt = oa.matrix.clamp_min(operator_alt, 0.0)
```

## 3. Reshape and read back

`reshape` is a free semantic view and stays on the device. `read_f32` is the
explicit synchronization point: it flushes pending eager work, waits, and
copies device values into a host list.

```python
reshaped = operator_result.reshape([3, 2])
values_a = operator_result.read_f32()
values_b = operator_alt.read_f32()
```

```text
[2, 3] [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
```

## 4. Run the matmul contract

`oa.matrix.matmul` computes the transposed product `[M, K] × [N, K]` into
`[M, N]`. A `[2, 3]` matrix of ones times a `[2, 3]` matrix of twos gives
`[2, 2]` of sixes, matching the Rust run.

```python
one = oa.matrix.ones([2, 3], None)
two = oa.matrix.full([2, 3], 2.0, None)
product = oa.matrix.matmul(one, two)
prod_vals = product.read_f32()
```

```text
[2, 2] [6.0, 6.0, 6.0, 6.0]
```

The Rust tutorial checks the same matmul against an independent CPU
reference and times eager and compiled replay execution.

## Source

[Python example](tu_core_basics.py) ·
[Full Rust walkthrough](../../../../rs/tutorials/core/tu_core_basics/tu_core_basics.md)
