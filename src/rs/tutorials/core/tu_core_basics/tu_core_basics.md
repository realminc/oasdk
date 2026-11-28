# Record, combine, and multiply matrices in Rust

The first OA compute tutorial: obtain the default `Runtime`, record domain
operations with no submit or wait ceremony, compose matrix expressions in
every supported spelling, verify tiled F32 matmul against an independent
CPU reference, and measure eager and compiled replay performance. The
output below is from the Rust run with OA 0.18.0 on Intel Iris Xe with
Mesa 26.3.1.

## Run in Rust

From the OASDK root, with `oa-dev` as a sibling checkout:

```sh
cargo run --release --example tu_core_basics
```

The default run covers recording, syntax, correctness, and both performance
sections. Use `--correctness-only` for the oracle sweep alone,
`--benchmark-only` for timing, or `--case NAME --warmup N --samples N` to
focus on one geometry.

## 1. Record on the default runtime

`Runtime::current()` returns this thread's default runtime, lazily
initializing the Vulkan instance, logical device, allocator, and queues on
first use. `matrix::ones`, `matrix::full`, and `matrix::add` record into its
private eager session and return `Matrix` values immediately as semantic
handles. `checkpoint` submits all pending eager work without blocking the
host, and `read_f32` is the explicit device to host boundary.

```rust
let runtime = Runtime::current()?;
let a = matrix::ones([64, 128])?;
let b = matrix::full([64, 128], 2.0_f32)?;
let output = matrix::add(&a, &b)?;
runtime.checkpoint()?.wait()?;
let values = output.read_f32()?;
```

```text
1. Default runtime and eager recording
   add(ones, full(2))[0] = 3.0 · checkpoint flushed, 8192 elements
```

## 2. Basic syntax

`matrix::matmul` computes the transposed product `C = A @ Bᵀ`, that is
`[M, K] × [N, K]` into `[M, N]`. Elementwise spelling is deliberately
separate from the matmul contract: the operator, method, associated call,
and module function forms share one kernel body and agree on the result.

```rust
let one = matrix::ones([2, 3])?;
let two = matrix::full([2, 3], 2.0)?;
let product = matrix::matmul(&one, &two)?;
let values = product.read_f32()?;
```

```text
[2,3] × [2,3]ᵀ -> [2, 2]: [6.0, 6.0, 6.0, 6.0]
operator / method / associated / module calls agree
```

## 3. Correctness against a CPU reference

Eight geometries cover square, tall, wide, irregular, GEMV, and MNIST shaped
problems. Deterministic inputs feed both an independent scalar CPU triple
loop and the device path; the normalized error must stay within FP32
tolerance.

```text
tiny-square      [    8,    8,    8] norm_err=7.26e-8 ok
square           [  128,  128,  128] norm_err=8.33e-8 ok
tall-skinny      [  512,   64,  128] norm_err=8.33e-8 ok
short-wide       [   64,  512,  128] norm_err=8.33e-8 ok
irregular        [  100,  130,   77] norm_err=1.37e-7 ok
gemv-decode      [    1,  256,  256] norm_err=8.33e-8 ok
mnist-hidden     [   64,  128,  784] norm_err=5.44e-8 ok
mnist-logits     [   64,   10,  128] norm_err=8.33e-8 ok
All correctness checks passed.
```

## 4. Eager submit and readback timing

The eager boundary measures a `matmul` call through synchronized full output
readback, including the device to host copy. A constant input oracle checks
every timed geometry before warmup so NaNs or a wrong kernel cannot hide
behind timing alone.

## 5. Compiled replay timing

`Runtime::capture` records the matmul once into an execution plan.
`submit_timed` replays it and reports the device timestamp, with no readback
in the measured region. The same oracle verifies the captured output before
timing starts.

```rust
let (plan, output) = runtime.capture(|| matrix::matmul(&left, &right))?;
runtime.submit(&plan)?.wait()?;
let duration = runtime.submit_timed(&plan)?.device_duration()?;
```

```text
square-1024      [ 1024, 1024, 1024] p50=    3.464 ms p95=   11.229 ms    620.00 GFLOP/s
BENCH name=square-1024 m=1024 n=1024 k=1024 boundary=replay_submit_timed p50_ms=3.463698 p95_ms=11.228802 gflops=619.997369
```

Replay timing reflects the kernel alone. These are exploratory numbers from
one process on one device, not release benchmark evidence.

The Python tutorial runs the same record, expression, and matmul sequence
with the `oa.matrix` module API.

## Source

[Rust example](tu_core_basics.rs) ·
[Full Python walkthrough](../../../../py/tutorials/core/tu_core_basics/tu_core_basics.md)
