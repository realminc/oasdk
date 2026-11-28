# Verify device admission in Rust

The smallest possible smoke test for a working OA install: enumerate or
select a Vulkan device, capture a minimal matrix operation into an
execution plan, submit it, and print structured device and execution
evidence. Exit code zero confirms the device is admitted; any runtime or
dispatch failure propagates a nonzero exit. Set `OA_VK_VALIDATION=1` to
enable Vulkan validation layers. The output below is from the Rust run with
OA 0.18.0 on Intel Iris Xe with Mesa 26.3.1.

## Run in Rust

From the OASDK root, with `oa-dev` as a sibling checkout:

```sh
cargo run --release --example tu_core_device_canary
```

Pass `--list-devices` to print every Vulkan device OA enumerates, or
`--device INDEX` to force a specific one.

```text
$ cargo run --release --example tu_core_device_canary -- --list-devices
[0] Intel(R) Iris(R) Xe Graphics (ADL GT2) · Integrated · Vulkan 1.4.363
[1] llvmpipe (LLVM 23.1.5, 256 bits) · Cpu · Vulkan 1.4.363
```

## 1. Build the runtime on the selected device

`Runtime::builder().devices(...)` accepts `DeviceSelection::Automatic` or an
explicit index. `Runtime::set_default` installs the built runtime so the
`matrix::` module functions, which record on the default runtime, produce
work that `capture` can see.

```rust
let runtime = Runtime::builder().devices(selection).build()?;
Runtime::set_default(runtime.clone());
```

## 2. Capture a minimal plan

`runtime.capture` records `ones + full(2) → add` as an execution plan
rather than running it eagerly. `submit` returns an `Event`; `wait`
completes the device work without a host readback.

```rust
let (plan, result) = runtime.capture(|| {
	let a = matrix::ones([64, 128])?;
	let b = matrix::full([64, 128], 2.0_f32)?;
	matrix::add(&a, &b)
})?;
runtime.submit(&plan)?.wait()?;
```

## 3. Inspect the diagnostics

`plan.diagnostics()` counts nodes and barriers in the compiled plan and
`read_f32` verifies the result. One `matrix.add.f32` dispatch writes the
output.

```text
OA device canary
  nodes:    1
  barriers: 0
  schema:   1
  add(Ones, full(2))[0] = 3.0
```

## 4. Emit the evidence report

`plan.debug_report_json` prints the full execution plan: compilation
status, dispatch and barrier statistics, per resource byte counts, kernel
selection and content hashes, and the recorded node list. The JSON is
suitable for capture into an evidence bundle or a scheduler admission
record.

```text
{
  "schema": "oa.execution_graph.v3",
  "name": "device_canary",
  "compiled": true,
  "stats": {"dispatches": 1, "barriers": 0, "kernel_fallbacks": 0, ...},
  "nodes": [
    {"index": 0, "operation": "oa::FnMatrix::add", "kernel": "matrix.add.f32",
     "dtype": "float32", "groups": [32, 1, 1], "queue": "compute", ...}
  ]
}
```

The Python bindings do not expose execution plan capture yet, so this
tutorial is Rust only.

## Source

[Rust example](tu_core_device_canary.rs)
