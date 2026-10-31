# tungsten-core: a guide

This file explains what the crate is for, how its parts fit together, where to start reading,
and why it is built the way it is ([Design decisions](#design-decisions)). The lowering that
produces the core IR is described in [`../tungsten-check/README.md`](../tungsten-check/README.md).

## What this crate is for

Tungsten programs have a rich representation (`tungsten-ir`'s `OptimizedProgram`). It is
built for compiling: generational `Ref`s, arenas, names everywhere, convenience variants such
as `ConstructExtern`, and annotations the backends rely on.

That representation is awkward to *reason about*. The long-term plan is to write a checker
(and later an interpreter) and **prove** things about them with
[Verus](https://github.com/verus-lang/verus). Proofs get harder with every enum variant, every
string and every float. So this crate defines a second, much plainer form of the same program:
the **core IR**.

You never write the core IR by hand. `tungsten-check` produces it from an `OptimizedProgram`
(see its `README.md`). This crate only holds:

1. the data types (`ir.rs`, `registry.rs`), and
2. a definition of what "well-formed" means for them (`wf.rs`).

There is no checker, interpreter or proof here yet. Those are later stages.

```
OptimizedProgram ──(tungsten-check::lower)──▶ core_ir::Program ──▶ [future: verified checker,
   (tungsten-ir)                               (this crate)          interpreter, proofs]
```

## The three files

| File | What's in it | Read it when |
| --- | --- | --- |
| `src/ir.rs` | Every data type: ids, `Type`, `Lit`, `Expr`, `Stmt`, `Function`, `Unit`, `Program`, … | Always. Start here. |
| `src/registry.rs` | `ExternDecl` and `RecordDecl`: the signatures of extern functions and the field types of records | You want to know what `Program.externs` / `Program.records` hold |
| `src/wf.rs` | `spec fn wf_program` and its helpers: "every id points at something that exists" | You want to know what a lowered program is guaranteed to satisfy |

`lib.rs` only declares the modules and holds one sanity test.

## Reading `ir.rs`

Read it **bottom-up**: look at `Program` (near the end) first, then follow its fields.

### `Program` is a set of tables

```rust
pub struct Program {
    pub units: Vec<Unit>,           // compute units (density functions)
    pub functions: Vec<Function>,   // every function, main or helper
    pub consts: Vec<ConstDecl>,     // constant initializers
    pub hosts: Vec<HostDecl>,       // host inputs (values the caller passes in)
    pub tables: Vec<TableDecl>,     // permutation tables / noise samplers
    pub inputs: Vec<InputDecl>,     // density inputs (one unit reading another)
    pub entries: Vec<Entry>,        // what the caller actually evaluates
    pub externs: Vec<ExternDecl>,   // extern function signatures
    pub records: Vec<RecordDecl>,   // record (struct) layouts
}
```

Everything else refers into these tables by **index**. `FnId(3)` means `functions[3]`,
`HostId(0)` means `hosts[0]`, and so on. Every id is a newtype around `usize`, so the compiler
won't let you mix a `FnId` up with a `ConstId`.

`LocalId` is the odd one out: it indexes the *enclosing function's* `locals`, not a global
table.

### How this differs from `tungsten-ir`

If you know `tungsten-ir`, these are the differences that will trip you up:

| In `tungsten-ir` | In the core IR | Why |
| --- | --- | --- |
| `Ref<T>` into an `Arena<T>` (can be stale) | plain `usize` index into a `Vec` | Easy to state "in bounds" and reason about |
| Names on everything | **No names at all** | Verus shouldn't deal with `String`. Names live in `tungsten-check`'s `LowerMap` |
| `f32` / `f64` values | `u32` / `u64` bit patterns (`Lit::F64(bits)`, `F64x3`) | Equality is exact, NaN included; no float axioms needed |
| `Field { base, field: "x", … }` | `Field { base, sel: FieldSel::Vec3(Axis::X), ty }` | The field is resolved to a position, no string lookup |
| `Construct` / `ConstructExtern` | `MakeVec3`, `MakePos3`, `MakeRecord` | One variant per shape |
| `ConstExpr` (a separate type) | the ordinary `Expr` | One expression type to reason about |
| `UnaryOperator::Ref(_)` | gone | It's the identity |
| `Function.parameters` (a list of var refs) | `Function.params: usize` | Parameters are simply `locals[0..params]` |
| A unit's own `density_inputs` arena | a global `inputs` table, each with an `owner` unit | One id space for all inputs |
| `VariableType::Extern("Name")` | `Type::Record(RecordId)` | Resolved against the registry |

### Things that look redundant but are on purpose

Some fields seem unnecessary because they could be derived. They are kept so that the future
checker can **verify** them rather than trust them:

- `ExternCall { params, ret, .. }`: the types the call was *annotated* with. The extern's real
  signature is in `externs[ext]`. The checker will compare the two.
- `Field { ty, .. }`: the annotated field type.
- `Unit { helpers, tables, hosts, consts, inputs }` and `Function { hosts }`: scoping lists.
  They say which things a unit or function may use. Nothing enforces them yet.
- `MakeRecord { fields }` stays in the order the fields were *written*, and may have
  duplicates or gaps. "Every field exactly once" is a checker rule.

The general rule: **if something can be wrong, it stays in the IR so the checker can catch
it.** The lowering doesn't quietly fix or drop it.

### Small things worth knowing

- `Dims` holds `i32`s and is not validated. "At least 1" is a checker rule.
- `ConstDecl` has no declared type. The checker will derive it from `init`.
- A constant's `init` is an ordinary `Expr`, but must not contain `Local`, `Call`, `Density`,
  and so on. `wf` already catches `Local` there (see below). The rest is for the checker.
- `Stmt::Repeat { count }` is a fixed iteration count. `Break` can appear anywhere
  syntactically; whether it is inside a loop is a checker rule.
- There's no type for "density input": you can only *read* one with `Expr::Density`. It is
  never a value you can store.
- Floats are bits, so the derived `Eq` is exact: NaN literals keep their payload, and `-0.0`
  and `0.0` are different literals.
- There are no exec functions (no constructors, no getters) inside `verus!{}`, because any exec
  fn that indexes a `Vec` there needs bounds proofs. Plain-Rust helpers live in tungsten-check.
- `BinOp` has the same 18 variants as `tungsten_ir::BinaryOperator`, in the same order.

### The `Clone` impls at the bottom

`Type`, `Expr`, `FieldInit` and `Stmt` are recursive. Verus rejects a derived `Clone` on them,
so their `Clone` impls are written by hand **after** the `verus! { … }` block closes
(`} // verus!`). They are plain structural clones; nothing interesting happens there.

## Reading `wf.rs`

`wf` stands for *well-formed*. It answers one question: **does every index point at something
that exists?** It says nothing about types.

The functions are `spec fn`s. They exist only for Verus and are erased in a normal build, so
you can't call them from regular Rust. Read them as precise documentation:

- `wf_program(p)` is the top. It says every unit, function, constant, host, input, entry,
  extern and record is well-formed, plus `wf_input_links`.
- `wf_function(f)`: `params <= locals.len()`, all types wf, the body wf with
  `nlocals = locals.len()`.
- `wf_expr(p, nlocals, e)`: a big `match`, one arm per `Expr` variant, checking ids against
  their tables and recursing into children.
  Constants are checked with `nlocals = 0`, so any `Local` in a constant fails.
- `wf_type`: `Record(r)` needs `r < records.len()`, recursively through `Array`.
- `wf_sel`: `FieldSel::Record(r, i)` needs record `r` to have a field `i`.
- `wf_input_links`: input `i` says its owner is unit `u` **if and only if** unit `u` lists
  input `i`.

What `wf` deliberately does **not** check, because these are checker rules: anything
type-related, exactly-once record fields, scoping lists, acyclicity of calls and constants,
dims, definite assignment, `Break` placement and density-read legality.

Termination: each recursive `spec fn` has `decreases` on its structural argument. Recursion
through `Vec` children under a guarded `forall` is accepted as is (vstd's
`axiom_vec_index_decreases`), so no `decreases_by` helpers were needed. `wf_stmts` isn't
recursive itself; it quantifies over the recursive `wf_stmt`, so no recursion mixes `Seq` and
datatype measures.

`tungsten-check` promises that every program it returns satisfies `wf_program`. Since you can't
run a `spec fn`, its tests use a hand-written plain-Rust copy instead
(`crates/tungsten-check/src/tests/wf_mirror.rs`). **If you change `wf.rs`, change the mirror
too.**

### Reading Verus syntax

Quick decoder for the bits that aren't ordinary Rust:

| Syntax | Meaning |
| --- | --- |
| `verus! { … }` | Verus code. Normal `cargo build` strips the ghost parts and keeps the data types |
| `pub open spec fn` | A mathematical definition, not executable code. `open` means other modules can see its body |
| `&&& a &&& b` | Just `a && b`, written as a bulleted list |
| `forall\|k: int\| #![trigger xs[k]] 0 <= k < xs.len() ==> P(xs[k])` | "P holds for every element of `xs`". The trigger is a hint for the solver; ignore it when reading |
| `decreases e` | Proof that the recursion terminates: each call is on a smaller `e` |
| `xs@` | The `Vec` viewed as a mathematical sequence (`Seq`) |
| `nat`, `int` | Mathematical naturals and integers (no overflow) |

## Building and verifying

- `cargo build` / `cargo test`: normal Rust. Verus isn't needed; the ghost code is erased.
- `scripts/verify.sh` (from the repo root): actually runs Verus on this crate. It downloads a
  pinned Verus into `target/`, checks the `vstd` pin matches, and **refuses** `admit(`,
  `assume(`, `external_body` or `#[verifier::external]` in this crate, so nothing can be
  "trusted without proof". If the bundled z3 won't start (it needs glibc 2.38), set
  `VERUS_Z3_PATH` to a Z3 4.16.0 binary. Verus rejects any other Z3 version unless you pass
  `-- -V no-solver-version-check`.
- Pins: `VERUS_TAG = 0.2026.10.26.d17fbd0` goes with `vstd =0.0.0-2026-10-25-0158` and Rust
  1.99.1 (installed by the script, selected through `RUSTUP_TOOLCHAIN`). The pairing comes from
  Verus's own `source/cargo-verus/toolchain-manifests/<tag>.toml`. Bump them together; the
  script's pin check enforces the `vstd` part. `cargo verus` needs no direct `verus_builtin*`
  pins.
- Expected Verus output: one lint warning, `inconsistent_fields` on `Expr` (`index` is
  `Option<Box<Expr>>` in `Density` and `Box<Expr>` in `Index`). It only disables `->index`
  syntax for that field. `autoderive_clone_without_spec` is silenced in `lib.rs`.
- The "no trusted gaps" rule allows `decreases` clauses and minimal termination helpers. It
  forbids lemmas, `ensures` on exec fns, loop invariants, `assume`, `admit` and
  `external_body`. A future `prim.rs` (trusted float/extern primitives) will be exempted.

## When you change something

- **Adding an `Expr` / `Stmt` / `Type` variant:** update the hand-written `Clone` at the bottom
  of `ir.rs`, the matching arm in `wf.rs`, the mirror in `tungsten-check`'s `wf_mirror.rs`, and
  the lowering in `tungsten-check/src/lower/`.
- **Adding a table to `Program`:** add a new id type, a `wf_*` clause in `wf_program`, and the
  lowering in `tungsten-check/src/lower/program.rs`.
- Run `scripts/verify.sh` afterwards. It's the acceptance test for this crate.

## Design decisions

Each line gives the choice and the alternative that was rejected, so it doesn't get
"simplified" back.

| Topic | Decision | Rejected, and why |
| --- | --- | --- |
| Scope | Data types, declaration tables and index-bounds `spec fn`s here; the lowering in tungsten-check. | Only `ir.rs`: the IR wouldn't be self-describing and nothing would exercise it. |
| Float literals | Raw bits (`F32(u32)`, `F64(u64)`), scales too. | Native floats: derived `PartialEq` breaks on NaN, and spec reasoning needs float axioms. |
| Normalization | Aggressive: resolved `FieldSel`, `MakeVec3`/`MakePos3`/`MakeRecord`, `ConstExpr` folded into `Expr`, `Ref` dropped, params are locals `0..k`. | A 1:1 mirror of tungsten-ir: more cases in every later proof. |
| Annotations | Kept (`ExternCall` params/ret, `Field.ty`, construct record ids); the checker verifies them. | Lowering checks and drops them: more trusted code. |
| Scoping | Units and functions keep their scoping lists with global ids; the checker enforces them. | Flattening: scoping bugs would only show up as backend failures. |
| Density inputs | One global `InputId` space with `owner: UnitId`. A function reading an input owned by `U` may only be listed in `U`. | A per-unit index resolved against the evaluating unit: silently reinterprets cross-unit reads. |
| Declaration tables | `externs`/`records` live in `Program`, so the checker needs only `&Program`. | A separate registry argument to the checker. |
| Names | Only in plain-Rust side tables (`Registry`, `LowerMap`). | Names inside core types: Verus would see `String`. |
| Derives | Derived `Clone`/`Debug`/`Eq`/… inside `verus!{}`, except a hand-written `Clone` for the recursive types (Verus rejects the derive as a possible cycle). | — |
| No proofs yet | Only termination (`decreases`). | `external_body` gaps, or reshaping the IR around tooling. |
| Verus source | crates.io `vstd`, `=`-pinned, run with `cargo verus`. | A git revision and calling `verus` directly. |
| Verus binary | `verify.sh` downloads the pinned release (sha256-checked) to `target/verus/<tag>/`. | Requiring it on `PATH`. |
| Re-export | Reachable as `tungsten_check::core_ir`. | As `core`: shadows `::core` in derive output. |
| CI | Verus runs locally only. | Gating CI on it. |

## What's coming later

Later stages (see `TESTING.md`) add a value domain (`value.rs`), `prim.rs`, spec semantics, a
verified checker (`check.rs`, `well_typed`), executable code (`exec/`) and proofs
(`proofs/`). The goal is that the checker's `Ok` implies the interpreter never gets stuck.

Open questions:

- Whether `CompileError` should become `#[non_exhaustive]`, since `Invalid` is feature-gated.
- Core IR snapshots (TESTING.md L2), deferred to the testsuite.
