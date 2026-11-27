// Import unconditional branches and check where they go after CFG lowering.
// RUN: yaml2obj %S/inputs/branch.yaml.wasm -o - | mlir-translate --import-wasm > %t.mlir
// RUN: FileCheck %s < %t.mlir
// RUN: mlir-opt --raise-wasm-mlir %t.mlir | FileCheck %s --check-prefix=CFG

/* Jump past both blocks, carrying 42 to the outer block's continuation.
(func (export "outer") (result i32)
  (block $outer (result i32)
    (block (result i32)
      i32.const 42
      br $outer)))
*/

// CHECK-LABEL:   wasmssa.func exported @outer() -> i32 {
// CHECK-NEXT:      wasmssa.block : {
// CHECK-NEXT:        wasmssa.block : {
// CHECK-NEXT:          %[[VALUE:.*]] = wasmssa.const 42 : i32
// CHECK-NEXT:          wasmssa.branch to level 1 with args(%[[VALUE]] : i32)
// CFG-LABEL:     func.func @outer
// CFG:               %[[VALUE:.*]] = arith.constant 42 : i32
// CFG-NEXT:          cf.br ^[[OUTER_EXIT:bb[0-9]+]](%[[VALUE]] : i32)
// CFG:           ^[[OUTER_EXIT]](%[[RESULT:[0-9]+]]: i32):
// CFG-NEXT:        return %[[RESULT]] : i32

/* Jump back to the loop header with the same loop argument.
(func (export "backedge") (result i32)
  i32.const 7
  (loop $again (param i32) (result i32)
    br $again))
*/

// CHECK-LABEL:   wasmssa.func exported @backedge() -> i32 {
// CHECK:           wasmssa.loop
// CHECK-NEXT:      ^bb0(%[[ARG:.*]]: i32):
// CHECK-NEXT:        wasmssa.branch to level 0 with args(%[[ARG]] : i32)
// CFG-LABEL:     func.func @backedge
// CFG:           ^[[LOOP_HEADER:bb[0-9]+]](%[[ARG:[0-9]+]]: i32):
// CFG-NEXT:        cf.br ^[[LOOP_HEADER]](%[[ARG]] : i32)

/* At function scope, br 0 targets the implicit function label.
(func (export "function_label") (result i32)
  i32.const 9
  br 0)
*/

// CHECK-LABEL:   wasmssa.func exported @function_label() -> i32 {
// CHECK-NEXT:      %[[VALUE:.*]] = wasmssa.const 9 : i32
// CHECK-NEXT:      wasmssa.return %[[VALUE]] : i32
// CHECK-NEXT:    }
// CFG-LABEL:     func.func @function_label
// CFG:             %[[VALUE:.*]] = arith.constant 9 : i32
// CFG-NEXT:        return %[[VALUE]] : i32

/* The then arm falls through; the else arm explicitly exits the if.
(func (export "if_else") (param i32) (result i32)
  local.get 0
  (if $done (result i32)
    (then i32.const 12)
    (else
      i32.const 13
      br $done)))
*/

// CHECK-LABEL:   wasmssa.func exported @if_else
// CHECK:           wasmssa.if
// CHECK-NEXT:        %[[THEN_VALUE:.*]] = wasmssa.const 12 : i32
// CHECK-NEXT:        wasmssa.block_return %[[THEN_VALUE]] : i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[ELSE_VALUE:.*]] = wasmssa.const 13 : i32
// CHECK-NEXT:        wasmssa.branch to level 0 with args(%[[ELSE_VALUE]] : i32)

/* From inside one block, br 1 returns from the function.
(func (export "nested_function_label") (result i32)
  (block (result i32)
    i32.const 10
    br 1))
*/

// CHECK-LABEL:   wasmssa.func exported @nested_function_label() -> i32 {
// CHECK-NEXT:      wasmssa.block : {
// CHECK-NEXT:        %[[VALUE:.*]] = wasmssa.const 10 : i32
// CHECK-NEXT:        wasmssa.return %[[VALUE]] : i32
// CFG-LABEL:     func.func @nested_function_label
// CFG:               %[[VALUE:.*]] = arith.constant 10 : i32
// CFG-NEXT:          return %[[VALUE]] : i32

/* Pass only the block's i32 result, discarding the extra i64.
(func (export "discard_stack_values") (result i32)
  (block $done (result i32)
    i64.const 1
    i32.const 23
    br $done))
*/

// CHECK-LABEL:   wasmssa.func exported @discard_stack_values() -> i32 {
// CHECK-NEXT:      wasmssa.block : {
// CHECK-NEXT:        wasmssa.const 1 : i64
// CHECK-NEXT:        %[[LIVE:.*]] = wasmssa.const 23 : i32
// CHECK-NEXT:        wasmssa.branch to level 0 with args(%[[LIVE]] : i32)
// CHECK-NEXT:      }> ^bb1
// CHECK-NEXT:    ^bb1(%[[RESULT:.*]]: i32):
// CHECK-NEXT:      wasmssa.return %[[RESULT]] : i32
// CHECK-NEXT:    }

/* The then arm branches out; the else arm still parses and falls through.
(func (export "branch_before_else") (param i32) (result i32)
  local.get 0
  (if $done (result i32)
    (then
      i32.const 42
      br $done)
    (else
      i32.const 99)))
*/

// CHECK-LABEL:   wasmssa.func exported @branch_before_else
// CHECK:           wasmssa.if
// CHECK-NEXT:        %[[THEN_VALUE:.*]] = wasmssa.const 42 : i32
// CHECK-NEXT:        wasmssa.branch to level 0 with args(%[[THEN_VALUE]] : i32)
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[ELSE_VALUE:.*]] = wasmssa.const 99 : i32
// CHECK-NEXT:        wasmssa.block_return %[[ELSE_VALUE]] : i32
