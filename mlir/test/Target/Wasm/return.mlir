// RUN: yaml2obj %S/inputs/return.yaml.wasm -o - | mlir-translate --import-wasm | FileCheck %s

/* Source code used to create this test:
(module
  (func (result i32)
    i32.const 42
    return
  )
  (func (param i32) (result i32)
    local.get 0
    (if (result i32)
      (then
        i32.const 42
        return
      )
      (else
        i32.const 99
      )
    )
  )
)
*/

// CHECK-LABEL:   wasmssa.func @func_0() -> i32 {
// CHECK-NEXT:      %[[VALUE:.*]] = wasmssa.const 42 : i32
// CHECK-NEXT:      wasmssa.return %[[VALUE]] : i32
// CHECK-NEXT:    }

// CHECK-LABEL:   wasmssa.func @func_1
// CHECK:           wasmssa.if
// CHECK-NEXT:        %[[THEN:.*]] = wasmssa.const 42 : i32
// CHECK-NEXT:        wasmssa.return %[[THEN]] : i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %[[ELSE:.*]] = wasmssa.const 99 : i32
// CHECK-NEXT:        wasmssa.block_return %[[ELSE]] : i32
