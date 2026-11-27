// Dead tails are validated without leaving instructions or CFG blocks behind.
// RUN: yaml2obj %S/inputs/dead-tails.yaml.wasm -o - | mlir-translate --import-wasm > %t
// RUN: FileCheck %s < %t
// RUN: mlir-opt --raise-wasm-mlir %t | FileCheck %s --check-prefix=CFG

// CHECK-LABEL: wasmssa.func exported @return_tail() -> i32 {
// CHECK-NEXT:    %[[RET:.*]] = wasmssa.const 42 : i32
// CHECK-NEXT:    wasmssa.return %[[RET]] : i32
// CHECK-NEXT:  }

// CHECK-LABEL: wasmssa.func exported @branch_tail
// CHECK-NEXT:    wasmssa.block : {
// CHECK-NEXT:      %[[BR:.*]] = wasmssa.const 23 : i32
// CHECK-NEXT:      wasmssa.branch to level 0 with args(%[[BR]] : i32)
// CHECK-NEXT:    }> ^bb1
// CHECK-NEXT:  ^bb1(%[[RESULT:.*]]: i32):
// CHECK-NEXT:    wasmssa.return %[[RESULT]] : i32
// CHECK-NEXT:  }

// CHECK-LABEL: wasmssa.func exported @else_is_live
// CHECK-NEXT:    {{.*}} = wasmssa.local_get
// CHECK-NEXT:    wasmssa.if
// CHECK-NEXT:      %[[THEN:.*]] = wasmssa.const 11 : i32
// CHECK-NEXT:      wasmssa.branch to level 0 with args(%[[THEN]] : i32)
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[ELSE:.*]] = wasmssa.const 12 : i32
// CHECK-NEXT:      wasmssa.block_return %[[ELSE]] : i32
// CHECK-NEXT:    }> ^bb1
// CHECK-NEXT:  ^bb1(%[[IF_RESULT:.*]]: i32):
// CHECK-NEXT:    wasmssa.return %[[IF_RESULT]] : i32
// CHECK-NEXT:  }

// CFG-LABEL: func.func @return_tail
// CFG:         %[[VALUE:.*]] = arith.constant 42 : i32
// CFG-NEXT:    return %[[VALUE]] : i32
// CFG-NEXT:  }
// CFG-LABEL: func.func @branch_tail
// CFG:         %[[VALUE:.*]] = arith.constant 23 : i32
// CFG-NEXT:    cf.br ^[[EXIT:bb[0-9]+]](%[[VALUE]] : i32)
// CFG:       ^[[EXIT]](%[[ARG:.*]]: i32):
// CFG-NEXT:    return %[[ARG]] : i32
// CFG-NEXT:  }
