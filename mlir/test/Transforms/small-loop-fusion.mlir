// RUN: mlir-opt -allow-unregistered-dialect %s -small-affine-loop-fusion -split-input-file | FileCheck %s

// CHECK-LABEL: func @should_fuse_raw_dep_for_locality() {
func @should_fuse_raw_dep_for_locality() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %m[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %v0 = affine.load %m[%i1] : memref<10xf32>
  }
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}
