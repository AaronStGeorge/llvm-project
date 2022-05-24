// RUN: mlir-opt %s \
// RUN:   -convert-vector-to-scf \
// RUN:   -convert-scf-to-cf \
// RUN:   -func-bufferize \
// RUN:   -arith-bufferize \
// RUN:   -finalizing-bufferize \
// RUN:   -gpu-kernel-outlining \
// RUN:   -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
// RUN:   -gpu-to-llvm \
// RUN:   -convert-vector-to-llvm \
// RUN:   -convert-memref-to-llvm \
// RUN:   -convert-complex-to-standard \
// RUN:   -convert-math-to-llvm \
// RUN:   -convert-complex-to-llvm \
// RUN:   -convert-math-to-libm \
// RUN:   -convert-func-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN: | TENSOR0="%mlir_integration_test_dir/data/mttkrp_b.tns" mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_integration_test_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext \
// RUN:   --shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void


func.func @main() {
  %i0 = arith.constant 0.0 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // dimensions of matrices for nell-2-modified
  // %I = arith.constant 12092 : index
  // %J = arith.constant 5000 : index
  // %K = arith.constant 9184 : index
  // %L = arith.constant 28818 : index
  // %nnz = arith.constant 5879419 : index
  // dimensions of matrices for mttkrp_b 
  %I = arith.constant 2 : index
  %J = arith.constant 5 : index
  %K = arith.constant 3 : index
  %L = arith.constant 4 : index
  %nnz = arith.constant 17 : index

  // Read the sparse B input from a file.
  %filename = call @getTensorFilename(%c0) : (index) -> !llvm.ptr<i8>
  %storage = call @read_coo(%filename) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>

  %b_coord_0 = call @coords(%storage, %c0) : (!llvm.ptr<i8>, index) -> (memref<?xindex>)
  %cast_b_coord_0 = memref.cast %b_coord_0 : memref<?xindex> to memref<*xindex>
  gpu.host_register %cast_b_coord_0 : memref<*xindex>

  %b_coord_1 = call @coords(%storage, %c1) : (!llvm.ptr<i8>, index) -> (memref<?xindex>)
  %cast_b_coord_1 = memref.cast %b_coord_1 : memref<?xindex> to memref<*xindex>
  gpu.host_register %cast_b_coord_1 : memref<*xindex>

  %b_coord_2 = call @coords(%storage, %c2) : (!llvm.ptr<i8>, index) -> (memref<?xindex>)
  %cast_b_coord_2 = memref.cast %b_coord_2 : memref<?xindex> to memref<*xindex>
  gpu.host_register %cast_b_coord_2 : memref<*xindex>

  %b_values = call @values(%storage) : (!llvm.ptr<i8>) -> (memref<?xf64>)
  %cast_b_values = memref.cast %b_values : memref<?xf64> to memref<*xf64>
  gpu.host_register %cast_b_values : memref<*xf64>

  // Initialize dense C and D inputs and dense output A.
  %c = memref.alloc(%K, %J) : memref<?x?xf64>
  %cast_c = memref.cast %c : memref<?x?xf64> to memref<*xf64>
  gpu.host_register %cast_c : memref<*xf64>
  scf.for %k = %c0 to %K step %c1 {
    scf.for %j = %c0 to %J step %c1 {
      %v0 = arith.muli %k, %J : index
      %v1 = arith.addi %v0, %j : index
      %v2 = arith.index_cast %v1 : index to i32
      %v = arith.sitofp %v2 : i32 to f64
      memref.store %v, %c[%k, %j] : memref<?x?xf64>
    }
  }

  %d = memref.alloc(%L, %J) : memref<?x?xf64>
  %cast_d = memref.cast %d : memref<?x?xf64> to memref<*xf64>
  gpu.host_register %cast_d : memref<*xf64>
  scf.for %l = %c0 to %L step %c1 {
    scf.for %j = %c0 to %J step %c1 {
      %v0 = arith.muli %l, %J : index
      %v1 = arith.addi %v0, %j : index
      %v2 = arith.index_cast %v1 : index to i32
      %v = arith.sitofp %v2 : i32 to f64
      memref.store %v, %d[%l, %j] : memref<?x?xf64>
    }
  }

  %a = memref.alloc(%I, %J) : memref<?x?xf64>
  %cast_a = memref.cast %a : memref<?x?xf64> to memref<*xf64>
  gpu.host_register %cast_a : memref<*xf64>
  scf.for %i = %c0 to %I step %c1 {
    scf.for %j = %c0 to %J step %c1 {
      memref.store %i0, %a[%i, %j] : memref<?x?xf64>
    }
  }

  // AND
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%j, %ty, %tz) in (%block_x = %J, %block_y = %c1, %block_z = %c1) {
    scf.for %i_k = %c0 to %nnz step %c1 {
      %i = memref.load %b_coord_0[%i_k] : memref<?xindex>
      %k = memref.load %b_coord_1[%i_k] : memref<?xindex>
      %l = memref.load %b_coord_2[%i_k] : memref<?xindex>
      %b_i_k_l = memref.load %b_values[%i_k] : memref<?xf64>

      %a_i_j = memref.load %a[%i, %j] : memref<?x?xf64>
      %d_l_j = memref.load %d[%l, %j] : memref<?x?xf64>
      %c_k_j = memref.load %c[%k, %j] : memref<?x?xf64>
      %0 = arith.mulf %b_i_k_l, %d_l_j : f64
      %1 = arith.mulf %0, %c_k_j : f64
      %2 = arith.addf %1, %a_i_j : f64
      memref.store %2, %a[%i, %j] : memref<?x?xf64>
    }
    gpu.terminator
  }

  call @printMemrefF64(%cast_a) : (memref<*xf64>) -> ()

  return
}

func.func private @printMemrefF64(memref<*xf64>) 
func.func private @printMemrefI32(memref<*xi32>)
func.func private @getTensorFilename(index) -> (!llvm.ptr<i8>)
func.func private @read_coo(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
func.func private @values(!llvm.ptr<i8>) -> memref<?xf64> attributes {llvm.emit_c_interface}
func.func private @coords(!llvm.ptr<i8>, index) -> memref<?xindex> attributes {llvm.emit_c_interface}