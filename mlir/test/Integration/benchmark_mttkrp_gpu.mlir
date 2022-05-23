// RUN: mlir-opt %s \
// RUN:   -gpu-kernel-outlining \
// RUN:   -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)' \
// RUN:   -gpu-to-llvm \
//  | mlir-cpu-runner \
//    --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
//    --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
//    --entry-point-result=void


func.func @main() {
  %i0 = arith.constant 0. : f64
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
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c2, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c6, %block_y = %c1, %block_z = %c1) {
    %val = memref.load %data[%bx, %tx] : memref<2x6xi32>
    %reduced = gpu.all_reduce and %val {} : (i32) -> (i32)
    memref.store %reduced, %sum[%bx] : memref<2xi32>
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