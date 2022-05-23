// RUN: mlir-opt %s \
// RUN: --convert-vector-to-scf \
// RUN: --convert-scf-to-cf \
// RUN: --func-bufferize \
// RUN: --arith-bufferize \
// RUN: --finalizing-bufferize \
// RUN: --convert-vector-to-llvm \
// RUN: --convert-memref-to-llvm \
// RUN: --convert-complex-to-standard \
// RUN: --convert-math-to-llvm \
// RUN: --convert-complex-to-llvm \
// RUN: --convert-math-to-libm \
// RUN: --convert-func-to-llvm \
// RUN: --reconcile-unrealized-casts |\
// RUN: TENSOR0="%mlir_integration_test_dir/data/mttkrp_b.tns" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext,%mlir_integration_test_dir/libmlir_runner_utils%shlibext

// OpenMP dialect seems to be not fully baked yet
// mlir-opt %s \
// --convert-scf-to-openmp 

// mlir-opt %s \
// --convert-parallel-loops-to-gpu \
// --gpu-kernel-outlining \
// --pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin)'
// --gpu-to-llvm 


module {
  func.func private @coords(!llvm.ptr<i8>, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func.func private @getTensorFilename(index) -> (!llvm.ptr<i8>)
  func.func private @printMemrefF64(memref<*xf64>) attributes { llvm.emit_c_interface }
  func.func private @printMemrefI64(memref<*xindex>) attributes { llvm.emit_c_interface }
  func.func private @read_coo(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func.func private @values(!llvm.ptr<i8>) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func.func private @rtclock() -> f64
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}

  func.func @output_memref_index(%0 : memref<?xindex>) -> () {
    %unranked = memref.cast %0 : memref<?xindex> to memref<*xindex>
    call @printMemrefI64(%unranked) : (memref<*xindex>) -> ()
    return
  }

  func.func @output_memref_f64(%unranked : memref<*xf64>) -> () {
    call @printMemrefF64(%unranked) : (memref<*xf64>) -> ()
    return
  }

  func.func @mttkrp_coo(%argb_coord_0 : memref<?xindex>,
                   %argb_coord_1 : memref<?xindex>,
                   %argb_coord_2 : memref<?xindex>,
                   %argb_values : memref<?xf64>,
                   %nnz : index,
                   %J : index,
                   %argc: memref<?x?xf64>,
                   %argd: memref<?x?xf64>,
                   %arga: memref<?x?xf64>) -> memref<?x?xf64> {
    // for (i = 0; i < I; i++)
    //   for (j = 0; j < J; j++)
    //     for (k = 0; k < K; k++)
    //       for (l = 0; l < L; l++)
    //         A[i,j] += B[i,k,l]*D[l,j]*C[k,j];
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // scf.for %j = %c0 to %J step %c1 {
    scf.parallel (%j) = (%c0) to (%J) step (%c1) {
      scf.for %i_k = %c0 to %nnz step %c1 {
        %i = memref.load %argb_coord_0[%i_k] : memref<?xindex>
        %k = memref.load %argb_coord_1[%i_k] : memref<?xindex>
        %l = memref.load %argb_coord_2[%i_k] : memref<?xindex>
        %b_i_k_l = memref.load %argb_values[%i_k] : memref<?xf64>

        %a_i_j = memref.load %arga[%i, %j] : memref<?x?xf64>
        %d_l_j = memref.load %argd[%l, %j] : memref<?x?xf64>
        %c_k_j = memref.load %argd[%k, %j] : memref<?x?xf64>
        %0 = arith.mulf %b_i_k_l, %d_l_j : f64
        %1 = arith.mulf %0, %c_k_j : f64
        %2 = arith.addf %1, %a_i_j : f64
        memref.store %2, %arga[%i, %j] : memref<?x?xf64>
      }
    } { mapping = [{processor = 6, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>}] }
    return %arga : memref<?x?xf64>
  }

  //
  // Main driver that reads matrix from file and calls the kernel.
  //
  func.func @entry() {
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
    %b_coord_1 = call @coords(%storage, %c1) : (!llvm.ptr<i8>, index) -> (memref<?xindex>)
    %b_coord_2 = call @coords(%storage, %c2) : (!llvm.ptr<i8>, index) -> (memref<?xindex>)
    %b_values = call @values(%storage) : (!llvm.ptr<i8>) -> (memref<?xf64>)

    // Initialize dense C and D inputs and dense output A.
    %c = memref.alloc(%K, %J) : memref<?x?xf64>
    scf.for %k = %c0 to %K step %c1 {
      scf.for %j = %c0 to %J step %c1 {
        %v0 = arith.muli %k, %J : index
        %v1 = arith.addi %v0, %j : index
        %v2 = arith.index_cast %v1 : index to i32
        %v = arith.sitofp %v2 : i32 to f64
        memref.store %v, %c[%k, %j] : memref<?x?xf64>
      }
    }
    // %unranked_c = memref.cast %c : memref<?x?xf64> to memref<*xf64>
    // call @output_memref_f64(%unranked_c) : (memref<*xf64>) -> ()

    %d = memref.alloc(%L, %J) : memref<?x?xf64>
    scf.for %l = %c0 to %L step %c1 {
      scf.for %j = %c0 to %J step %c1 {
        %v0 = arith.muli %l, %J : index
        %v1 = arith.addi %v0, %j : index
        %v2 = arith.index_cast %v1 : index to i32
        %v = arith.sitofp %v2 : i32 to f64
        memref.store %v, %d[%l, %j] : memref<?x?xf64>
      }
    }
    // %unranked_d = memref.cast %d : memref<?x?xf64> to memref<*xf64>
    // call @output_memref_f64(%unranked_d) : (memref<*xf64>) -> ()

    %a = memref.alloc(%I, %J) : memref<?x?xf64>
    scf.for %i = %c0 to %I step %c1 {
      scf.for %j = %c0 to %J step %c1 {
        memref.store %i0, %a[%i, %j] : memref<?x?xf64>
      }
    }
    // %unranked_a = memref.cast %a : memref<?x?xf64> to memref<*xf64>
    // call @output_memref_f64(%unranked_a) : (memref<*xf64>) -> ()

    %t_start_mttkrp_coo = call @nanoTime() : () -> i64
    // Call kernel.
    %out = call @mttkrp_coo(%b_coord_0, %b_coord_1,
                            %b_coord_2, %b_values,
                            %nnz, %J, %c, %d,
                            %a) : (memref<?xindex>, memref<?xindex>,
                                       memref<?xindex>, memref<?xf64>, index, index, memref<?x?xf64>, 
                                       memref<?x?xf64>, memref<?x?xf64>) 
                                       -> memref<?x?xf64>
    %t_end_mttkrp_coo = call @nanoTime() : () -> i64
    %t_mttkrp_coo = arith.subi %t_end_mttkrp_coo, %t_start_mttkrp_coo: i64

    vector.print %t_mttkrp_coo : i64

    // Expected output from  mttkrp_b.tns:
    // ( ( 16075, 21930, 28505, 35800, 43815 ), ( 10000, 14225, 19180, 24865, 31280 ) )
    %unranked_a = memref.cast %a : memref<?x?xf64> to memref<*xf64>
    call @output_memref_f64(%unranked_a) : (memref<*xf64>) -> ()

    return
  }
}
