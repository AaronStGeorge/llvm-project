// RUN: mlir-opt %s \
// RUN:   --convert-vector-to-scf --convert-scf-to-std \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize --lower-affine \
// RUN:   --convert-vector-to-llvm --convert-memref-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/mttkrp_b.tns" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext,%mlir_integration_test_dir/libmlir_runner_utils%shlibext


module {
  func private @coords(!llvm.ptr<i8>, index) -> memref<?xindex> attributes {llvm.emit_c_interface}
  func private @getTensorFilename(index) -> (!llvm.ptr<i8>)
  func private @print_memref_f64(memref<*xf64>) attributes { llvm.emit_c_interface }
  func private @print_memref_i64(memref<*xindex>) attributes { llvm.emit_c_interface }
  func private @read_coo(!llvm.ptr<i8>) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  func private @values(!llvm.ptr<i8>) -> memref<?xf64> attributes {llvm.emit_c_interface}
  func private @rtclock() -> f64
  func private @nano_time() -> i64 attributes {llvm.emit_c_interface}

  func @output_memref_index(%0 : memref<?xindex>) -> () {
    %unranked = memref.cast %0 : memref<?xindex> to memref<*xindex>
    call @print_memref_i64(%unranked) : (memref<*xindex>) -> ()
    return
  }

  func @output_memref_f64(%0 : memref<?xf64>) -> () {
    %unranked = memref.cast %0 : memref<?xf64> to memref<*xf64>
    call @print_memref_f64(%unranked) : (memref<*xf64>) -> ()
    return
  }

  func @mttkrp_coo(%argb_coord_0 : memref<?xindex>,
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
    scf.for %i_k = %c0 to %nnz step %c1 {
      scf.for %j = %c0 to %J step %c1 {
        %i = memref.load %argb_coord_0[%i_k] : memref<?xindex>
        %k = memref.load %argb_coord_1[%i_k] : memref<?xindex>
        %l = memref.load %argb_coord_2[%i_k] : memref<?xindex>
        %a_i_j = memref.load %arga[%i, %j] : memref<?x?xf64>
        %b_i_k_l = memref.load %argb_values[%i_k] : memref<?xf64>
        %d_l_j = memref.load %argd[%l, %j] : memref<?x?xf64>
        %c_k_j = memref.load %argd[%k, %j] : memref<?x?xf64>
        %0 = arith.mulf %b_i_k_l, %d_l_j : f64
        %1 = arith.mulf %0, %c_k_j : f64
        %2 = arith.addf %1, %a_i_j : f64
        memref.store %2, %arga[%i, %j] : memref<?x?xf64>
      }
    }
    return %arga : memref<?x?xf64>
  }

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func @entry() {
    %i0 = arith.constant 0. : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // dimensions of matrices
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
    call @output_memref_index(%b_coord_0) : (memref<?xindex>) -> ()
    call @output_memref_index(%b_coord_1) : (memref<?xindex>) -> ()
    call @output_memref_index(%b_coord_2) : (memref<?xindex>) -> ()
    call @output_memref_f64(%b_values) : (memref<?xf64>) -> ()

    // Initialize dense C and D inputs and dense output A.
    %cdata = memref.alloc(%K, %J) : memref<?x?xf64>
    scf.for %k = %c0 to %K step %c1 {
      scf.for %j = %c0 to %J step %c1 {
        %v0 = arith.muli %k, %J : index
        %v1 = arith.addi %v0, %j : index
        %v2 = arith.index_cast %v1 : index to i32
        %v = arith.sitofp %v2 : i32 to f64
        memref.store %v, %cdata[%k, %j] : memref<?x?xf64>
      }
    }
    %c = bufferization.to_tensor %cdata : memref<?x?xf64>

    %ddata = memref.alloc(%L, %J) : memref<?x?xf64>
    scf.for %l = %c0 to %L step %c1 {
      scf.for %j = %c0 to %J step %c1 {
        %v0 = arith.muli %l, %J : index
        %v1 = arith.addi %v0, %j : index
        %v2 = arith.index_cast %v1 : index to i32
        %v = arith.sitofp %v2 : i32 to f64
        memref.store %v, %ddata[%l, %j] : memref<?x?xf64>
      }
    }
    %d = bufferization.to_tensor %ddata : memref<?x?xf64>

    %adata = memref.alloc(%I, %J) : memref<?x?xf64>
    scf.for %i = %c0 to %I step %c1 {
      scf.for %j = %c0 to %J step %c1 {
        memref.store %i0, %adata[%i, %j] : memref<?x?xf64>
      }
    }
    %a = bufferization.to_tensor %adata : memref<?x?xf64>

    %t_start_mttkrp_coo = call @nano_time() : () -> i64
    // Call kernel.
    %out = call @mttkrp_coo(%b_coord_0, %b_coord_1,
                            %b_coord_2, %b_values,
                            %nnz, %J, %cdata, %ddata,
                            %adata) : (memref<?xindex>, memref<?xindex>,
                                       memref<?xindex>, memref<?xf64>,
                                       index, index, memref<?x?xf64>, 
                                       memref<?x?xf64>, memref<?x?xf64>) 
                                       -> memref<?x?xf64>
    %t_end_mttkrp_coo = call @nano_time() : () -> i64
    %t_mttkrp_coo = arith.subi %t_end_mttkrp_coo, %t_start_mttkrp_coo: i64

    vector.print %t_mttkrp_coo : i64

    %v = vector.transfer_read %out[%c0, %c0], %i0
          : memref<?x?xf64>, vector<2x5xf64>
    vector.print %v : vector<2x5xf64>

    return
  }
}
