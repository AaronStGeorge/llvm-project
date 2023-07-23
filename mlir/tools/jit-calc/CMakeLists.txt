set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_mlir_tool(jit-calc
  jit-calc.cpp
  )
llvm_update_compile_flags(jit-calc)
target_link_libraries(jit-calc PRIVATE
  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRExecutionEngine
  MLIRIR
  MLIRJitRunner
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  MLIRParser
  MLIRTargetLLVMIRExport
  MLIRSupport
  )
