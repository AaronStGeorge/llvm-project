#include "PassDetail.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "my-pass"

using namespace mlir;

namespace {
    /// Loop invariant code motion (LICM) pass.
    struct MyPass
            : public MyPassBase<MyPass> {
        void runOnFunction() override;
    };
} // end anonymous namespace

std::unique_ptr<Pass> mlir::createMyPass() {
    return std::make_unique<MyPass>();
}

void MyPass::runOnFunction() {
    LLVM_DEBUG(llvm::dbgs() << "TACO: Hello from my pass!\n");
}

