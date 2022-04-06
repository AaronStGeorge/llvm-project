#include "PassDetail.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/LoopFusionUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "my-pass"

using namespace mlir;

namespace IDidNotWrite = mlir;

namespace {
struct MyPass : public MyPassBase<MyPass> {
  void runOnFunction() override;
};
} // end anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createMyPass() {
  return std::make_unique<MyPass>();
}

namespace {
/// This stores information related to an affine for op.
struct AffineForOpInfo {
  Operation *op;
  DenseSet<Value> reads;
  DenseSet<Value> writes;

  explicit AffineForOpInfo(Operation *op, DenseSet<Value> reads,
                           DenseSet<Value> writes)
      : op(op), reads(reads), writes(writes) {}
  ~AffineForOpInfo() = default;
  AffineForOpInfo(const AffineForOpInfo &other) = delete;
  AffineForOpInfo(AffineForOpInfo &&other) noexcept = default;
  AffineForOpInfo &operator=(const AffineForOpInfo &other) = delete;
  AffineForOpInfo &operator=(AffineForOpInfo &&other) noexcept = delete;
};
} // namespace

void MyPass::runOnFunction() {
  // Find Affine For Ops.
  std::vector<AffineForOpInfo> forOps;
  unsigned nodeId = 0;
  for (auto &forOp : getFunction().front()) {
    if (!isa<AffineForOp>(forOp)) {
      continue;
    }
    DenseSet<Value> reads;
    DenseSet<Value> writes;
    forOp.walk([&](Operation *op) {
      if (isa<AffineReadOpInterface>(op)) {
        reads.insert(cast<AffineReadOpInterface>(op).getMemRef());
      } else if (isa<AffineWriteOpInterface>(op)) {
        writes.insert(cast<AffineWriteOpInterface>(op).getMemRef());
      }
    });
    AffineForOpInfo info{&forOp, std::move(reads), std::move(writes)};
    forOps.push_back(std::move(info));
    nodeId++;
  }

  // Make all legal fusions. For ops in array will be in the order they appeared
  // in source code so anything after a given node is a potentially viable
  // fusion candidate.
  for (unsigned srcId = 0; srcId < forOps.size(); srcId++) {
    const AffineForOpInfo &srcInfo{forOps.at(srcId)};

    for (unsigned dstId = srcId + 1; dstId < nodeId; dstId++) {
      const AffineForOpInfo &dstInfo{forOps.at(dstId)};

      if (llvm::any_of(dstInfo.reads, [&](Value op) {
            return srcInfo.writes.count(op) > 0;
          })) {
        auto srcAffineForOp = cast<AffineForOp>(srcInfo.op);
        auto dstAffineForOp = cast<AffineForOp>(dstInfo.op);

        ComputationSliceState computationSliceState = ComputationSliceState();
        FusionResult result = IDidNotWrite::canFuseLoops(
            srcAffineForOp, dstAffineForOp, 1, &computationSliceState,
            FusionStrategy::ProducerConsumer);

        if (result.value == FusionResult::Success) {
          IDidNotWrite::fuseLoops(srcAffineForOp, dstAffineForOp,
                                  computationSliceState);
          srcAffineForOp.erase();
        }
      }
    }
  }
}
