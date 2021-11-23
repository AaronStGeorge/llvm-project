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

namespace {
/// Loop invariant code motion (LICM) pass.
struct MyPass : public MyPassBase<MyPass> {
  void runOnFunction() override;
};
} // end anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createMyPass() {
  return std::make_unique<MyPass>();
}

namespace {
struct Node {
  unsigned nodeId;
  Operation *op;
  SmallVector<Operation *> loads;
  SmallVector<Operation *> stores;

  explicit Node(unsigned nodeId, Operation *op, SmallVector<Operation *> loads,
                SmallVector<Operation *> stores)
      : nodeId(nodeId), op(op), loads(loads), stores(stores) {}
  ~Node() = default;
  Node(const Node &other) = delete;
  Node(Node &&other) noexcept = default;
  Node &operator=(const Node &other) = delete;
  Node &operator=(Node &&other) noexcept = delete;
};
} // namespace

void MyPass::runOnFunction() {
  auto f{getFunction()};
  DenseMap<unsigned, Node> nodes;
  unsigned nodeId = 0;
  for (auto &forOp : f.front()) {
    if (!isa<AffineForOp>(forOp)) {
      continue;
    }
    SmallVector<Operation *, 4> loads;
    SmallVector<Operation *, 4> stores;
    forOp.walk([&](Operation *op) {
      if (isa<AffineReadOpInterface>(op)) {
        loads.push_back(op);
      } else if (isa<AffineWriteOpInterface>(op)) {
        stores.push_back(op);
      }
    });
    Node n{nodeId, &forOp, std::move(loads), std::move(stores)};
    nodes.insert({nodeId, std::move(n)});
    nodeId++;
  }

  DenseMap<unsigned, unsigned> producerConsumerCandidates;
  for (auto &idAndNode : nodes) {
    unsigned srcId{idAndNode.first};
    const Node &srcNode{idAndNode.second};
    DenseSet<Value> memRefs;
    for (Operation *store : srcNode.stores) {
      memRefs.insert(cast<AffineWriteOpInterface>(store).getMemRef());
    }
    for (unsigned dstId = srcId + 1; dstId < nodeId; dstId++) {
      const Node &dstNode{nodes.find(dstId)->second};
      if (llvm::any_of(dstNode.loads, [&](Operation *op) {
            auto loadOp = cast<AffineReadOpInterface>(op);
            return memRefs.count(loadOp.getMemRef()) > 0;
          })) {
        auto srcAffineForOp = cast<AffineForOp>(srcNode.op);
        auto dstAffineForOp = cast<AffineForOp>(dstNode.op);
        ComputationSliceState computationSliceState = ComputationSliceState();
        FusionResult result = mlir::canFuseLoops(
            srcAffineForOp, dstAffineForOp, 1, &computationSliceState,
            FusionStrategy::ProducerConsumer);
        if (result.value == FusionResult::Success) {
          fuseLoops(srcAffineForOp, dstAffineForOp, computationSliceState);
          srcAffineForOp.erase();
        }
      }
    }
  }
}
