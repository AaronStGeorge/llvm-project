#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/LoopFusionUtils.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iomanip>
#include <sstream>
#define DEBUG_TYPE "small-affine-loop-fusion"

using namespace mlir;

namespace {
/// Loop fusion pass. This pass currently supports a greedy fusion policy,
/// which fuses loop nests with single-writer/single-reader memref dependences
/// with the goal of improving locality.

// TODO: Support fusion of source loop nests which write to multiple
// memrefs, where each memref can have multiple users (if profitable).
// TODO: Extend this pass to check for fusion preventing dependences,
// and add support for more general loop fusion algorithms.

struct SmallLoopFusion : public SmallAffineLoopFusionBase<SmallLoopFusion> {
  SmallLoopFusion() = default;
  SmallLoopFusion(unsigned fastMemorySpace, uint64_t localBufSizeThresholdBytes,
                  bool maximalFusion) {}

  void runOnFunction() override;
};

} // end anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createSmallLoopFusionPass() {
  return std::make_unique<SmallLoopFusion>(0, 0, false);
}

namespace {

// LoopNestStateCollector walks loop nests and collects load and store
// operations, and whether or not a region holding op other than ForOp and IfOp
// was encountered in the loop nest.
struct LoopNestStateCollector {
  SmallVector<AffineForOp, 4> forOps;
  SmallVector<Operation *, 4> loadOpInsts;
  SmallVector<Operation *, 4> storeOpInsts;
  bool hasNonAffineRegionOp = false;

  void collect(Operation *opToWalk) {
    opToWalk->walk([&](Operation *op) {
      if (isa<AffineForOp>(op))
        forOps.push_back(cast<AffineForOp>(op));
      else if (op->getNumRegions() != 0 && !isa<AffineIfOp>(op))
        hasNonAffineRegionOp = true;
      else if (isa<AffineReadOpInterface>(op))
        loadOpInsts.push_back(op);
      else if (isa<AffineWriteOpInterface>(op))
        storeOpInsts.push_back(op);
    });
  }
};

// MemRefDependenceGraph is a graph data structure where graph nodes are
// top-level operations in a FuncOp which contain load/store ops, and edges
// are memref dependences between the nodes.
// TODO: Add a more flexible dependence graph representation.
// TODO: Add a depth parameter to dependence graph construction.
struct MemRefDependenceGraph {
public:
  // Node represents a node in the graph. A Node is either an entire loop nest
  // rooted at the top level which contains loads/stores, or a top level
  // load/store.
  struct Node {
    // The unique identifier of this node in the graph.
    unsigned id;
    // The top-level statement which is (or contains) a load/store.
    Operation *op;
    // List of load operations.
    SmallVector<Operation *, 4> loads;
    // List of store op insts.
    SmallVector<Operation *, 4> stores;
    Node(unsigned id, Operation *op) : id(id), op(op) {}

    // Returns the load op count for 'memref'.
    unsigned getLoadOpCount(Value memref) {
      unsigned loadOpCount = 0;
      for (auto *loadOpInst : loads) {
        if (memref == cast<AffineReadOpInterface>(loadOpInst).getMemRef())
          ++loadOpCount;
      }
      return loadOpCount;
    }

    // Returns the store op count for 'memref'.
    unsigned getStoreOpCount(Value memref) {
      unsigned storeOpCount = 0;
      for (auto *storeOpInst : stores) {
        if (memref == cast<AffineWriteOpInterface>(storeOpInst).getMemRef())
          ++storeOpCount;
      }
      return storeOpCount;
    }
  };

  // Edge represents a data dependence between nodes in the graph.
  struct Edge {
    // The id of the node at the other end of the edge.
    // If this edge is stored in Edge = Node.inEdges[i], then
    // 'Node.inEdges[i].id' is the identifier of the source node of the edge.
    // If this edge is stored in Edge = Node.outEdges[i], then
    // 'Node.outEdges[i].id' is the identifier of the dest node of the edge.
    unsigned id;
    // The SSA value on which this edge represents a dependence.
    // If the value is a memref, then the dependence is between graph nodes
    // which contain accesses to the same memref 'value'. If the value is a
    // non-memref value, then the dependence is between a graph node which
    // defines an SSA value and another graph node which uses the SSA value
    // (e.g. a constant or load operation defining a value which is used inside
    // a loop nest).
    Value value;
  };

  // Map from node id to Node.
  DenseMap<unsigned, Node> nodes;
  // Map from node id to list of input edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> inEdges;
  // Map from node id to list of output edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> outEdges;
  // Map from memref to a count on the dependence edges associated with that
  // memref.
  DenseMap<Value, unsigned> memrefEdgeCount;
  // The next unique identifier to use for newly created graph nodes.
  unsigned nextNodeId = 0;

  MemRefDependenceGraph() {}

  // Initializes the dependence graph based on operations in 'f'.
  // Returns true on success, false otherwise.
  bool init(FuncOp f);

  // Returns the graph node for 'id'.
  Node *getNode(unsigned id) {
    auto it = nodes.find(id);
    assert(it != nodes.end());
    return &it->second;
  }

  // Adds a node with 'op' to the graph and returns its unique identifier.
  unsigned addNode(Operation *op) {
    Node node(nextNodeId++, op);
    nodes.insert({node.id, node});
    return node.id;
  }

  // Remove node 'id' (and its associated edges) from graph.
  void removeNode(unsigned id) {
    // Remove each edge in 'inEdges[id]'.
    if (inEdges.count(id) > 0) {
      SmallVector<Edge, 2> oldInEdges = inEdges[id];
      for (auto &inEdge : oldInEdges) {
        removeEdge(inEdge.id, id, inEdge.value);
      }
    }
    // Remove each edge in 'outEdges[id]'.
    if (outEdges.count(id) > 0) {
      SmallVector<Edge, 2> oldOutEdges = outEdges[id];
      for (auto &outEdge : oldOutEdges) {
        removeEdge(id, outEdge.id, outEdge.value);
      }
    }
    // Erase remaining node state.
    inEdges.erase(id);
    outEdges.erase(id);
    nodes.erase(id);
  }

  // Returns true iff there is an edge from node 'srcId' to node 'dstId' which
  // is for 'value' if non-null, or for any value otherwise. Returns false
  // otherwise.
  bool hasEdge(unsigned srcId, unsigned dstId, Value value = nullptr) {
    if (outEdges.count(srcId) == 0 || inEdges.count(dstId) == 0) {
      return false;
    }
    bool hasOutEdge = llvm::any_of(outEdges[srcId], [=](Edge &edge) {
      return edge.id == dstId && (!value || edge.value == value);
    });
    bool hasInEdge = llvm::any_of(inEdges[dstId], [=](Edge &edge) {
      return edge.id == srcId && (!value || edge.value == value);
    });
    return hasOutEdge && hasInEdge;
  }

  // Adds an edge from node 'srcId' to node 'dstId' for 'value'.
  void addEdge(unsigned srcId, unsigned dstId, Value value) {
    if (!hasEdge(srcId, dstId, value)) {
      outEdges[srcId].push_back({dstId, value});
      inEdges[dstId].push_back({srcId, value});
      if (value.getType().isa<MemRefType>())
        memrefEdgeCount[value]++;
    }
  }

  // Removes an edge from node 'srcId' to node 'dstId' for 'value'.
  void removeEdge(unsigned srcId, unsigned dstId, Value value) {
    assert(inEdges.count(dstId) > 0);
    assert(outEdges.count(srcId) > 0);
    if (value.getType().isa<MemRefType>()) {
      assert(memrefEdgeCount.count(value) > 0);
      memrefEdgeCount[value]--;
    }
    // Remove 'srcId' from 'inEdges[dstId]'.
    for (auto it = inEdges[dstId].begin(); it != inEdges[dstId].end(); ++it) {
      if ((*it).id == srcId && (*it).value == value) {
        inEdges[dstId].erase(it);
        break;
      }
    }
    // Remove 'dstId' from 'outEdges[srcId]'.
    for (auto it = outEdges[srcId].begin(); it != outEdges[srcId].end(); ++it) {
      if ((*it).id == dstId && (*it).value == value) {
        outEdges[srcId].erase(it);
        break;
      }
    }
  }

  // Adds ops in 'loads' and 'stores' to node at 'id'.
  void addToNode(unsigned id, const SmallVectorImpl<Operation *> &loads,
                 const SmallVectorImpl<Operation *> &stores) {
    Node *node = getNode(id);
    for (auto *loadOpInst : loads)
      node->loads.push_back(loadOpInst);
    for (auto *storeOpInst : stores)
      node->stores.push_back(storeOpInst);
  }

  void clearNodeLoadAndStores(unsigned id) {
    Node *node = getNode(id);
    node->loads.clear();
    node->stores.clear();
  }

  // Calls 'callback' for each input edge incident to node 'id' which carries a
  // memref dependence.
  void forEachMemRefInputEdge(unsigned id,
                              const std::function<void(Edge)> &callback) {
    if (inEdges.count(id) > 0)
      forEachMemRefEdge(inEdges[id], callback);
  }

  // Calls 'callback' for each output edge from node 'id' which carries a
  // memref dependence.
  void forEachMemRefOutputEdge(unsigned id,
                               const std::function<void(Edge)> &callback) {
    if (outEdges.count(id) > 0)
      forEachMemRefEdge(outEdges[id], callback);
  }

  // Calls 'callback' for each edge in 'edges' which carries a memref
  // dependence.
  void forEachMemRefEdge(ArrayRef<Edge> edges,
                         const std::function<void(Edge)> &callback) {
    for (const auto &edge : edges) {
      // Skip if 'edge' is not a memref dependence edge.
      if (!edge.value.getType().isa<MemRefType>())
        continue;
      assert(nodes.count(edge.id) > 0);
      // Skip if 'edge.id' is not a loop nest.
      if (!isa<AffineForOp>(getNode(edge.id)->op))
        continue;
      // Visit current input edge 'edge'.
      callback(edge);
    }
  }

  void printGViz(raw_ostream &os) const {
    os << "digraph G {"
       << "\n";
    for (const auto &idAndNode : nodes) {
      auto it = inEdges.find(idAndNode.first);
      it = outEdges.find(idAndNode.first);
      if (it != outEdges.end()) {
        for (const auto &e : it->second) {
          os << idAndNode.first << " -> " << e.id << " [label =\"" << e.value
             << "\"];"
             << "\n";
        }
      }
    }
    os << "}"
       << "\n";
  }

  void printNodesInOrder(raw_ostream &os) const {
    for (unsigned i = 0; i < nodes.size(); i++) {
      for (const auto &pair : nodes) {
        if (pair.first == i) {
          os << "Node: " << pair.first << "\n";
          pair.getSecond().op->dump();
        }
      }
    }
  }

  void print(raw_ostream &os) const {
    os << "\nMemRefDependenceGraph\n";
    os << "\nNodes:\n";
    for (const auto &idAndNode : nodes) {
      os << "Node: " << idAndNode.first << "\n";
      auto it = inEdges.find(idAndNode.first);
      if (it != inEdges.end()) {
        for (const auto &e : it->second)
          os << "  InEdge: " << e.id << " " << e.value << "\n";
      }
      it = outEdges.find(idAndNode.first);
      if (it != outEdges.end()) {
        for (const auto &e : it->second)
          os << "  OutEdge: " << e.id << " " << e.value << "\n";
      }
    }
  }
  void dump() const { print(llvm::errs()); }
  void dumpGViz() const { printGViz(llvm::errs()); }
  void dumpNodesInOrder() const { printNodesInOrder(llvm::errs()); }
};

/// TODO: this returns a (potentially empty) list of AffineForOps parent to this
/// op in the mdg that contain a store into the same memref that dstID contains
/// a load.

/// Returns in 'srcIdCandidates' the producer fusion candidates for consumer
/// 'dstId'. Candidates are sorted by node id order. This order corresponds to
/// the program order when the 'mdg' is created. However, program order is not
/// guaranteed and must not be required by the client. Program order won't be
/// held if the 'mdg' is reused from a previous fusion step or if the node
/// creation order changes in the future to support more advanced cases.
// TODO: Move this to a loop fusion utility once 'mdg' is also moved.
static void getProducerCandidates(unsigned dstId, MemRefDependenceGraph *mdg,
                                  SmallVectorImpl<unsigned> &srcIdCandidates) {
  // Skip if no input edges along which to fuse.
  if (mdg->inEdges.count(dstId) == 0)
    return;

  // Gather memrefs from loads in 'dstId'.
  auto *dstNode = mdg->getNode(dstId);
  DenseSet<Value> consumedMemrefs;
  for (Operation *load : dstNode->loads)
    consumedMemrefs.insert(cast<AffineReadOpInterface>(load).getMemRef());

  // Traverse 'dstId' incoming edges and gather the nodes that contain a store
  // to one of the consumed memrefs.
  for (auto &srcEdge : mdg->inEdges[dstId]) {
    auto *srcNode = mdg->getNode(srcEdge.id);
    // Skip if 'srcNode' is not a loop nest.
    if (!isa<AffineForOp>(srcNode->op))
      continue;

    if (any_of(srcNode->stores, [&](Operation *op) {
          auto storeOp = cast<AffineWriteOpInterface>(op);
          return consumedMemrefs.count(storeOp.getMemRef()) > 0;
        }))
      srcIdCandidates.push_back(srcNode->id);
  }

  std::sort(srcIdCandidates.begin(), srcIdCandidates.end());
  // TODO: what the fuck does this shit below do?
  srcIdCandidates.erase(
      std::unique(srcIdCandidates.begin(), srcIdCandidates.end()),
      srcIdCandidates.end());
}


} // end anonymous namespace

// Initializes the data dependence graph by walking operations in 'f'.
// Assigns each node in the graph a node id based on program order in 'f'.
// TODO: Add support for taking a Block arg to construct the
// dependence graph at a different depth.
bool MemRefDependenceGraph::init(FuncOp f) {
  LLVM_DEBUG(llvm::dbgs() << "--- Initializing MDG ---\n");
  DenseMap<Value, SetVector<unsigned>> memrefAccesses;

  // TODO: support multi-block functions.
  if (!llvm::hasSingleElement(f))
    return false;

  DenseMap<Operation *, unsigned> forToNodeMap;
  for (auto &op : f.front()) {
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      // Create graph node 'id' to represent top-level 'forOp' and record
      // all loads and store accesses it contains.
      LoopNestStateCollector collector;
      collector.collect(&op);
      // Return false if a region holding op other than 'affine.for' and
      // 'affine.if' was found (not currently supported).
      if (collector.hasNonAffineRegionOp)
        return false;
      Node node(nextNodeId++, &op);
      for (auto *opInst : collector.loadOpInsts) {
        node.loads.push_back(opInst);
        auto memref = cast<AffineReadOpInterface>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
      }
      for (auto *opInst : collector.storeOpInsts) {
        node.stores.push_back(opInst);
        auto memref = cast<AffineWriteOpInterface>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
      }
      forToNodeMap[&op] = node.id;
      nodes.insert({node.id, node});
    } else if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
      // Create graph node for top-level load op.
      Node node(nextNodeId++, &op);
      node.loads.push_back(&op);
      auto memref = cast<AffineReadOpInterface>(op).getMemRef();
      memrefAccesses[memref].insert(node.id);
      nodes.insert({node.id, node});
    } else if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      // Create graph node for top-level store op.
      Node node(nextNodeId++, &op);
      node.stores.push_back(&op);
      auto memref = cast<AffineWriteOpInterface>(op).getMemRef();
      memrefAccesses[memref].insert(node.id);
      nodes.insert({node.id, node});
    } else if (op.getNumRegions() != 0) {
      // Return false if another region is found (not currently supported).
      return false;
    } else if (op.getNumResults() > 0 && !op.use_empty()) {
      // Create graph node for top-level producer of SSA values, which
      // could be used by loop nest nodes.
      Node node(nextNodeId++, &op);
      nodes.insert({node.id, node});
    } else if (isa<CallOpInterface>(op)) {
      // Create graph node for top-level Call Op that takes any argument of
      // memref type. Call Op that returns one or more memref type results
      // is already taken care of, by the previous conditions.
      if (llvm::any_of(op.getOperandTypes(),
                       [&](Type t) { return t.isa<MemRefType>(); })) {
        Node node(nextNodeId++, &op);
        nodes.insert({node.id, node});
      }
    } else if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Create graph node for top-level op, which could have a memory write
      // side effect.
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      effectInterface.getEffects(effects);
      if (llvm::any_of(effects, [](const MemoryEffects::EffectInstance &it) {
            return isa<MemoryEffects::Write, MemoryEffects::Free>(
                it.getEffect());
          })) {
        Node node(nextNodeId++, &op);
        nodes.insert({node.id, node});
      }
    }
  }

  for (auto &idAndNode : nodes) {
    LLVM_DEBUG(llvm::dbgs() << "Create node " << idAndNode.first << " for:\n"
                            << *(idAndNode.second.op) << "\n");
    (void)idAndNode;
  }

  // Add dependence edges between nodes which produce SSA values and their
  // users. Load ops can be considered as the ones producing SSA values.
  for (auto &idAndNode : nodes) {
    const Node &node = idAndNode.second;
    // Stores don't define SSA values, skip them.
    if (!node.stores.empty())
      continue;
    auto *opInst = node.op;
    for (auto value : opInst->getResults()) {
      for (auto *user : value.getUsers()) {
        SmallVector<AffineForOp, 4> loops;
        getLoopIVs(*user, &loops);
        if (loops.empty())
          continue;
        assert(forToNodeMap.count(loops[0].getOperation()) > 0);
        unsigned userLoopNestId = forToNodeMap[loops[0].getOperation()];
        addEdge(node.id, userLoopNestId, value);
      }
    }
  }

  // Walk memref access lists and add graph edges between dependent nodes.
  for (auto &memrefAndList : memrefAccesses) {
    unsigned n = memrefAndList.second.size();
    for (unsigned i = 0; i < n; ++i) {
      unsigned srcId = memrefAndList.second[i];
      bool srcHasStore =
          getNode(srcId)->getStoreOpCount(memrefAndList.first) > 0;
      for (unsigned j = i + 1; j < n; ++j) {
        unsigned dstId = memrefAndList.second[j];
        bool dstHasStore =
            getNode(dstId)->getStoreOpCount(memrefAndList.first) > 0;
        if (srcHasStore || dstHasStore)
          addEdge(srcId, dstId, memrefAndList.first);
      }
    }
  }
  return true;
}

namespace {

// GreedyFusion greedily fuses loop nests which have a producer/consumer or
// input-reuse relationship on a memref, with the goal of improving locality.
//
// The steps of the producer-consumer fusion algorithm are as follows:
//
// *) A worklist is initialized with node ids from the dependence graph.
// *) For each node id in the worklist:
//   *) Pop an AffineForOp of the worklist. This 'dstAffineForOp' will be a
//      candidate destination AffineForOp into which fusion will be attempted.
//   *) Add each LoadOp currently in 'dstAffineForOp' into list 'dstLoadOps'.
//   *) For each LoadOp in 'dstLoadOps' do:
//      *) Look up dependent loop nests which have a single store op to the same
//         memref.
//      *) Check if dependences would be violated by the fusion.
//      *) Get a computation slice of 'srcLoopNest', which adjusts its loop
//         bounds to be functions of 'dstLoopNest' IVs and symbols.
//      *) Fuse the 'srcLoopNest' computation slice into the 'dstLoopNest',
//         at a loop depth determined by the cost model in 'isFusionProfitable'.
//      *) Add the newly fused load/store operations to the state,
//         and also add newly fused load ops to 'dstLoopOps' to be considered
//         as fusion dst load ops in another iteration.
//      *) Remove old src loop nest and its associated state.
//
// The steps of the input-reuse fusion algorithm are as follows:
//
// *) Initialize 'worklist' with node ids from the dependence graph.
// *) For each 'dstNode' in the worklist:
//   *) Find a candidate sibling node 'sibNode' to fuse with 'dstNode' which
//      loads from the same memref, but which has no dependence paths to/from.
//   *) Get a computation slice of 'sibLoopNest', which adjusts its loop
//      bounds to be functions of 'dstLoopNest' IVs and symbols.
//   *) Fuse the 'sibLoopNest' computation slice into the 'dstLoopNest',
//      at a loop depth determined by the cost model in 'isFusionProfitable'.
//      This function also checks that the memref write region of 'sibLoopNest',
//      is preserved in the fused loop nest.
//   *) Update graph state to reflect the fusion of 'sibNode' into 'dstNode'.
//
// Given a graph where top-level operations are vertices in the set 'V' and
// edges in the set 'E' are dependences between vertices, this algorithm
// takes O(V) time for initialization, and has runtime O(V + E).
//
// This greedy algorithm is not 'maximal' due to the current restriction of
// fusing along single producer consumer edges, but there is a TODO: to fix
// this.
//
// TODO: Experiment with other fusion policies.
struct GreedyFusion {
public:
  // The data dependence graph to traverse during fusion.
  MemRefDependenceGraph *mdg;
  // Worklist of graph nodes visited during the fusion pass.
  SmallVector<unsigned, 8> worklist;
  // Parameter for local buffer size threshold.
  unsigned localBufSizeThreshold;
  // Parameter for fast memory space.
  Optional<unsigned> fastMemorySpace;
  // The amount of additional computation that is tolerated while fusing
  // pair-wise as a fraction of the total computation.
  double computeToleranceThreshold;

  using Node = MemRefDependenceGraph::Node;

  GreedyFusion(MemRefDependenceGraph *mdg, unsigned localBufSizeThreshold,
               Optional<unsigned> fastMemorySpace, bool maximalFusion,
               double computeToleranceThreshold)
      : mdg(mdg), localBufSizeThreshold(localBufSizeThreshold),
        fastMemorySpace(fastMemorySpace),
        computeToleranceThreshold(computeToleranceThreshold) {}

  /// Initializes 'worklist' with nodes from 'mdg'.
  void init() {
    // TODO: Add a priority queue for prioritizing nodes by different
    // metrics (e.g. arithmetic intensity/flops-to-bytes ratio).
    worklist.clear();
    for (auto &idAndNode : mdg->nodes) {
      worklist.push_back(idAndNode.first);
    }
  }

  /// Run only producer/consumer fusion on the `mdg`.
  void runProducerConsumerFusionOnly() {
    fuseProducerConsumerNodes(
        /*maxSrcUserCount=*/std::numeric_limits<unsigned>::max());
  }

  void fuseProducerConsumerNodes(unsigned maxSrcUserCount) {
    LLVM_DEBUG(llvm::dbgs() << "--- Producer/Consumer Fusion ---\n");
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();

      // Skip if this node was removed (fused into another node).
      if (mdg->nodes.count(dstId) == 0)
        continue;
      // Get 'dstNode' into which to attempt fusion.
      auto *dstNode = mdg->getNode(dstId);
      // Skip if 'dstNode' is not a loop nest.
      if (!isa<AffineForOp>(dstNode->op))
        continue;
      // TODO: how can a loop nest return values? What Does that even mean.
      // Skip if 'dstNode' is a loop nest returning values.
      // TODO: support loop nests that return values.
      if (dstNode->op->getNumResults() > 0)
        continue;

      LLVM_DEBUG(llvm::dbgs() << "Evaluating dst loop " << dstId << "\n");

      auto dstAffineForOp = cast<AffineForOp>(dstNode->op);

      // Try to fuse 'dstNode' with candidate producer loops until a fixed point
      // is reached. Fusing two loops may expose new fusion opportunities.
      bool dstNodeChanged;
      do {
        // Gather src loop candidates for 'dstNode' and visit them in "quasi"
        // reverse program order to minimize the number of iterations needed to
        // reach the fixed point. Note that this is a best effort approach since
        // 'getProducerCandidates' does not always guarantee that program order
        // in 'srcIdCandidates'.
        dstNodeChanged = false;
        SmallVector<unsigned, 16> srcIdCandidates;
        getProducerCandidates(dstId, mdg, srcIdCandidates);

        for (unsigned srcId : llvm::reverse(srcIdCandidates)) {
          // Get 'srcNode' from which to attempt fusion into 'dstNode'.
          auto *srcNode = mdg->getNode(srcId);
          auto srcAffineForOp = cast<AffineForOp>(srcNode->op);
          LLVM_DEBUG(llvm::dbgs() << "Evaluating src loop " << srcId
                                  << " for dst loop " << dstId << "\n");

          // AARON: We are assuming that there are no affine uses of variables
          // between the loops. This can make it illegal to fuse the first loop
          // into the second.

          // TODO: AARON: Be able to articulate what "loop depth" is in this circumstance
          ComputationSliceState a = ComputationSliceState();
          ComputationSliceState& bestSlice = a;
          FusionResult result = mlir::canFuseLoops(
              srcAffineForOp, dstAffineForOp,
              /*dstLoopDepth=*/1, &bestSlice, FusionStrategy::ProducerConsumer);

          if (result.value != FusionResult::Success) {
            LLVM_DEBUG(llvm::dbgs()
                           << "Can't fuse: not legal\n");
            continue;
          }

          // Fuse computation slice of 'srcLoopNest' into 'dstLoopNest'.
          fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice);
          dstNodeChanged = true;

          LLVM_DEBUG(llvm::dbgs()
                     << "Fused src loop " << srcId << " into dst loop " << dstId
                     << "\n");

          // Originally conditional, but we're just always removing src loop
          LLVM_DEBUG(llvm::dbgs()
                     << "Removing src loop " << srcId << " after fusion\n");
          // srcNode is no longer valid after it is removed from mdg.
          srcAffineForOp.erase();
          mdg->removeNode(srcId);
          srcNode = nullptr;
        }
      } while (dstNodeChanged);
    }
  }
};

} // end anonymous namespace

void SmallLoopFusion::runOnFunction() {
  MemRefDependenceGraph g;
  if (!g.init(getFunction()))
    return;

    LLVM_DEBUG(llvm::dbgs() << "--- Input Function ---\n");
    getFunction().dump();
    LLVM_DEBUG(llvm::dbgs() << "--- Nodes in order ---\n");
    g.dumpNodesInOrder();

  GreedyFusion fusion(&g, 0, 0, true, 0.30f);

  fusion.runProducerConsumerFusionOnly();
}
