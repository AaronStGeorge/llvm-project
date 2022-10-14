#include <iostream>
#include <mlir/IR/Builders.h>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

struct ExampleOpInterfaceTraits {
  /// Define a base concept class that specifies the virtual interface to be
  /// implemented.
  struct Concept {
    virtual ~Concept() = default;

    /// This is an example of a non-static hook to an operation.
    virtual unsigned exampleInterfaceHook(mlir::Operation *op) const = 0;
  };

  /// Define a model class that specializes a concept on a given operation type.
  template <typename ConcreteOp>
  struct Model : public Concept {
    /// Override the method to dispatch on the concrete operation.
    unsigned exampleInterfaceHook(mlir::Operation *op) const final {
      return llvm::cast<ConcreteOp>(op).exampleInterfaceHook();
    }
  };

  /// Unlike `Model`, `FallbackModel` passes the type object through to the
  /// hook, making it accessible in the method body even if the method is not
  /// defined in the class itself and thus has no `this` access. ODS
  /// automatically generates this class for all interfaces.
  template <typename ConcreteOp>
  struct FallbackModel : public Concept {
    unsigned exampleInterfaceHook(mlir::Operation *op) const override {
//      getImpl()->exampleInterfaceHook(type);
    }
  };

  template <typename ConcreteModel, typename ConcreteType>
  struct ExternalModel : public FallbackModel<ConcreteModel> {
    unsigned exampleInterfaceHook(mlir::Operation *op) const override {
      // Default implementation can be provided here.
//      return type.cast<ConcreteType>().callSomeTypeSpecificMethod();
    }
  };
};

//// This is the implementation of a dialect fallback for `ExampleOpInterface`.
//struct FallbackExampleOpInterface
//    : public ExampleOpInterface::FallbackModel<
//        FallbackExampleOpInterface> {
//  static bool classof(Operation *op) { return true; }
//
//  unsigned exampleInterfaceHook(Operation *op) const;
//  unsigned exampleStaticInterfaceHook() const;
//};

/// Define the main interface class that analyses and transformations will
/// interface with.
class ExampleOpInterface : public mlir::OpInterface<ExampleOpInterface,
                                              ExampleOpInterfaceTraits> {
public:
  /// Inherit the base class constructor to support LLVM-style casting.
  using OpInterface<ExampleOpInterface, ExampleOpInterfaceTraits>::OpInterface;

  /// The interface dispatches to 'getImpl()', a method provided by the base
  /// `OpInterface` class that returns an instance of the concept.
  unsigned exampleInterfaceHook() {
    return getImpl()->exampleInterfaceHook(getOperation());
  }
};





class BarOp;
class BarOpAdaptor {
public:
  BarOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  BarOpAdaptor(BarOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getBody();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class BarOp : public ::mlir::Op<BarOp, ::mlir::OpTrait::OpInvariants, ExampleOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BarOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("standalone.bar");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getBody();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
public:
  unsigned int exampleInterfaceHook();
};

//===----------------------------------------------------------------------===//
// ::mlir::standalone::BarOp definitions
//===----------------------------------------------------------------------===//

BarOpAdaptor::BarOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("standalone.bar", odsAttrs.getContext());
}

BarOpAdaptor::BarOpAdaptor(BarOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BarOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BarOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BarOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
          std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BarOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange BarOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &BarOpAdaptor::getBody() {
  return *odsRegions[0];
}

::mlir::LogicalResult BarOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

unsigned BarOp::exampleInterfaceHook() {
  return 9;
}

std::pair<unsigned, unsigned> BarOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BarOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BarOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BarOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &BarOp::getBody() {
  return (*this)->getRegion(0);
}

void BarOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
  (void)odsState.addRegion();
}

void BarOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BarOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}


static ::mlir::LogicalResult __mlir_ods_local_region_constraint_StandaloneOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
                                       << (regionName.empty() ? " " : " ('" + regionName + "') ")
                                       << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}

::mlir::LogicalResult BarOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_StandaloneOps0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult BarOp::verifyInvariants() {
  return verifyInvariantsImpl();
}



class StandaloneDialect : public ::mlir::Dialect {
  explicit StandaloneDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~StandaloneDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("standalone");
  }
};


void StandaloneDialect::initialize() {
  addOperations<
      BarOp
  >();
}

StandaloneDialect::StandaloneDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<StandaloneDialect>()) {
  initialize();
}
StandaloneDialect::~StandaloneDialect() = default;


int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<StandaloneDialect>();
  mlir::OpBuilder builder(&context);
  BarOp barop = builder.create<BarOp>(builder.getUnknownLoc());
  if (ExampleOpInterface example = llvm::dyn_cast<ExampleOpInterface>(barop.getOperation())) {
    llvm::errs() << "hook returned = " << example.exampleInterfaceHook() << "\n";
  }
//  printf("bla\n", barop->getName());
}