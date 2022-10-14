#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

struct ExampleOpInterfaceTraits {
  struct Concept {
    virtual ~Concept() = default;

    virtual unsigned ohYeah(mlir::Operation *op) const = 0;
  };

  template <typename ConcreteOp>
  struct Model : public Concept {
    unsigned ohYeah(mlir::Operation *op) const final {
      return llvm::cast<ConcreteOp>(op).ohYeah();
    }
  };

  template <typename ConcreteOp>
  struct FallbackModel : public Concept {
    unsigned ohYeah(mlir::Operation *op) const override {}
  };

  template <typename ConcreteModel, typename ConcreteType>
  struct ExternalModel : public FallbackModel<ConcreteModel> {
    unsigned ohYeah(mlir::Operation *op) const override {}
  };
};

/// Define the main interface class that analyses and transformations will
/// interface with.
class ExampleOpInterface
    : public mlir::OpInterface<ExampleOpInterface, ExampleOpInterfaceTraits> {
public:
  /// Inherit the base class constructor to support LLVM-style casting.
  using OpInterface<ExampleOpInterface, ExampleOpInterfaceTraits>::OpInterface;

  /// The interface dispatches to 'getImpl()', a method provided by the base
  /// `OpInterface` class that returns an instance of the concept.
  unsigned ohYeah() { return getImpl()->ohYeah(getOperation()); }
};

class MyOp : public ::mlir::Op<MyOp, ExampleOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;

  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("standalone.bar");
  }
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState) {}

  unsigned int ohYeah() { return 69; }
};

class StandaloneDialect : public ::mlir::Dialect {
  explicit StandaloneDialect(::mlir::MLIRContext *context)
      : mlir::Dialect(getDialectNamespace(), context,
                      ::mlir::TypeID::get<StandaloneDialect>()) {
    addOperations<MyOp>();
  }

  friend class ::mlir::MLIRContext;

public:
  ~StandaloneDialect() override = default;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("standalone");
  }
};

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<StandaloneDialect>();
  mlir::OpBuilder builder(&context);
  MyOp barop = builder.create<MyOp>(builder.getUnknownLoc());
  if (ExampleOpInterface example =
          llvm::dyn_cast<ExampleOpInterface>(barop.getOperation())) {
    llvm::errs() << "ohYeah returned: " << example.ohYeah() << "\n";
  }
}