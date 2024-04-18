/* Copyright 2023 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/include/mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/include/mlir/IR/AffineMap.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/DialectRegistry.h"
#include "mlir/include/mlir/IR/Matchers.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_LEGALIZATIONPASS
#define GEN_PASS_DEF_LEGALIZATIONPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

namespace {
struct LegalizationPass : public impl::LegalizationPassBase<LegalizationPass> {
  LegalizationPass() = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override;
};

template <typename DefiningOp>
LogicalResult matchAndRewriteTransferOfExpandOrCollapseShape(
    vector::TransferReadOp op, PatternRewriter &rewriter) {
  if (op.hasOutOfBoundsDim()) {
    return rewriter.notifyMatchFailure(op, "out of bounds transfer dim");
  }
  if (op.getMask()) {
    return rewriter.notifyMatchFailure(op, "masked transfer");
  }
  if (!op.getPermutationMap().isIdentity()) {
    return rewriter.notifyMatchFailure(op, "non identity permutation map");
  }
  auto expand = op.getSource().template getDefiningOp<DefiningOp>();
  if (!expand) {
    return rewriter.notifyMatchFailure(
        op, "not a tensor.expand_shape/collapse_shape");
  }
  if (auto result_type = op.getType().template dyn_cast<ShapedType>();
      !result_type ||
      result_type.getShape() != expand.getResultType().getShape()) {
    return rewriter.notifyMatchFailure(op, "output type mismatch");
  }
  SmallVector<Value> indices = {op.getIndices().begin(), op.getIndices().end()};
  if (absl::c_any_of(
          indices, [](Value index) { return !isConstantIntValue(index, 0); })) {
    return rewriter.notifyMatchFailure(op, "non zero indices");
  }
  auto expand_src_type = expand.getSrcType();
  // We know from preconditions that there are no out of bound dims.
  SmallVector<bool> in_bounds(expand_src_type.getRank(), true);
  rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
      op, op.getType(),
      rewriter.create<vector::TransferReadOp>(
          op.getLoc(),
          VectorType::get(expand_src_type.getShape(),
                          expand_src_type.getElementType()),
          expand.getSrc(),
          SmallVector<Value>(
              expand_src_type.getRank(),
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0)),
          AffineMapAttr::get(AffineMap::getMultiDimIdentityMap(
              expand_src_type.getRank(), op->getContext())),
          op.getPadding(), /*mask=*/Value(),
          rewriter.getBoolArrayAttr(in_bounds)));
  return success();
}

// Rewrite `vector.transfer_read(tensor.expand_shape)` as
// `vector.shape_cast(vector.transfer_read)`.
struct TransferReadOfExpandShape
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteTransferOfExpandOrCollapseShape<
        tensor::ExpandShapeOp>(op, rewriter);
  }
};

// Rewrite `vector.transfer_read(tensor.collapse_shape)` as
// `vector.shape_cast(vector.transfer_read)`.
struct TransferReadOfCollapseShape
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteTransferOfExpandOrCollapseShape<
        tensor::CollapseShapeOp>(op, rewriter);
  }
};

// Rewrite a `vector.transfer_read` of a dense tensor constant as a dense
// vector constant.
struct TransferReadOfConstant
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr constant_elements;
    Attribute constant_value;
    if (matchPattern(op.getSource(), m_Constant(&constant_elements)) &&
        constant_elements.isSplat()) {
      constant_value = constant_elements.getSplatValue<Attribute>();
    } else {
      return rewriter.notifyMatchFailure(op, "not an arith.constant");
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, op.getVectorType(),
        DenseElementsAttr::get(op.getVectorType(), constant_value));
    return success();
  }
};

void LegalizationPass::runOnOperation() {
  auto func = getOperation();
  MLIRContext *ctx = func.getContext();

  RewritePatternSet patterns(ctx);
  vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
  patterns.add<TransferReadOfExpandShape>(ctx);
  patterns.add<TransferReadOfCollapseShape>(ctx);
  patterns.add<TransferReadOfConstant>(ctx);

  SmallVector<Operation *> operations;
  func.walk([&](vector::TransferReadOp op) { operations.push_back(op); });
  func.walk([&](vector::TransferWriteOp op) { operations.push_back(op); });
  if (failed(applyOpPatternsAndFold(operations, std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizationPass() {
  return std::make_unique<LegalizationPass>();
}

}  // namespace mlir::tpu
