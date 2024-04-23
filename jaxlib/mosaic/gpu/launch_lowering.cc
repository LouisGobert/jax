/* Copyright 2024 The JAX Authors.

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
#include <vector>

#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/DialectRegistry.h"
#include "mlir/include/mlir/IR/Location.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/TypeRange.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/ValueRange.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Support/TypeID.h"

namespace mosaic {
namespace gpu {

namespace {

mlir::Value packKernelArgs(mlir::OpBuilder &builder,
                           mlir::gpu::LaunchFuncOp launch) {
  std::vector<mlir::Type> kernel_operand_types;
  kernel_operand_types.reserve(launch.getNumKernelOperands());
  for (mlir::Value operand : launch.getKernelOperands()) {
    kernel_operand_types.push_back(operand.getType());
  }
  auto kernel_args_struct_ty = mlir::LLVM::LLVMStructType::getLiteral(
      builder.getContext(), kernel_operand_types);
  auto ptr_ty = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Value c1 = builder.create<mlir::LLVM::ConstantOp>(
      launch.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(1));
  mlir::Value kernel_args_struct = builder.create<mlir::LLVM::AllocaOp>(
      launch.getLoc(), ptr_ty, kernel_args_struct_ty, c1);
  mlir::Value kernel_args_array = builder.create<mlir::LLVM::AllocaOp>(
      launch.getLoc(), ptr_ty,
      mlir::LLVM::LLVMArrayType::get(builder.getI64Type(),
                                     launch.getNumKernelOperands()),
      c1);

  for (auto [i, operand] : llvm::enumerate(launch.getKernelOperands())) {
    mlir::LLVM::GEPArg gep_arg(i);
    mlir::Value storage_ptr = builder.create<mlir::LLVM::GEPOp>(
        launch.getLoc(), ptr_ty, operand.getType(), kernel_args_struct,
        gep_arg);
    builder.create<mlir::LLVM::StoreOp>(launch.getLoc(), operand, storage_ptr);
    mlir::Value array_slot_ptr = builder.create<mlir::LLVM::GEPOp>(
        launch.getLoc(), ptr_ty, builder.getI64Type(), kernel_args_array,
        gep_arg);
    builder.create<mlir::LLVM::StoreOp>(launch.getLoc(), storage_ptr,
                                        array_slot_ptr);
  }
  return kernel_args_array;
}

void emitRuntimeDecls(mlir::ModuleOp module) {
  auto ptr_ty = mlir::LLVM::LLVMPointerType::get(module.getContext());
  auto i32 = mlir::IntegerType::get(module.getContext(), 32);
  auto i64 = mlir::IntegerType::get(module.getContext(), 64);
  auto decl_builder = mlir::OpBuilder::atBlockBegin(module.getBody());
  decl_builder.create<mlir::func::FuncOp>(
      module.getLoc(), decl_builder.getStringAttr("mgpuLaunchKernel"),
      mlir::FunctionType::get(module.getContext(),
                              {ptr_ty, i64, i64, i64, i64, i64, i64, i32,
                               ptr_ty, ptr_ty, ptr_ty, i64},
                              {}),
      decl_builder.getStringAttr("private"), /*arg_attr=*/nullptr,
      /*res_attrs=*/nullptr);
  decl_builder.create<mlir::func::FuncOp>(
      module.getLoc(), decl_builder.getStringAttr("mgpuModuleLoad"),
      mlir::FunctionType::get(module.getContext(), {ptr_ty, i64}, {ptr_ty}),
      decl_builder.getStringAttr("private"), /*arg_attr=*/nullptr,
      /*res_attrs=*/nullptr);
  decl_builder.create<mlir::func::FuncOp>(
      module.getLoc(), decl_builder.getStringAttr("mgpuModuleGetFunction"),
      mlir::FunctionType::get(module.getContext(), {ptr_ty, ptr_ty}, {ptr_ty}),
      decl_builder.getStringAttr("private"), /*arg_attr=*/nullptr,
      /*res_attrs=*/nullptr);
}

void buildInitFunction(mlir::OpBuilder &module_builder,
                       mlir::func::FuncOp init_func,
                       llvm::StringRef kernel_name,
                       mlir::gpu::ObjectAttr object) {
  auto i64 = mlir::IntegerType::get(init_func.getContext(), 64);
  auto ptr_ty = mlir::LLVM::LLVMPointerType::get(init_func.getContext());
  mlir::Location loc = init_func.getLoc();
  auto builder =
      mlir::OpBuilder::atBlockBegin(&init_func.getBody().emplaceBlock());
  auto binary_global_decl = module_builder.create<mlir::LLVM::GlobalOp>(
      loc,
      mlir::LLVM::LLVMArrayType::get(builder.getI8Type(),
                                     object.getObject().size()),
      /*is_constant=*/true,
      /*linkage=*/mlir::LLVM::Linkage::Internal,
      /*name=*/
      builder.getStringAttr(kernel_name.str() + "_kernel_binary"),
      /*value=*/object.getObject());
  mlir::Value binary_addr = builder.create<mlir::LLVM::AddressOfOp>(
      init_func.getLoc(), binary_global_decl);
  mlir::Value binary_size = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64, builder.getI64IntegerAttr(object.getObject().size()));
  mlir::Value module_handle =
      builder
          .create<mlir::func::CallOp>(
              loc, "mgpuModuleLoad", ptr_ty,
              mlir::ValueRange{binary_addr, binary_size})
          .getResult(0);

  // TODO(apaszke): This will create duplicate globals if the kernel
  // is called from multiple functions!
  mlir::StringAttr kernel_name_global_name =
      builder.getStringAttr(kernel_name.str() + "_name");
  auto kernel_name_global = module_builder.create<mlir::LLVM::GlobalOp>(
      loc,
      mlir::LLVM::LLVMArrayType::get(builder.getI8Type(),
                                     kernel_name.size() + 1),
      /*is_constant=*/true,
      /*linkage=*/mlir::LLVM::Linkage::Internal,
      /*name=*/kernel_name_global_name,
      /*value=*/
      builder.getStringAttr(
          llvm::Twine(kernel_name).concat(llvm::Twine('\0'))));
  mlir::Value kernel_name_ptr =
      builder.create<mlir::LLVM::AddressOfOp>(loc, kernel_name_global);
  mlir::Value kernel_handle =
      builder
          .create<mlir::func::CallOp>(
              loc, "mgpuModuleGetFunction", ptr_ty,
              mlir::ValueRange{module_handle, kernel_name_ptr})
          .getResult(0);
  builder.create<mlir::func::ReturnOp>(loc, kernel_handle);
}

mlir::LogicalResult launchPreloadedKernel(mlir::func::FuncOp func,
                                          mlir::gpu::LaunchFuncOp launch,
                                          mlir::Value kernel_handle) {
  auto ptr_ty = mlir::LLVM::LLVMPointerType::get(func.getContext());
  // Lower gpu.launch_func to a call to mgpuLaunchKernel.
  mlir::OpBuilder builder(launch);
  mlir::Value dynamic_smem = launch.getDynamicSharedMemorySize();
  if (!dynamic_smem) {
    dynamic_smem = builder.create<mlir::LLVM::ConstantOp>(
        launch.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0));
  }
  mlir::Value arg_ptr_array = packKernelArgs(builder, launch);
  if (launch.hasClusterSize()) {
    return launch.emitOpError("Clusters not supported yet.");
  }
  mlir::gpu::KernelDim3 grid = launch.getGridSizeOperandValues();
  mlir::gpu::KernelDim3 block = launch.getBlockSizeOperandValues();
  mlir::Value llvm_nullptr =
      builder.create<mlir::LLVM::ZeroOp>(launch.getLoc(), ptr_ty);
  mlir::Value stream = launch.getAsyncObject();
  mlir::Value param_count = builder.create<mlir::LLVM::ConstantOp>(
      launch.getLoc(), builder.getI64Type(),
      builder.getI64IntegerAttr(launch.getNumKernelOperands()));
  builder.create<mlir::func::CallOp>(
      launch.getLoc(), "mgpuLaunchKernel", mlir::TypeRange{},
      mlir::ValueRange{kernel_handle, grid.x, grid.y, grid.z, block.x, block.y,
                       block.z, dynamic_smem, stream, arg_ptr_array,
                       llvm_nullptr, param_count});
  return mlir::success();
}

class GpuLaunchLoweringPass : public ::mlir::OperationPass<mlir::ModuleOp> {
 public:
  GpuLaunchLoweringPass()
      : ::mlir::OperationPass<mlir::ModuleOp>(
            ::mlir::TypeID::get<GpuLaunchLoweringPass>()) {}
  GpuLaunchLoweringPass(const GpuLaunchLoweringPass &other)
      : ::mlir::OperationPass<mlir::ModuleOp>(other) {}
  GpuLaunchLoweringPass &operator=(const GpuLaunchLoweringPass &) = delete;
  GpuLaunchLoweringPass(GpuLaunchLoweringPass &&) = delete;
  GpuLaunchLoweringPass &operator=(GpuLaunchLoweringPass &&) = delete;
  ~GpuLaunchLoweringPass() = default;

  // Pass boilerplate...
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("gpu-launch-lowering");
  }
  ::llvm::StringRef getArgument() const override { return getArgumentName(); }
  ::llvm::StringRef getDescription() const override { return ""; }
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("GpuLaunchLoweringPass");
  }
  ::llvm::StringRef getName() const override { return getPassName(); }
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<GpuLaunchLoweringPass>();
  }
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<GpuLaunchLoweringPass>(
        *static_cast<const GpuLaunchLoweringPass *>(this));
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuLaunchLoweringPass)

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    auto ptr_ty = mlir::LLVM::LLVMPointerType::get(module.getContext());
    emitRuntimeDecls(module);
    for (mlir::Operation &op : *module.getBody()) {
      if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(&op)) {
        if (func.isDeclaration() ||
            !func->getAttr(
                mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName())) {
          continue;
        }
        auto module_builder = mlir::OpBuilder::atBlockBegin(module.getBody());
        auto init_func = module_builder.create<mlir::func::FuncOp>(
            op.getLoc(), func.getName().str() + "_init",
            mlir::FunctionType::get(func->getContext(), {}, {ptr_ty}));
        init_func->setAttr(mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                           mlir::UnitAttr::get(func->getContext()));
        bool had_launch = false;
        auto result = getOperation()->walk([&](mlir::gpu::LaunchFuncOp launch)
                                               -> mlir::WalkResult {
          if (had_launch) {
            launch->emitOpError("Only one launch per function supported.");
            return mlir::WalkResult::interrupt();
          }
          had_launch = true;
          auto binary =
              mlir::SymbolTable::lookupNearestSymbolFrom<mlir::gpu::BinaryOp>(
                  launch, launch.getKernelModuleName());
          if (!binary) {
            launch.emitError("Failed to find the gpu.binary op for ")
                << launch.getKernelModuleName();
            return mlir::WalkResult::interrupt();
          }
          if (binary.getObjects().size() != 1) {
            binary.emitOpError("Expected exactly one object in the binary.");
            return mlir::WalkResult::interrupt();
          }
          mlir::gpu::ObjectAttr object =
              mlir::cast<mlir::gpu::ObjectAttr>(*binary.getObjects().begin());
          if (object.getFormat() != mlir::gpu::CompilationTarget::Fatbin &&
              object.getFormat() != mlir::gpu::CompilationTarget::Binary) {
            binary.emitOpError("Expected a binary or a fatbin object.");
            return mlir::WalkResult::interrupt();
          }

          buildInitFunction(module_builder, init_func,
                            launch.getKernelName().getValue(), object);

          // Add a new function argument for the kernel handle.
          func.insertArgument(0, ptr_ty,
                              mlir::DictionaryAttr::get(func.getContext()),
                              mlir::UnknownLoc::get(func.getContext()));
          mlir::Value kernel_handle = func.getArgument(0);
          if (launchPreloadedKernel(func, launch, kernel_handle).failed()) {
            return mlir::WalkResult::interrupt();
          }
          launch.erase();

          // TODO(apaszke): Generate a destructor function.
          // builder.CreateCall(getModuleUnloadFn(), {moduleObject});

          return mlir::WalkResult::advance();
        });
        if (!had_launch) {
          init_func.erase();
        }
        if (result == mlir::WalkResult::interrupt()) {
          signalPassFailure();
        }
      }
    }
  }
};

}  // namespace

void registerGpuLaunchLoweringPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<GpuLaunchLoweringPass>();
  });
}

}  // namespace gpu
}  // namespace mosaic
