/* Copyright 2024 The OpenXLA Authors.

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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {

using ::mlir::ArrayRef;
using ::mlir::NamedAttribute;

namespace {

std::unique_ptr<HloInstruction> CreateAddTritonCustomCall(
    Shape tuple_shape, HloInstruction* param_0, HloInstruction* param_1) {
  mlir::MLIRContext context_;
  mlir::Builder builder(&context_);

  // Create the backend_config for the triton custom call.
  const std::string kMLIRText = R"(
  module {
    tt.func public @add_one(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}, %arg3: !tt.ptr<f32, 1> {tt.divisibility = 32 : i32}) {
      %0 = tt.get_program_id x : i32
      %1 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
      %2 = tt.load %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
      %cst = arith.constant 1.000000e+00 : f32
      %3 = arith.addf %1, %cst : f32
      %4 = tt.load %arg2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
      tt.store %arg2, %3 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
      %5 = tt.load %arg3 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
      tt.store %arg3, %2 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<f32>
      tt.return
    }
  }
  )";

  NamedAttribute name =
      builder.getNamedAttr("name", builder.getStringAttr("add_one"));
  NamedAttribute ir =
      builder.getNamedAttr("ir", builder.getStringAttr(kMLIRText));
  NamedAttribute num_stages =
      builder.getNamedAttr("num_stages", builder.getI32IntegerAttr(3));
  NamedAttribute num_warps =
      builder.getNamedAttr("num_warps", builder.getI32IntegerAttr(4));
  NamedAttribute grid_x =
      builder.getNamedAttr("grid_x", builder.getI32IntegerAttr(1));
  NamedAttribute grid_y =
      builder.getNamedAttr("grid_y", builder.getI32IntegerAttr(1));
  NamedAttribute grid_z =
      builder.getNamedAttr("grid_z", builder.getI32IntegerAttr(1));
  NamedAttribute debug =
      builder.getNamedAttr("debug", builder.getBoolAttr(false));

  std::vector<NamedAttribute> attributes = {
      name, ir, num_stages, num_warps, grid_x, grid_y, grid_z, debug};
  ArrayRef<NamedAttribute> attributesRef(attributes);
  mlir::DictionaryAttr backend_config =
      mlir::DictionaryAttr::get(&context_, attributesRef);

  // Parse the backend_config into a string.
  std::string backend_config_str;
  llvm::raw_string_ostream(backend_config_str) << backend_config;

  return HloInstruction::CreateCustomCall(tuple_shape, {param_0, param_1},
                                          "__gpu$xla.gpu.triton",
                                          backend_config_str);
}

}  // namespace

class GpuIrEmitterUnnestedTest : public GpuCodegenTest {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
};

TEST_F(GpuIrEmitterUnnestedTest,
       EmitTritonCustomCallWithCorrectLoweringAndWithoutNoaliasOrAlignment) {
  if (!GetCudaComputeCapability().IsAtLeastAmpere()) {
    GTEST_SKIP() << "Triton support is only enabled for Ampere GPUs and up.";
  }

  // Tests that the lowering of a Triton custom call produces the correct LLVM
  // IR, and that the arguments do not specify noalias or alignment attributes.

  HloComputation::Builder computation_builder(TestName());

  // Create parameters and custom call in the computation builder.
  Shape scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "arg_1"));

  computation_builder.AddInstruction(
      CreateAddTritonCustomCall(tuple_shape, param_0, param_1));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());

  // Check that the compiled llvm ir matches the expected lowering of our tt ir.
  // We check that the arguments do not specify noalias or alignment attributes,
  // as this prevents recompilation based on the alignment of the input buffers.
  CompileAndVerifyIr(std::move(module),
                     R"(
; CHECK: @add_one
; CHECK-NOT: noalias align
; CHECK-SAME: dereferenceable(4) %arg0
; CHECK-NOT: noalias align
; CHECK-SAME: dereferenceable(4) %arg1
; CHECK-NOT: noalias align
; CHECK-SAME: dereferenceable(4) %arg2
; CHECK-NOT: noalias align
; CHECK-SAME: dereferenceable(4) %arg3
; CHECK-DAG:  addrspacecast ptr %arg0 to ptr addrspace(1)
; CHECK-DAG:  addrspacecast ptr %arg1 to ptr addrspace(1)
; CHECK-DAG:  addrspacecast ptr %arg2 to ptr addrspace(1)
; CHECK-DAG:  addrspacecast ptr %arg3 to ptr addrspace(1)
      )",
                     /*match_optimized_ir=*/false);
}

TEST_F(GpuIrEmitterUnnestedTest, CanNotEmitTritonCustomCallOnPreAmpereGpu) {
  if (GetCudaComputeCapability().IsAtLeastAmpere()) {
    GTEST_SKIP() << "Running on Ampere or more recent GPU, skipping.";
  }

  HloComputation::Builder computation_builder(TestName());

  // Create parameters and custom call in the computation builder.
  Shape scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "arg_1"));

  computation_builder.AddInstruction(
      CreateAddTritonCustomCall(tuple_shape, param_0, param_1));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());

  EXPECT_THAT(
      CompileToExecutable(std::move(module), /*run_optimization_passes=*/false),
      tsl::testing::StatusIs(
          absl::StatusCode::kFailedPrecondition,
          ::testing::HasSubstr("Triton support is only enabled for Ampere GPUs "
                               "(compute capability 8.0) and up, but got")));
}

class TritonCustomCallTest : public HloTestBase {};

TEST_F(TritonCustomCallTest, NoArgumentDeduplication) {
  if (auto cc = backend()
                    .default_stream_executor()
                    ->GetDeviceDescription()
                    .cuda_compute_capability();
      !cc.IsAtLeastAmpere()) {
    GTEST_SKIP() << "Triton support is only enabled for Ampere GPUs and up.";
  }

  // Tests that no argument deduplication is done for Triton kernels.
  //
  // Triton kernels are compiled on the first call and re-used for all the
  // following calls. So, if we are unlucky, we could end up calling the
  // compiled kernel with fewer arguments than it expects in the presence
  // of argument deduplication.
  //
  // For example,
  //
  //  * The first call is f(x, y). The arguments are distinct, no deduplication
  //    is done at compilation time and the compiled kernel expects two
  //    arguments.
  //  * The second call is f(x, x). The arguments are deduplicated and we
  //    call the previously compiled kernel with just x, causing a crash.

  HloComputation::Builder computation_builder(TestName());

  Shape scalar_shape = xla::ShapeUtil::MakeShape(xla::F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "arg_1"));

  auto* instr_0 = computation_builder.AddInstruction(
      CreateAddTritonCustomCall(tuple_shape, param_0, param_1));
  computation_builder.AddInstruction(
      CreateAddTritonCustomCall(tuple_shape, instr_0, instr_0));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
}

std::unique_ptr<HloInstruction> CreateAddTritonCustomCallWGMMA(
    Shape shape, HloInstruction* param_0, HloInstruction* param_1, HloInstruction* param_2) {
  mlir::MLIRContext context_;
  mlir::Builder builder(&context_);

  // Create the backend_config for the triton custom call.
  const std::string kMLIRText = R"(

module @mha_forward {
  tt.func public @mha_forward(%arg0: !tt.ptr<bf16> {tt.divisibility = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 32 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 32 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 32 : i32}) {
    %0 = tt.get_program_id x : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %1 = arith.muli %c0_i32, %c64_i32 : i32
    %c128_i32 = arith.constant 128 : i32
    %2 = arith.muli %c0_i32_0, %c128_i32 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c128_i32_3 = arith.constant 128 : i32
    %3 = arith.muli %c0_i32_1, %c128_i32_3 : i32
    %c64_i32_4 = arith.constant 64 : i32
    %4 = arith.muli %c0_i32_2, %c64_i32_4 : i32
    %c0_i32_5 = arith.constant 0 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %c64_i32_7 = arith.constant 64 : i32
    %5 = arith.muli %c0_i32_5, %c64_i32_7 : i32
    %c128_i32_8 = arith.constant 128 : i32
    %6 = arith.muli %c0_i32_6, %c128_i32_8 : i32
    %c0_i32_9 = arith.constant 0 : i32
    %c0_i32_10 = arith.constant 0 : i32
    %c64_i32_11 = arith.constant 64 : i32
    %7 = arith.muli %c0_i32_9, %c64_i32_11 : i32
    %c128_i32_12 = arith.constant 128 : i32
    %8 = arith.muli %c0_i32_10, %c128_i32_12 : i32
    %c0_i32_13 = arith.constant 0 : i32
    %9 = tt.splat %c0_i32_13 : i32 -> tensor<64x128xi32>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %12 = tt.broadcast %11 : tensor<64x1xi32> -> tensor<64x128xi32>
    %13 = tt.splat %1 : i32 -> tensor<64x128xi32>
    %14 = arith.addi %12, %13 : tensor<64x128xi32>
    %c128_i32_14 = arith.constant 128 : i32
    %15 = tt.splat %c128_i32_14 : i32 -> tensor<64x128xi32>
    %16 = arith.muli %14, %15 : tensor<64x128xi32>
    %17 = arith.addi %9, %16 : tensor<64x128xi32>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %19 = tt.expand_dims %18 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %20 = tt.broadcast %19 : tensor<1x128xi32> -> tensor<64x128xi32>
    %21 = tt.splat %2 : i32 -> tensor<64x128xi32>
    %22 = arith.addi %20, %21 : tensor<64x128xi32>
    %c1_i32 = arith.constant 1 : i32
    %23 = tt.splat %c1_i32 : i32 -> tensor<64x128xi32>
    %24 = arith.muli %22, %23 : tensor<64x128xi32>
    %25 = arith.addi %17, %24 : tensor<64x128xi32>
    %26 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
    %27 = tt.addptr %26, %25 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
    %28 = tt.load %27 : tensor<64x128x!tt.ptr<bf16>>
    %c0_i32_15 = arith.constant 0 : i32
    %29 = tt.splat %c0_i32_15 : i32 -> tensor<128x64xi32>
    %30 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %31 = tt.expand_dims %30 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %32 = tt.broadcast %31 : tensor<128x1xi32> -> tensor<128x64xi32>
    %33 = tt.splat %3 : i32 -> tensor<128x64xi32>
    %34 = arith.addi %32, %33 : tensor<128x64xi32>
    %c64_i32_16 = arith.constant 64 : i32
    %35 = tt.splat %c64_i32_16 : i32 -> tensor<128x64xi32>
    %36 = arith.muli %34, %35 : tensor<128x64xi32>
    %37 = arith.addi %29, %36 : tensor<128x64xi32>
    %38 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %39 = tt.expand_dims %38 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %40 = tt.broadcast %39 : tensor<1x64xi32> -> tensor<128x64xi32>
    %41 = tt.splat %4 : i32 -> tensor<128x64xi32>
    %42 = arith.addi %40, %41 : tensor<128x64xi32>
    %c1_i32_17 = arith.constant 1 : i32
    %43 = tt.splat %c1_i32_17 : i32 -> tensor<128x64xi32>
    %44 = arith.muli %42, %43 : tensor<128x64xi32>
    %45 = arith.addi %37, %44 : tensor<128x64xi32>
    %46 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
    %47 = tt.addptr %46, %45 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    %48 = tt.load %47 : tensor<128x64x!tt.ptr<bf16>>
    %cst = arith.constant 0.000000e+00 : f32
    %49 = tt.splat %cst : f32 -> tensor<64x64xf32>
    %50 = tt.dot %28, %48, %49, inputPrecision = tf32 : tensor<64x128xbf16> * tensor<128x64xbf16> -> tensor<64x64xf32>
    %51 = arith.truncf %50 : tensor<64x64xf32> to tensor<64x64xbf16>
    %c0_i32_18 = arith.constant 0 : i32
    %52 = tt.splat %c0_i32_18 : i32 -> tensor<64x128xi32>
    %53 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %54 = tt.expand_dims %53 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %55 = tt.broadcast %54 : tensor<64x1xi32> -> tensor<64x128xi32>
    %56 = tt.splat %5 : i32 -> tensor<64x128xi32>
    %57 = arith.addi %55, %56 : tensor<64x128xi32>
    %c128_i32_19 = arith.constant 128 : i32
    %58 = tt.splat %c128_i32_19 : i32 -> tensor<64x128xi32>
    %59 = arith.muli %57, %58 : tensor<64x128xi32>
    %60 = arith.addi %52, %59 : tensor<64x128xi32>
    %61 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %62 = tt.expand_dims %61 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %63 = tt.broadcast %62 : tensor<1x128xi32> -> tensor<64x128xi32>
    %64 = tt.splat %6 : i32 -> tensor<64x128xi32>
    %65 = arith.addi %63, %64 : tensor<64x128xi32>
    %c1_i32_20 = arith.constant 1 : i32
    %66 = tt.splat %c1_i32_20 : i32 -> tensor<64x128xi32>
    %67 = arith.muli %65, %66 : tensor<64x128xi32>
    %68 = arith.addi %60, %67 : tensor<64x128xi32>
    %69 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
    %70 = tt.addptr %69, %68 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
    %71 = tt.load %70 : tensor<64x128x!tt.ptr<bf16>>
    %cst_21 = arith.constant 0.000000e+00 : f32
    %72 = tt.splat %cst_21 : f32 -> tensor<64x128xf32>
    %73 = tt.dot %51, %71, %72, inputPrecision = tf32 : tensor<64x64xbf16> * tensor<64x128xbf16> -> tensor<64x128xf32>
    %74 = arith.truncf %73 : tensor<64x128xf32> to tensor<64x128xbf16>
    %c0_i32_22 = arith.constant 0 : i32
    %75 = tt.splat %c0_i32_22 : i32 -> tensor<64x128xi32>
    %76 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %77 = tt.expand_dims %76 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %78 = tt.broadcast %77 : tensor<64x1xi32> -> tensor<64x128xi32>
    %79 = tt.splat %7 : i32 -> tensor<64x128xi32>
    %80 = arith.addi %78, %79 : tensor<64x128xi32>
    %c128_i32_23 = arith.constant 128 : i32
    %81 = tt.splat %c128_i32_23 : i32 -> tensor<64x128xi32>
    %82 = arith.muli %80, %81 : tensor<64x128xi32>
    %83 = arith.addi %75, %82 : tensor<64x128xi32>
    %84 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %85 = tt.expand_dims %84 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %86 = tt.broadcast %85 : tensor<1x128xi32> -> tensor<64x128xi32>
    %87 = tt.splat %8 : i32 -> tensor<64x128xi32>
    %88 = arith.addi %86, %87 : tensor<64x128xi32>
    %c1_i32_24 = arith.constant 1 : i32
    %89 = tt.splat %c1_i32_24 : i32 -> tensor<64x128xi32>
    %90 = arith.muli %88, %89 : tensor<64x128xi32>
    %91 = arith.addi %83, %90 : tensor<64x128xi32>
    %92 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>>
    %93 = tt.addptr %92, %91 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi32>
    %94 = tt.load %93 : tensor<64x128x!tt.ptr<bf16>>
    tt.store %93, %74 : tensor<64x128x!tt.ptr<bf16>>
    tt.return
  }
} 
  )";

  NamedAttribute name =
      builder.getNamedAttr("name", builder.getStringAttr("mha_forward"));
  NamedAttribute ir =
      builder.getNamedAttr("ir", builder.getStringAttr(kMLIRText));
  NamedAttribute num_stages =
      builder.getNamedAttr("num_stages", builder.getI32IntegerAttr(3));
  NamedAttribute num_warps =
      builder.getNamedAttr("num_warps", builder.getI32IntegerAttr(8));
  NamedAttribute grid_x =
      builder.getNamedAttr("grid_x", builder.getI32IntegerAttr(1));
  NamedAttribute grid_y =
      builder.getNamedAttr("grid_y", builder.getI32IntegerAttr(1));
  NamedAttribute grid_z =
      builder.getNamedAttr("grid_z", builder.getI32IntegerAttr(1));
  NamedAttribute debug =
      builder.getNamedAttr("debug", builder.getBoolAttr(false));

  std::vector<NamedAttribute> attributes = {
      name, ir, num_stages, num_warps, grid_x, grid_y, grid_z, debug};
  ArrayRef<NamedAttribute> attributesRef(attributes);
  mlir::DictionaryAttr backend_config =
      mlir::DictionaryAttr::get(&context_, attributesRef);

  // Parse the backend_config into a string.
  std::string backend_config_str;
  llvm::raw_string_ostream(backend_config_str) << backend_config;

  return HloInstruction::CreateCustomCall(shape, {param_0, param_1, param_2},
                                          "__gpu$xla.gpu.triton",
                                          backend_config_str);
}

TEST_F(TritonCustomCallTest, WGMMASegfault) {
  if (auto cc = backend()
                    .default_stream_executor()
                    ->GetDeviceDescription()
                    .cuda_compute_capability();
      !cc.IsAtLeastAmpere()) {
    GTEST_SKIP() << "Triton support is only enabled for Ampere GPUs and up.";
  }

  HloComputation::Builder computation_builder(TestName());

  Shape shape1 = xla::ShapeUtil::MakeShape(xla::BF16, {64, 128});
  Shape shape2 = xla::ShapeUtil::MakeShape(xla::BF16, {128, 64});

  HloInstruction* param_0 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape1, "arg_0"));

  HloInstruction* param_1 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape2, "arg_1"));

  HloInstruction* param_2 = computation_builder.AddInstruction(
      HloInstruction::CreateParameter(2, shape1, "arg_2"));

  auto* instr_0 = computation_builder.AddInstruction(
      CreateAddTritonCustomCallWGMMA(shape1, param_0, param_1, param_2));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(computation_builder.Build());
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
}

}  // namespace gpu
}  // namespace xla
