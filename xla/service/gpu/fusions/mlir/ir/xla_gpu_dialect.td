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

#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_DIALECT
#define XLA_SERVICE_GPU_FUSIONS_MLIR_DIALECT

include "mlir/IR/DialectBase.td"

def XlaGpuDialect : Dialect {
  let name = "xla_gpu";

  let description = [{
    This dialect contains ops required for lowering HLO to LLVM.
  }];

  let cppNamespace = "::xla::gpu";
  let useDefaultAttributePrinterParser = 1;
}

#endif // MLIR_HLO_DIALECT_MHLO_IR_HLO_DIALECT
