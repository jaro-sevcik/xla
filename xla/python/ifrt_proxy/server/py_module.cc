// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "xla/python/ifrt_proxy/server/py_module.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/function.h"  // IWYU pragma: keep
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt_proxy/server/grpc_server.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/py_client.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"

namespace nb = ::nanobind;

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

absl::Status GetServer(std::string proxy_server_address, nb_class_ptr<PyClient> client) {
  DCHECK(PyGILState_Check());
  auto ifrt_client = client->shared_ptr_ifrt_client();
  auto factory = [ifrt_client]() -> absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> {
    return ifrt_client;
  };

  TF_ASSIGN_OR_RETURN(auto server, GrpcServer::CreateFromIfrtClientFactory(proxy_server_address, factory));

  std::cerr << "Running server\n";

  server->Wait();

  return absl::OkStatus();
}

}  // namespace

void BuildIfrtProxyServerSubmodule(nb::module_& m) {
  nb::module_ sub_module = m.def_submodule("ifrt_proxy", "IFRT proxy");

  std::cerr << "Hello\n";

  sub_module.def("get_server", xla::ThrowIfErrorWrapper(GetServer),
                 nb::arg("proxy_server_address"), nb::arg("local_client"));
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
