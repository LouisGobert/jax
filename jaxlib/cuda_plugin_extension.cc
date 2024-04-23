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
#include <Python.h>

#include <string>
#include <string_view>
#include <utility>

#include "nanobind/nanobind.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/py_client_gpu.h"
#include "xla/status.h"
#include "xla/stream_executor/cuda/cuda_driver.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/util.h"

namespace nb = nanobind;

namespace xla {
namespace {
Status RegisterCustomCallTarget(const PJRT_Api* c_api, nb::str fn_name,
                                nb::capsule fn, int api_version) {
  static const char* const kName = "xla._CUSTOM_CALL_TARGET";
  if (std::string_view(fn.name()) != kName) {
    return InvalidArgument(
        "Argument to RegisterCustomCallTargetRegistry was not a "
        "xla._CUSTOM_CALL_TARGET capsule.");
  }

  if (c_api->extension_start == nullptr) {
    return Unimplemented("The plugin does not have extension.");
  }
  const PJRT_Extension_Base* next =
      reinterpret_cast<const PJRT_Extension_Base*>(c_api->extension_start);
  while (next != nullptr &&
         next->type !=
             PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call) {
    next = next->next;
  }
  if (next == nullptr) {
    return Unimplemented("The plugin does not have a custom call extension.");
  }

  PJRT_Gpu_Register_Custom_Call_Args args;
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  args.function_name = fn_name.c_str();
  args.function_name_size = nb::len(fn_name);
#if PJRT_API_GPU_EXTENSION_VERSION >= 1
  args.api_version = api_version;
#endif
  args.custom_call_function = static_cast<void*>(fn.data());
  RETURN_STATUS_IF_PJRT_ERROR(
      reinterpret_cast<const PJRT_Gpu_Custom_Call*>(next)->custom_call(&args),
      c_api);
  return OkStatus();
}

nb::dict Registrations() {
  nb::dict dict;
  dict["xla_python_gpu_callback"] =
      jax::EncapsulateFunction(xla::XlaPythonGpuCallback);
  return dict;
}
}  // namespace

NB_MODULE(cuda_plugin_extension, m) {
  tsl::ImportNumpy();
  m.def(
      "register_custom_call_target",
      [](nb::capsule c_api, nb::str fn_name, nb::capsule fn,
         nb::str xla_platform_name, int api_version) {
        xla::ThrowIfError(
            RegisterCustomCallTarget(static_cast<const PJRT_Api*>(c_api.data()),
                                     fn_name, std::move(fn), api_version));
      },
      nb::arg("c_api"), nb::arg("fn_name"), nb::arg("fn"),
      nb::arg("xla_platform_name"), nb::arg("api_version") = 0);
  m.def("registrations", &Registrations);
  m.def(
      "get_device_ordinal_from_cuda_array_interface",
      [](const nb::dict& cai) {
        if (!cai.contains("data")) {
          xla::ThrowIfError(absl::InvalidArgumentError(
              "CUDA Array Interface does not define `data`"));
        }
        auto data = nb::cast<nb::tuple>(cai["data"]);
        auto data_value = nb::cast<std::intptr_t>(data[0]);
        void* data_ptr = reinterpret_cast<void*>(data_value);
        if (data_value == 0) {
          return 0;
        }
        return stream_executor::gpu::CreatedContexts::GetDeviceOrdinal(
            data_ptr);
      },
      nb::arg("cai"));
}
}  // namespace xla
