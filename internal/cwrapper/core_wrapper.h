// C wrapper header for OpenVINO Core API
// This provides a C interface that CGO can call

#ifndef OPENVINO_CORE_WRAPPER_H
#define OPENVINO_CORE_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Forward declarations
typedef void* OpenVINOCore;
typedef void* OpenVINOModel;
typedef void* OpenVINOCompiledModel;
typedef void* OpenVINOInferRequest;

// Error handling
typedef struct {
    int32_t code;
    char* message;
} OpenVINOError;

// Core API
OpenVINOCore openvino_core_create(OpenVINOError* error);
void openvino_core_destroy(OpenVINOCore core);

// Device enumeration
char** openvino_core_get_available_devices(OpenVINOCore core, int32_t* count, OpenVINOError* error);
void openvino_core_free_device_list(char** devices, int32_t count);

// Model loading
OpenVINOModel openvino_core_read_model(OpenVINOCore core, const char* model_path, OpenVINOError* error);
void openvino_model_destroy(OpenVINOModel model);

// Model compilation
OpenVINOCompiledModel openvino_core_compile_model(
    OpenVINOCore core,
    OpenVINOModel model,
    const char* device,
    OpenVINOError* error
);
void openvino_compiled_model_destroy(OpenVINOCompiledModel compiled_model);

// Infer request
OpenVINOInferRequest openvino_compiled_model_create_infer_request(
    OpenVINOCompiledModel compiled_model,
    OpenVINOError* error
);
void openvino_infer_request_destroy(OpenVINOInferRequest request);

// Error handling
void openvino_error_free(OpenVINOError* error);

#ifdef __cplusplus
}
#endif

#endif // OPENVINO_CORE_WRAPPER_H
