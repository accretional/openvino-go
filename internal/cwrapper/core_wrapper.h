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
typedef void* OpenVINOTensor;

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

// Inference operations
int32_t openvino_infer_request_set_input_tensor(
    OpenVINOInferRequest request,
    const char* name,
    const void* data,
    int32_t* shape,
    int32_t shape_size,
    int32_t data_type,  // 0=float32, 1=int64, 2=int32, 3=uint8
    OpenVINOError* error
);

int32_t openvino_infer_request_set_input_tensor_by_index(
    OpenVINOInferRequest request,
    int32_t index,
    const void* data,
    int32_t* shape,
    int32_t shape_size,
    int32_t data_type,
    OpenVINOError* error
);

int32_t openvino_infer_request_infer(OpenVINOInferRequest request, OpenVINOError* error);

OpenVINOTensor openvino_infer_request_get_output_tensor(
    OpenVINOInferRequest request,
    const char* name,
    OpenVINOError* error
);

OpenVINOTensor openvino_infer_request_get_output_tensor_by_index(
    OpenVINOInferRequest request,
    int32_t index,
    OpenVINOError* error
);

// Tensor operations
void* openvino_tensor_get_data(OpenVINOTensor tensor, int32_t* data_type, OpenVINOError* error);
int32_t* openvino_tensor_get_shape(OpenVINOTensor tensor, int32_t* shape_size, OpenVINOError* error);
void openvino_tensor_free_shape(int32_t* shape);
void openvino_tensor_destroy(OpenVINOTensor tensor);

// Error handling
void openvino_error_free(OpenVINOError* error);

#ifdef __cplusplus
}
#endif

#endif // OPENVINO_CORE_WRAPPER_H
