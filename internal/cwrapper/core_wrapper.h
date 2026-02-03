// C wrapper header for OpenVINO Core API
// This provides a C interface that CGO can call

#ifndef OPENVINO_CORE_WRAPPER_H
#define OPENVINO_CORE_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Opaque handle types (to handle unsafe pointers)
typedef struct openvino_core* OpenVINOCore;
typedef struct openvino_model* OpenVINOModel;
typedef struct openvino_compiled_model* OpenVINOCompiledModel;
typedef struct openvino_infer_request* OpenVINOInferRequest;
typedef struct openvino_tensor* OpenVINOTensor;
typedef struct openvino_variable_state* OpenVINOVariableState;

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
OpenVINOCompiledModel openvino_core_compile_model_with_properties(
    OpenVINOCore core,
    OpenVINOModel model,
    const char* device,
    const char* property_keys,
    const char* property_values,
    int32_t property_count,
    OpenVINOError* error
);
void openvino_compiled_model_destroy(OpenVINOCompiledModel compiled_model);

// Memory management
int32_t openvino_compiled_model_release_memory(
    OpenVINOCompiledModel compiled_model,
    OpenVINOError* error
);

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

// Batch tensor operations (requires model with batch dimension)
int32_t openvino_infer_request_set_tensors(
    OpenVINOInferRequest request,
    const char* name,
    OpenVINOTensor* tensors,
    int32_t tensor_count,
    OpenVINOError* error
);

int32_t openvino_infer_request_set_tensors_by_index(
    OpenVINOInferRequest request,
    int32_t index,
    OpenVINOTensor* tensors,
    int32_t tensor_count,
    OpenVINOError* error
);

// Output tensor pre-allocation (zero-copy)
int32_t openvino_infer_request_set_output_tensor(
    OpenVINOInferRequest request,
    const char* name,
    OpenVINOTensor tensor,
    OpenVINOError* error
);

int32_t openvino_infer_request_set_output_tensor_by_index(
    OpenVINOInferRequest request,
    int32_t index,
    OpenVINOTensor tensor,
    OpenVINOError* error
);

int32_t openvino_infer_request_infer(OpenVINOInferRequest request, OpenVINOError* error);

// Asynchronous inference operations
int32_t openvino_infer_request_start_async(OpenVINOInferRequest request, OpenVINOError* error);
int32_t openvino_infer_request_wait(OpenVINOInferRequest request, OpenVINOError* error);
int32_t openvino_infer_request_wait_for(OpenVINOInferRequest request, int64_t timeout_ms, OpenVINOError* error);

// Input tensor retrieval
OpenVINOTensor openvino_infer_request_get_input_tensor(
    OpenVINOInferRequest request,
    const char* name,
    OpenVINOError* error
);

OpenVINOTensor openvino_infer_request_get_input_tensor_by_index(
    OpenVINOInferRequest request,
    int32_t index,
    OpenVINOError* error
);

// Output tensor retrieval
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

// Tensor creation
OpenVINOTensor openvino_tensor_new(
    int32_t data_type,
    int32_t* shape,
    int32_t shape_size,
    OpenVINOError* error
);

OpenVINOTensor openvino_tensor_new_with_data(
    int32_t data_type,
    int32_t* shape,
    int32_t shape_size,
    const void* data,
    OpenVINOError* error
);

// Tensor metadata operations
int64_t openvino_tensor_get_size(OpenVINOTensor tensor, OpenVINOError* error);
int64_t openvino_tensor_get_byte_size(OpenVINOTensor tensor, OpenVINOError* error);
int32_t openvino_tensor_get_element_type(OpenVINOTensor tensor, OpenVINOError* error);
int32_t openvino_tensor_set_shape(OpenVINOTensor tensor, int32_t* shape, int32_t shape_size, OpenVINOError* error);

// Model I/O information
typedef struct {
    char* name;
    int32_t* shape;
    int32_t shape_size;
    int32_t data_type;
} OpenVINOPortInfo;

OpenVINOPortInfo* openvino_model_get_inputs(OpenVINOModel model, int32_t* count, OpenVINOError* error);
OpenVINOPortInfo* openvino_model_get_outputs(OpenVINOModel model, int32_t* count, OpenVINOError* error);
void openvino_model_free_port_info(OpenVINOPortInfo* ports, int32_t count);

// Error handling
void openvino_error_free(OpenVINOError* error);

// VariableState operations
int32_t openvino_infer_request_query_state(
    OpenVINOInferRequest request,
    OpenVINOVariableState** states,
    int32_t* state_count,
    OpenVINOError* error
);

int32_t openvino_infer_request_reset_state(
    OpenVINOInferRequest request,
    OpenVINOError* error
);

void openvino_variable_state_destroy(OpenVINOVariableState state);

const char* openvino_variable_state_get_name(
    OpenVINOVariableState state,
    OpenVINOError* error
);

OpenVINOTensor openvino_variable_state_get_state(
    OpenVINOVariableState state,
    OpenVINOError* error
);

int32_t openvino_variable_state_set_state(
    OpenVINOVariableState state,
    OpenVINOTensor tensor,
    OpenVINOError* error
);

int32_t openvino_variable_state_reset(
    OpenVINOVariableState state,
    OpenVINOError* error
);

void openvino_variable_state_free_name(const char* name);

#ifdef __cplusplus
}
#endif

#endif // OPENVINO_CORE_WRAPPER_H
