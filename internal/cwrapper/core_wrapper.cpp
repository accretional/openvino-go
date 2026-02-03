// This wraps OpenVINO C++ API in C functions for CGO

#include "core_wrapper.h"
#include <openvino/openvino.hpp>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <sstream>
#include <map>
#include <chrono>

static void set_error(OpenVINOError* error, int32_t code, const char* message) {
    if (error) {
        error->code = code;
        if (error->message) {
            free(error->message);
        }
        error->message = strdup(message);
    }
}

static void set_error_from_exception(OpenVINOError* error, const std::exception& e) {
    set_error(error, -1, e.what());
}

static ov::element::Type get_element_type(int32_t data_type) {
    switch (data_type) {
        case 0: return ov::element::f32;      // float32
        case 1: return ov::element::i64;       // int64
        case 2: return ov::element::i32;       // int32
        case 3: return ov::element::u8;        // uint8
        case 4: return ov::element::f64;       // float64
        case 5: return ov::element::i8;        // int8
        case 6: return ov::element::u16;        // uint16
        case 7: return ov::element::i16;        // int16
        case 8: return ov::element::u32;        // uint32
        case 9: return ov::element::u64;        // uint64
        case 10: return ov::element::f16;      // float16
        case 11: return ov::element::bf16;      // bfloat16
        default: return ov::element::f32;
    }
}


static size_t calculate_total_elements(const int32_t* shape, int32_t shape_size) {
    size_t total = 1;
    for (int32_t i = 0; i < shape_size; i++) {
        total *= static_cast<size_t>(shape[i]);
    }
    return total;
}

static int32_t element_type_to_int32(ov::element::Type type) {
    if (type == ov::element::f32) return 0;
    if (type == ov::element::i64) return 1;
    if (type == ov::element::i32) return 2;
    if (type == ov::element::u8) return 3;
    if (type == ov::element::f64) return 4;
    if (type == ov::element::i8) return 5;
    if (type == ov::element::u16) return 6;
    if (type == ov::element::i16) return 7;
    if (type == ov::element::u32) return 8;
    if (type == ov::element::u64) return 9;
    if (type == ov::element::f16) return 10;
    if (type == ov::element::bf16) return 11;
    return 0; // Default to float32
}

extern "C" {

OpenVINOCore openvino_core_create(OpenVINOError* error) {
    try {
        ov::Core* core = new ov::Core();
        return reinterpret_cast<OpenVINOCore>(core);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

void openvino_core_destroy(OpenVINOCore core) {
    if (core) {
        delete reinterpret_cast<ov::Core*>(core);
    }
}

char** openvino_core_get_available_devices(OpenVINOCore core, int32_t* count, OpenVINOError* error) {
    try {
        ov::Core* c = reinterpret_cast<ov::Core*>(core);
        std::vector<std::string> devices = c->get_available_devices();

        *count = static_cast<int32_t>(devices.size());
        char** result = static_cast<char**>(malloc(sizeof(char*) * devices.size()));

        for (size_t i = 0; i < devices.size(); i++) {
            result[i] = strdup(devices[i].c_str());
        }

        return result;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        *count = 0;
        return nullptr;
    }
}

void openvino_core_free_device_list(char** devices, int32_t count) {
    if (devices) {
        for (int32_t i = 0; i < count; i++) {
            free(devices[i]);
        }
        free(devices);
    }
}

OpenVINOModel openvino_core_read_model(OpenVINOCore core, const char* model_path, OpenVINOError* error) {
    try {
        ov::Core* c = reinterpret_cast<ov::Core*>(core);
        std::shared_ptr<ov::Model>* model = new std::shared_ptr<ov::Model>(
            c->read_model(model_path)
        );
        return reinterpret_cast<OpenVINOModel>(model);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

void openvino_model_destroy(OpenVINOModel model) {
    if (model) {
        delete reinterpret_cast<std::shared_ptr<ov::Model>*>(model);
    }
}

OpenVINOCompiledModel openvino_core_compile_model(
    OpenVINOCore core,
    OpenVINOModel model,
    const char* device,
    OpenVINOError* error
) {
    try {
        ov::Core* c = reinterpret_cast<ov::Core*>(core);
        std::shared_ptr<ov::Model>* m = reinterpret_cast<std::shared_ptr<ov::Model>*>(model);

        ov::CompiledModel* compiled = new ov::CompiledModel(
            c->compile_model(*m, device)
        );
        return reinterpret_cast<OpenVINOCompiledModel>(compiled);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

OpenVINOCompiledModel openvino_core_compile_model_with_properties(
    OpenVINOCore core,
    OpenVINOModel model,
    const char* device,
    const char* property_keys,
    const char* property_values,
    int32_t property_count,
    OpenVINOError* error
) {
    try {
        ov::Core* c = reinterpret_cast<ov::Core*>(core);
        std::shared_ptr<ov::Model>* m = reinterpret_cast<std::shared_ptr<ov::Model>*>(model);

        // Parse property keys and values (comma-separated strings)
        std::map<std::string, std::string> props;
        std::istringstream keys_stream(property_keys);
        std::istringstream values_stream(property_values);
        std::string key, value;
        
        for (int32_t i = 0; i < property_count; i++) {
            if (!std::getline(keys_stream, key, ',') || !std::getline(values_stream, value, ',')) {
                set_error(error, -1, "Invalid property format");
                return nullptr;
            }
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            props[key] = value;
        }

        // Build ov::AnyMap from properties
        ov::AnyMap config;
        for (const auto& prop : props) {
            // Handle PERFORMANCE_HINT as string
            if (prop.first == "PERFORMANCE_HINT") {
                config[prop.first] = prop.second; // "LATENCY" or "THROUGHPUT"
            }
            // Handle numeric properties
            else if (prop.first == "INFERENCE_NUM_THREADS" || prop.first == "NUM_STREAMS") {
                try {
                    int32_t int_val = std::stoi(prop.second);
                    config[prop.first] = int_val;
                } catch (...) {
                    // If not an integer, use as string
                    config[prop.first] = prop.second;
                }
            }
            // Handle other properties that might be numeric
            else if (prop.first.find("STREAM") != std::string::npos || 
                     prop.first.find("THREAD") != std::string::npos) {
                try {
                    int32_t int_val = std::stoi(prop.second);
                    config[prop.first] = int_val;
                } catch (...) {
                    config[prop.first] = prop.second;
                }
            }
            // Default: use as string
            else {
                config[prop.first] = prop.second;
            }
        }

        ov::CompiledModel* compiled = new ov::CompiledModel(
            c->compile_model(*m, device, config)
        );
        return reinterpret_cast<OpenVINOCompiledModel>(compiled);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

void openvino_compiled_model_destroy(OpenVINOCompiledModel compiled_model) {
    if (compiled_model) {
        delete reinterpret_cast<ov::CompiledModel*>(compiled_model);
    }
}

int32_t openvino_compiled_model_release_memory(
    OpenVINOCompiledModel compiled_model,
    OpenVINOError* error
) {
    try {
        ov::CompiledModel* cm = reinterpret_cast<ov::CompiledModel*>(compiled_model);
        cm->release_memory();
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

OpenVINOInferRequest openvino_compiled_model_create_infer_request(
    OpenVINOCompiledModel compiled_model,
    OpenVINOError* error
) {
    try {
        ov::CompiledModel* cm = reinterpret_cast<ov::CompiledModel*>(compiled_model);

        ov::InferRequest* request = new ov::InferRequest(
            cm->create_infer_request()
        );
        return reinterpret_cast<OpenVINOInferRequest>(request);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

void openvino_infer_request_destroy(OpenVINOInferRequest request) {
    if (request) {
        delete reinterpret_cast<ov::InferRequest*>(request);
    }
}

int32_t openvino_infer_request_set_input_tensor(
    OpenVINOInferRequest request,
    const char* name,
    const void* data,
    int32_t* shape,
    int32_t shape_size,
    int32_t data_type,
    OpenVINOError* error
) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);

        ov::Shape ov_shape;
        for (int32_t i = 0; i < shape_size; i++) {
            ov_shape.push_back(static_cast<size_t>(shape[i]));
        }

        ov::element::Type element_type = get_element_type(data_type);

        size_t total_elements = calculate_total_elements(shape, shape_size);
        size_t data_size = total_elements * element_type.size();

        ov::Tensor tensor(element_type, ov_shape);
        std::memcpy(tensor.data(), data, data_size);

        req->set_tensor(name, tensor);

        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int32_t openvino_infer_request_set_input_tensor_by_index(
    OpenVINOInferRequest request,
    int32_t index,
    const void* data,
    int32_t* shape,
    int32_t shape_size,
    int32_t data_type,
    OpenVINOError* error
) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);

        ov::Shape ov_shape;
        for (int32_t i = 0; i < shape_size; i++) {
            ov_shape.push_back(static_cast<size_t>(shape[i]));
        }

        ov::element::Type element_type = get_element_type(data_type);

        size_t total_elements = calculate_total_elements(shape, shape_size);
        size_t data_size = total_elements * element_type.size();

        ov::Tensor tensor(element_type, ov_shape);
        std::memcpy(tensor.data(), data, data_size);

        req->set_input_tensor(static_cast<size_t>(index), tensor);

        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int32_t openvino_infer_request_infer(OpenVINOInferRequest request, OpenVINOError* error) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        req->infer();
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int32_t openvino_infer_request_set_tensors(
    OpenVINOInferRequest request,
    const char* name,
    OpenVINOTensor* tensors,
    int32_t tensor_count,
    OpenVINOError* error
) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        
        std::vector<ov::Tensor> ov_tensors;
        ov_tensors.reserve(tensor_count);
        
        for (int32_t i = 0; i < tensor_count; i++) {
            ov::Tensor* t = reinterpret_cast<ov::Tensor*>(tensors[i]);
            ov_tensors.push_back(*t);
        }
        
        req->set_tensors(name, ov_tensors);
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int32_t openvino_infer_request_set_tensors_by_index(
    OpenVINOInferRequest request,
    int32_t index,
    OpenVINOTensor* tensors,
    int32_t tensor_count,
    OpenVINOError* error
) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        
        std::vector<ov::Tensor> ov_tensors;
        ov_tensors.reserve(tensor_count);
        
        for (int32_t i = 0; i < tensor_count; i++) {
            ov::Tensor* t = reinterpret_cast<ov::Tensor*>(tensors[i]);
            ov_tensors.push_back(*t);
        }
        
        // Get the input port by index
        auto inputs = req->get_compiled_model().inputs();
        if (index < 0 || static_cast<size_t>(index) >= inputs.size()) {
            set_error(error, -1, "Invalid input index");
            return -1;
        }
        
        req->set_tensors(inputs[index], ov_tensors);
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int32_t openvino_infer_request_set_output_tensor(
    OpenVINOInferRequest request,
    const char* name,
    OpenVINOTensor tensor,
    OpenVINOError* error
) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        ov::Tensor* t = reinterpret_cast<ov::Tensor*>(tensor);
        
        // Use set_tensor which works for both input and output by name
        req->set_tensor(name, *t);
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int32_t openvino_infer_request_set_output_tensor_by_index(
    OpenVINOInferRequest request,
    int32_t index,
    OpenVINOTensor tensor,
    OpenVINOError* error
) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        ov::Tensor* t = reinterpret_cast<ov::Tensor*>(tensor);
        
        req->set_output_tensor(static_cast<size_t>(index), *t);
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int32_t openvino_infer_request_start_async(OpenVINOInferRequest request, OpenVINOError* error) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        req->start_async();
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int32_t openvino_infer_request_wait(OpenVINOInferRequest request, OpenVINOError* error) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        req->wait();
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int32_t openvino_infer_request_wait_for(OpenVINOInferRequest request, int64_t timeout_ms, OpenVINOError* error) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        bool completed = req->wait_for(std::chrono::milliseconds(timeout_ms));
        return completed ? 0 : 1; // 0 = completed, 1 = timeout
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

OpenVINOTensor openvino_infer_request_get_input_tensor(
    OpenVINOInferRequest request,
    const char* name,
    OpenVINOError* error
) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        ov::Tensor tensor = req->get_tensor(name);

        ov::Tensor* tensor_ptr = new ov::Tensor(tensor);
        return reinterpret_cast<OpenVINOTensor>(tensor_ptr);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

OpenVINOTensor openvino_infer_request_get_input_tensor_by_index(
    OpenVINOInferRequest request,
    int32_t index,
    OpenVINOError* error
) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        ov::Tensor tensor = req->get_input_tensor(static_cast<size_t>(index));

        ov::Tensor* tensor_ptr = new ov::Tensor(tensor);
        return reinterpret_cast<OpenVINOTensor>(tensor_ptr);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

OpenVINOTensor openvino_infer_request_get_output_tensor(
    OpenVINOInferRequest request,
    const char* name,
    OpenVINOError* error
) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        ov::Tensor tensor = req->get_tensor(name);

        ov::Tensor* tensor_ptr = new ov::Tensor(tensor);
        return reinterpret_cast<OpenVINOTensor>(tensor_ptr);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

OpenVINOTensor openvino_infer_request_get_output_tensor_by_index(
    OpenVINOInferRequest request,
    int32_t index,
    OpenVINOError* error
) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        ov::Tensor tensor = req->get_output_tensor(static_cast<size_t>(index));

        ov::Tensor* tensor_ptr = new ov::Tensor(tensor);
        return reinterpret_cast<OpenVINOTensor>(tensor_ptr);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

void* openvino_tensor_get_data(OpenVINOTensor tensor, int32_t* data_type, OpenVINOError* error) {
    try {
        ov::Tensor* t = reinterpret_cast<ov::Tensor*>(tensor);

        ov::element::Type element_type = t->get_element_type();
        if (element_type == ov::element::f32) {
            *data_type = 0; // float32
        } else if (element_type == ov::element::i64) {
            *data_type = 1; // int64
        } else if (element_type == ov::element::i32) {
            *data_type = 2; // int32
        } else if (element_type == ov::element::u8) {
            *data_type = 3; // uint8
        } else {
            *data_type = 0; // Default to float32
        }

        return t->data();
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

int32_t* openvino_tensor_get_shape(OpenVINOTensor tensor, int32_t* shape_size, OpenVINOError* error) {
    try {
        ov::Tensor* t = reinterpret_cast<ov::Tensor*>(tensor);
        ov::Shape shape = t->get_shape();

        *shape_size = static_cast<int32_t>(shape.size());
        int32_t* result = static_cast<int32_t*>(malloc(sizeof(int32_t) * shape.size()));

        for (size_t i = 0; i < shape.size(); i++) {
            result[i] = static_cast<int32_t>(shape[i]);
        }

        return result;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        *shape_size = 0;
        return nullptr;
    }
}

void openvino_tensor_free_shape(int32_t* shape) {
    if (shape) {
        free(shape);
    }
}

void openvino_tensor_destroy(OpenVINOTensor tensor) {
    if (tensor) {
        delete reinterpret_cast<ov::Tensor*>(tensor);
    }
}

OpenVINOTensor openvino_tensor_new(
    int32_t data_type,
    int32_t* shape,
    int32_t shape_size,
    OpenVINOError* error
) {
    try {
        ov::element::Type element_type = get_element_type(data_type);
        
        ov::Shape ov_shape;
        for (int32_t i = 0; i < shape_size; i++) {
            ov_shape.push_back(static_cast<size_t>(shape[i]));
        }
        
        ov::Tensor* tensor = new ov::Tensor(element_type, ov_shape);
        return reinterpret_cast<OpenVINOTensor>(tensor);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

OpenVINOTensor openvino_tensor_new_with_data(
    int32_t data_type,
    int32_t* shape,
    int32_t shape_size,
    const void* data,
    OpenVINOError* error
) {
    try {
        ov::element::Type element_type = get_element_type(data_type);
        
        ov::Shape ov_shape;
        for (int32_t i = 0; i < shape_size; i++) {
            ov_shape.push_back(static_cast<size_t>(shape[i]));
        }
        
        size_t total_elements = calculate_total_elements(shape, shape_size);
        size_t data_size = total_elements * element_type.size();
        
        ov::Tensor* tensor = new ov::Tensor(element_type, ov_shape);
        std::memcpy(tensor->data(), data, data_size);
        
        return reinterpret_cast<OpenVINOTensor>(tensor);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

int64_t openvino_tensor_get_size(OpenVINOTensor tensor, OpenVINOError* error) {
    try {
        ov::Tensor* t = reinterpret_cast<ov::Tensor*>(tensor);
        return static_cast<int64_t>(t->get_size());
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int64_t openvino_tensor_get_byte_size(OpenVINOTensor tensor, OpenVINOError* error) {
    try {
        ov::Tensor* t = reinterpret_cast<ov::Tensor*>(tensor);
        return static_cast<int64_t>(t->get_byte_size());
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int32_t openvino_tensor_get_element_type(OpenVINOTensor tensor, OpenVINOError* error) {
    try {
        ov::Tensor* t = reinterpret_cast<ov::Tensor*>(tensor);
        ov::element::Type type = t->get_element_type();
        return element_type_to_int32(type);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int32_t openvino_tensor_set_shape(OpenVINOTensor tensor, int32_t* shape, int32_t shape_size, OpenVINOError* error) {
    try {
        ov::Tensor* t = reinterpret_cast<ov::Tensor*>(tensor);
        
        ov::Shape ov_shape;
        for (int32_t i = 0; i < shape_size; i++) {
            ov_shape.push_back(static_cast<size_t>(shape[i]));
        }
        
        t->set_shape(ov_shape);
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

void openvino_error_free(OpenVINOError* error) {
    if (error && error->message) {
        free(error->message);
        error->message = nullptr;
    }
}

int32_t openvino_infer_request_query_state(
    OpenVINOInferRequest request,
    OpenVINOVariableState** states,
    int32_t* state_count,
    OpenVINOError* error
) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        std::vector<ov::VariableState> variable_states = req->query_state();
        
        *state_count = static_cast<int32_t>(variable_states.size());
        
        if (variable_states.empty()) {
            *states = nullptr;
            return 0;
        }
        
        // Allocate array of VariableState pointers
        OpenVINOVariableState* result = static_cast<OpenVINOVariableState*>(
            malloc(sizeof(OpenVINOVariableState) * variable_states.size())
        );
        
        // Create new VariableState objects and store pointers
        for (size_t i = 0; i < variable_states.size(); i++) {
            ov::VariableState* vs = new ov::VariableState(std::move(variable_states[i]));
            result[i] = reinterpret_cast<OpenVINOVariableState>(vs);
        }
        
        *states = result;
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        *state_count = 0;
        *states = nullptr;
        return -1;
    }
}

int32_t openvino_infer_request_reset_state(
    OpenVINOInferRequest request,
    OpenVINOError* error
) {
    try {
        ov::InferRequest* req = reinterpret_cast<ov::InferRequest*>(request);
        req->reset_state();
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

void openvino_variable_state_destroy(OpenVINOVariableState state) {
    if (state) {
        delete reinterpret_cast<ov::VariableState*>(state);
    }
}

const char* openvino_variable_state_get_name(
    OpenVINOVariableState state,
    OpenVINOError* error
) {
    try {
        ov::VariableState* vs = reinterpret_cast<ov::VariableState*>(state);
        std::string name = vs->get_name();
        return strdup(name.c_str());
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

OpenVINOTensor openvino_variable_state_get_state(
    OpenVINOVariableState state,
    OpenVINOError* error
) {
    try {
        ov::VariableState* vs = reinterpret_cast<ov::VariableState*>(state);
        ov::Tensor tensor = vs->get_state();
        
        // Create a new Tensor object to return
        ov::Tensor* tensor_ptr = new ov::Tensor(std::move(tensor));
        return reinterpret_cast<OpenVINOTensor>(tensor_ptr);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

int32_t openvino_variable_state_set_state(
    OpenVINOVariableState state,
    OpenVINOTensor tensor,
    OpenVINOError* error
) {
    try {
        ov::VariableState* vs = reinterpret_cast<ov::VariableState*>(state);
        ov::Tensor* t = reinterpret_cast<ov::Tensor*>(tensor);
        
        vs->set_state(*t);
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

int32_t openvino_variable_state_reset(
    OpenVINOVariableState state,
    OpenVINOError* error
) {
    try {
        ov::VariableState* vs = reinterpret_cast<ov::VariableState*>(state);
        vs->reset();
        return 0;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return -1;
    }
}

void openvino_variable_state_free_name(const char* name) {
    if (name) {
        free(const_cast<char*>(name));
    }
}

// Helper: fill shape from PartialShape; use -1 for dynamic dimensions
static void fill_shape_from_partial(ov::PartialShape ps, int32_t* shape_out, int32_t* shape_size) {
    *shape_size = static_cast<int32_t>(ps.size());
    for (size_t j = 0; j < ps.size(); j++) {
        const ov::Dimension& d = ps[j];
        shape_out[j] = d.is_dynamic() ? -1 : static_cast<int32_t>(d.get_length());
    }
}

OpenVINOPortInfo* openvino_model_get_inputs(OpenVINOModel model, int32_t* count, OpenVINOError* error) {
    try {
        std::shared_ptr<ov::Model>* m = reinterpret_cast<std::shared_ptr<ov::Model>*>(model);
        const auto& inputs = (*m)->inputs();
        
        *count = static_cast<int32_t>(inputs.size());
        OpenVINOPortInfo* result = static_cast<OpenVINOPortInfo*>(malloc(sizeof(OpenVINOPortInfo) * inputs.size()));
        
        for (size_t i = 0; i < inputs.size(); i++) {
            const auto& input = inputs[i];
            result[i].name = strdup(input.get_any_name().c_str());
            
            ov::PartialShape ps = input.get_partial_shape();
            result[i].shape_size = static_cast<int32_t>(ps.size());
            result[i].shape = static_cast<int32_t*>(malloc(sizeof(int32_t) * ps.size()));
            fill_shape_from_partial(ps, result[i].shape, &result[i].shape_size);
            
            result[i].data_type = element_type_to_int32(input.get_element_type());
        }
        
        return result;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        *count = 0;
        return nullptr;
    }
}

OpenVINOPortInfo* openvino_model_get_outputs(OpenVINOModel model, int32_t* count, OpenVINOError* error) {
    try {
        std::shared_ptr<ov::Model>* m = reinterpret_cast<std::shared_ptr<ov::Model>*>(model);
        const auto& outputs = (*m)->outputs();
        
        *count = static_cast<int32_t>(outputs.size());
        OpenVINOPortInfo* result = static_cast<OpenVINOPortInfo*>(malloc(sizeof(OpenVINOPortInfo) * outputs.size()));
        
        for (size_t i = 0; i < outputs.size(); i++) {
            const auto& output = outputs[i];
            result[i].name = strdup(output.get_any_name().c_str());
            
            ov::PartialShape ps = output.get_partial_shape();
            result[i].shape_size = static_cast<int32_t>(ps.size());
            result[i].shape = static_cast<int32_t*>(malloc(sizeof(int32_t) * ps.size()));
            fill_shape_from_partial(ps, result[i].shape, &result[i].shape_size);
            
            result[i].data_type = element_type_to_int32(output.get_element_type());
        }
        
        return result;
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        *count = 0;
        return nullptr;
    }
}

void openvino_model_free_port_info(OpenVINOPortInfo* ports, int32_t count) {
    if (ports) {
        for (int32_t i = 0; i < count; i++) {
            if (ports[i].name) {
                free(ports[i].name);
            }
            if (ports[i].shape) {
                free(ports[i].shape);
            }
        }
        free(ports);
    }
}

} // extern "C"
