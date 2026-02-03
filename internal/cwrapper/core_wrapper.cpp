// This wraps OpenVINO C++ API in C functions for CGO

#include "core_wrapper.h"
#include <openvino/openvino.hpp>
#include <string>
#include <vector>
#include <cstring>
#include <memory>
#include <sstream>
#include <map>

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
        case 0: return ov::element::f32;  // float32
        case 1: return ov::element::i64;  // int64
        case 2: return ov::element::i32;  // int32
        case 3: return ov::element::u8;   // uint8
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

void openvino_error_free(OpenVINOError* error) {
    if (error && error->message) {
        free(error->message);
        error->message = nullptr;
    }
}

static int32_t element_type_to_int32(ov::element::Type type) {
    if (type == ov::element::f32) return 0;
    if (type == ov::element::i64) return 1;
    if (type == ov::element::i32) return 2;
    if (type == ov::element::u8) return 3;
    return 0; // Default to float32
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
            
            const auto& shape = input.get_shape();
            result[i].shape_size = static_cast<int32_t>(shape.size());
            result[i].shape = static_cast<int32_t*>(malloc(sizeof(int32_t) * shape.size()));
            for (size_t j = 0; j < shape.size(); j++) {
                result[i].shape[j] = static_cast<int32_t>(shape[j]);
            }
            
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
            
            const auto& shape = output.get_shape();
            result[i].shape_size = static_cast<int32_t>(shape.size());
            result[i].shape = static_cast<int32_t*>(malloc(sizeof(int32_t) * shape.size()));
            for (size_t j = 0; j < shape.size(); j++) {
                result[i].shape[j] = static_cast<int32_t>(shape[j]);
            }
            
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
