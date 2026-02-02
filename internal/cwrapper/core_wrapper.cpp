// C++ wrapper implementation for OpenVINO Core API
// This wraps OpenVINO C++ API in C functions for CGO

#include "core_wrapper.h"
#include <openvino/openvino.hpp>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

// Helper function to convert C++ exception to error
static void set_error(OpenVINOError* error, int32_t code, const char* message) {
    if (error) {
        error->code = code;
        if (error->message) {
            free(error->message);
        }
        error->message = strdup(message);
    }
}

// Helper function to set error from exception
static void set_error_from_exception(OpenVINOError* error, const std::exception& e) {
    set_error(error, -1, e.what());
}

extern "C" {

OpenVINOCore openvino_core_create(OpenVINOError* error) {
    try {
        ov::Core* core = new ov::Core();
        return static_cast<OpenVINOCore>(core);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

void openvino_core_destroy(OpenVINOCore core) {
    if (core) {
        delete static_cast<ov::Core*>(core);
    }
}

char** openvino_core_get_available_devices(OpenVINOCore core, int32_t* count, OpenVINOError* error) {
    try {
        ov::Core* c = static_cast<ov::Core*>(core);
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
        ov::Core* c = static_cast<ov::Core*>(core);
        std::shared_ptr<ov::Model>* model = new std::shared_ptr<ov::Model>(
            c->read_model(model_path)
        );
        return static_cast<OpenVINOModel>(model);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

void openvino_model_destroy(OpenVINOModel model) {
    if (model) {
        delete static_cast<std::shared_ptr<ov::Model>*>(model);
    }
}

OpenVINOCompiledModel openvino_core_compile_model(
    OpenVINOCore core,
    OpenVINOModel model,
    const char* device,
    OpenVINOError* error
) {
    try {
        ov::Core* c = static_cast<ov::Core*>(core);
        std::shared_ptr<ov::Model>* m = static_cast<std::shared_ptr<ov::Model>*>(model);
        
        std::shared_ptr<ov::CompiledModel>* compiled = new std::shared_ptr<ov::CompiledModel>(
            c->compile_model(*m, device)
        );
        return static_cast<OpenVINOCompiledModel>(compiled);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

void openvino_compiled_model_destroy(OpenVINOCompiledModel compiled_model) {
    if (compiled_model) {
        delete static_cast<std::shared_ptr<ov::CompiledModel>*>(compiled_model);
    }
}

OpenVINOInferRequest openvino_compiled_model_create_infer_request(
    OpenVINOCompiledModel compiled_model,
    OpenVINOError* error
) {
    try {
        std::shared_ptr<ov::CompiledModel>* cm = 
            static_cast<std::shared_ptr<ov::CompiledModel>*>(compiled_model);
        
        ov::InferRequest* request = new ov::InferRequest(
            cm->get()->create_infer_request()
        );
        return static_cast<OpenVINOInferRequest>(request);
    } catch (const std::exception& e) {
        set_error_from_exception(error, e);
        return nullptr;
    }
}

void openvino_infer_request_destroy(OpenVINOInferRequest request) {
    if (request) {
        delete static_cast<ov::InferRequest*>(request);
    }
}

void openvino_error_free(OpenVINOError* error) {
    if (error && error->message) {
        free(error->message);
        error->message = nullptr;
    }
}

} // extern "C"
