# Makefile for openvino-go
# Builds C++ wrapper, CGO bindings, and Go package

.PHONY: all clean test examples install-deps

# Directories
CW_DIR = internal/cwrapper
CGO_DIR = internal/cgo
PKG_DIR = pkg/openvino
EXAMPLES_DIR = examples

# OpenVINO paths (adjust these based on your installation)
OPENVINO_ROOT ?= /opt/intel/openvino
OPENVINO_INCLUDE = $(OPENVINO_ROOT)/runtime/include
OPENVINO_LIB = $(OPENVINO_ROOT)/runtime/lib/intel64

# C++ compiler flags
CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O2 -Wall
INCLUDES = -I$(OPENVINO_INCLUDE)

# Build targets
all: cwrapper cgo

# Build C++ wrapper library
cwrapper:
	@echo "Building C++ wrapper..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(CW_DIR)/core_wrapper.cpp -o $(CW_DIR)/core_wrapper.o
	$(CXX) -shared -o $(CW_DIR)/libopenvino_wrapper.so $(CW_DIR)/core_wrapper.o \
		-L$(OPENVINO_LIB) -lov::runtime

# Build CGO bindings (this will be handled by Go build)
cgo:
	@echo "CGO bindings will be built with 'go build'"

# Run tests
test:
	go test ./...

# Build examples
examples:
	go build -o $(EXAMPLES_DIR)/hello-world/hello-world $(EXAMPLES_DIR)/hello-world/main.go

# Clean build artifacts
clean:
	rm -f $(CW_DIR)/*.o $(CW_DIR)/*.so
	rm -f $(EXAMPLES_DIR)/*/hello-world
	go clean -cache

# Install dependencies (placeholder for now)
install-deps:
	@echo "Please install OpenVINO following: https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html"
