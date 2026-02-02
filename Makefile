# Makefile for openvino-go
# Builds C++ wrapper, CGO bindings, and Go package
# Designed for Linux systems

.PHONY: all clean test examples install-deps check-env

# Directories
CW_DIR = internal/cwrapper
CGO_DIR = internal/cgo
PKG_DIR = pkg/openvino
EXAMPLES_DIR = examples

# OpenVINO paths (adjust these based on your installation)
# For OpenVINO 2024.x, the default path is typically:
# /opt/intel/openvino or /usr/local/openvino
OPENVINO_ROOT ?= /opt/intel/openvino
OPENVINO_INCLUDE = $(OPENVINO_ROOT)/runtime/include
OPENVINO_LIB = $(OPENVINO_ROOT)/runtime/lib/intel64

# Detect architecture (default to intel64, can be overridden)
ARCH ?= intel64
OPENVINO_LIB = $(OPENVINO_ROOT)/runtime/lib/$(ARCH)

# C++ compiler flags
CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O2 -Wall -Wno-deprecated-declarations
INCLUDES = -I$(OPENVINO_INCLUDE)

# Library flags for Linux
LDFLAGS = -L$(OPENVINO_LIB) -lov::runtime -Wl,-rpath,$(OPENVINO_LIB)

# Build targets
all: check-env cwrapper
	@echo "Build complete. CGO bindings will be built automatically when using 'go build'"

# Check if OpenVINO is installed
check-env:
	@if [ ! -d "$(OPENVINO_INCLUDE)" ]; then \
		echo "Error: OpenVINO not found at $(OPENVINO_ROOT)"; \
		echo "Please set OPENVINO_ROOT environment variable or install OpenVINO"; \
		echo "Example: export OPENVINO_ROOT=/opt/intel/openvino"; \
		exit 1; \
	fi
	@if [ ! -d "$(OPENVINO_LIB)" ]; then \
		echo "Warning: OpenVINO library directory not found at $(OPENVINO_LIB)"; \
		echo "You may need to adjust the ARCH variable (current: $(ARCH))"; \
	fi

# Build C++ wrapper library
cwrapper: check-env
	@echo "Building C++ wrapper for Linux..."
	@echo "Using OpenVINO at: $(OPENVINO_ROOT)"
	@echo "Include path: $(OPENVINO_INCLUDE)"
	@echo "Library path: $(OPENVINO_LIB)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(CW_DIR)/core_wrapper.cpp -o $(CW_DIR)/core_wrapper.o
	$(CXX) -shared -o $(CW_DIR)/libopenvino_wrapper.so $(CW_DIR)/core_wrapper.o $(LDFLAGS)
	@echo "C++ wrapper built successfully: $(CW_DIR)/libopenvino_wrapper.so"
	@echo "Note: Make sure LD_LIBRARY_PATH includes $(OPENVINO_LIB) or use rpath"

# Build CGO bindings (this will be handled by Go build)
cgo:
	@echo "CGO bindings will be built with 'go build'"
	@echo "Make sure to build the C++ wrapper first with 'make cwrapper'"

# Run tests
test: cwrapper
	go test ./...

# Build examples
examples: cwrapper
	@echo "Building hello-world example..."
	@echo "Note: Make sure LD_LIBRARY_PATH includes $(OPENVINO_LIB)"
	@echo "Example: export LD_LIBRARY_PATH=\$$LD_LIBRARY_PATH:$(OPENVINO_LIB)"
	go build -o $(EXAMPLES_DIR)/hello-world/hello-world $(EXAMPLES_DIR)/hello-world/main.go
	@echo "Example built: $(EXAMPLES_DIR)/hello-world/hello-world"

# Clean build artifacts
clean:
	rm -f $(CW_DIR)/*.o $(CW_DIR)/*.so
	rm -f $(EXAMPLES_DIR)/*/hello-world
	go clean -cache
	@echo "Clean complete"

# Install dependencies (placeholder for now)
install-deps:
	@echo "Please install OpenVINO following: https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html"
	@echo ""
	@echo "For Linux, you can:"
	@echo "1. Download from: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html"
	@echo "2. Extract to /opt/intel/openvino"
	@echo "3. Run: source /opt/intel/openvino/setupvars.sh"
	@echo "4. Set environment: export OPENVINO_ROOT=/opt/intel/openvino"