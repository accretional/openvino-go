// Package openvino provides idiomatic Go bindings for Intel OpenVINO Runtime
package openvino

import (
	"github.com/accretional/openvino-go/internal/cgo"
)

// Core represents an OpenVINO Core instance
type Core struct {
	core *cgo.Core
}

// NewCore creates a new OpenVINO Core instance
func NewCore() (*Core, error) {
	core, err := cgo.CreateCore()
	if err != nil {
		return nil, err
	}
	return &Core{core: core}, nil
}

// Close releases the Core instance
func (c *Core) Close() {
	if c.core != nil {
		c.core.Destroy()
	}
}

// GetAvailableDevices returns a list of available devices
func (c *Core) GetAvailableDevices() ([]string, error) {
	return c.core.GetAvailableDevices()
}

// Model represents an OpenVINO Model
type Model struct {
	model *cgo.Model
}

// ReadModel loads a model from a file path
func (c *Core) ReadModel(modelPath string) (*Model, error) {
	model, err := c.core.ReadModel(modelPath)
	if err != nil {
		return nil, err
	}
	return &Model{model: model}, nil
}

// Close releases the Model instance
func (m *Model) Close() {
	if m.model != nil {
		m.model.Destroy()
	}
}

// CompiledModel represents a compiled OpenVINO model
type CompiledModel struct {
	compiled *cgo.CompiledModel
}

// CompileModel compiles a model for a specific device
func (c *Core) CompileModel(model *Model, device string) (*CompiledModel, error) {
	compiled, err := c.core.CompileModel(model.model, device)
	if err != nil {
		return nil, err
	}
	return &CompiledModel{compiled: compiled}, nil
}

// Close releases the CompiledModel instance
func (cm *CompiledModel) Close() {
	if cm.compiled != nil {
		cm.compiled.Destroy()
	}
}

// InferRequest represents an inference request
type InferRequest struct {
	request *cgo.InferRequest
}

// CreateInferRequest creates a new inference request
func (cm *CompiledModel) CreateInferRequest() (*InferRequest, error) {
	request, err := cm.compiled.CreateInferRequest()
	if err != nil {
		return nil, err
	}
	return &InferRequest{request: request}, nil
}

// Close releases the InferRequest instance
func (ir *InferRequest) Close() {
	if ir.request != nil {
		ir.request.Destroy()
	}
}

// DataType represents the data type of a tensor
type DataType = cgo.DataType

const (
	DataTypeFloat32 = cgo.DataTypeFloat32
	DataTypeInt64   = cgo.DataTypeInt64
	DataTypeInt32   = cgo.DataTypeInt32
	DataTypeUint8   = cgo.DataTypeUint8
)

// SetInputTensor sets an input tensor by name
func (ir *InferRequest) SetInputTensor(name string, data interface{}, shape []int64, dataType DataType) error {
	return ir.request.SetInputTensor(name, data, shape, dataType)
}

// SetInputTensorByIndex sets an input tensor by index
func (ir *InferRequest) SetInputTensorByIndex(index int32, data interface{}, shape []int64, dataType DataType) error {
	return ir.request.SetInputTensorByIndex(index, data, shape, dataType)
}

// Infer runs synchronous inference
func (ir *InferRequest) Infer() error {
	return ir.request.Infer()
}

// Tensor represents an OpenVINO tensor
type Tensor struct {
	tensor *cgo.Tensor
}

// GetOutputTensor gets an output tensor by name
func (ir *InferRequest) GetOutputTensor(name string) (*Tensor, error) {
	tensor, err := ir.request.GetOutputTensor(name)
	if err != nil {
		return nil, err
	}
	return &Tensor{tensor: tensor}, nil
}

// GetOutputTensorByIndex gets an output tensor by index
func (ir *InferRequest) GetOutputTensorByIndex(index int32) (*Tensor, error) {
	tensor, err := ir.request.GetOutputTensorByIndex(index)
	if err != nil {
		return nil, err
	}
	return &Tensor{tensor: tensor}, nil
}

// Close releases the Tensor instance
func (t *Tensor) Close() {
	if t.tensor != nil {
		t.tensor.Destroy()
	}
}

// GetDataAsFloat32 returns the tensor data as float32 slice
func (t *Tensor) GetDataAsFloat32() ([]float32, error) {
	return t.tensor.GetDataAsFloat32()
}

// GetDataAsInt64 returns the tensor data as int64 slice
func (t *Tensor) GetDataAsInt64() ([]int64, error) {
	return t.tensor.GetDataAsInt64()
}

// GetShape returns the tensor shape
func (t *Tensor) GetShape() ([]int32, error) {
	return t.tensor.GetShape()
}
