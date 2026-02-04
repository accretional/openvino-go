package openvino

import "github.com/accretional/openvino-go/internal/cgo"

type Tensor struct {
	tensor *cgo.Tensor
}

func (ir *InferRequest) GetOutputTensor(name string) (*Tensor, error) {
	tensor, err := ir.request.GetOutputTensor(name)
	if err != nil {
		return nil, err
	}
	return &Tensor{tensor: tensor}, nil
}

func (ir *InferRequest) GetOutputTensorByIndex(index int32) (*Tensor, error) {
	tensor, err := ir.request.GetOutputTensorByIndex(index)
	if err != nil {
		return nil, err
	}
	return &Tensor{tensor: tensor}, nil
}

func (t *Tensor) Close() {
	if t.tensor != nil {
		t.tensor.Destroy()
	}
}

func (t *Tensor) GetDataAsFloat32() ([]float32, error) {
	return t.tensor.GetDataAsFloat32()
}

func (t *Tensor) GetDataAsInt64() ([]int64, error) {
	return t.tensor.GetDataAsInt64()
}

func (t *Tensor) GetDataAsFloat64() ([]float64, error) {
	return t.tensor.GetDataAsFloat64()
}

func (t *Tensor) GetDataAsInt32() ([]int32, error) {
	return t.tensor.GetDataAsInt32()
}

func (t *Tensor) GetDataAsUint8() ([]uint8, error) {
	return t.tensor.GetDataAsUint8()
}

func (t *Tensor) GetDataAsInt8() ([]int8, error) {
	return t.tensor.GetDataAsInt8()
}

func (t *Tensor) GetDataAsUint16() ([]uint16, error) {
	return t.tensor.GetDataAsUint16()
}

func (t *Tensor) GetDataAsInt16() ([]int16, error) {
	return t.tensor.GetDataAsInt16()
}

func (t *Tensor) GetDataAsUint32() ([]uint32, error) {
	return t.tensor.GetDataAsUint32()
}

func (t *Tensor) GetDataAsUint64() ([]uint64, error) {
	return t.tensor.GetDataAsUint64()
}

func (t *Tensor) GetShape() ([]int32, error) {
	return t.tensor.GetShape()
}

// NewTensor creates a new empty tensor with the specified data type and shape.
func NewTensor(dataType DataType, shape []int64) (*Tensor, error) {
	tensor, err := cgo.NewTensor(cgo.DataType(dataType), shape)
	if err != nil {
		return nil, err
	}
	return &Tensor{tensor: tensor}, nil
}

// NewTensorWithData creates a new tensor with the specified data type, shape, and initial data.
func NewTensorWithData(dataType DataType, shape []int64, data interface{}) (*Tensor, error) {
	tensor, err := cgo.NewTensorWithData(cgo.DataType(dataType), shape, data)
	if err != nil {
		return nil, err
	}
	return &Tensor{tensor: tensor}, nil
}

// GetSize returns the total number of elements in the tensor.
func (t *Tensor) GetSize() (int64, error) {
	return t.tensor.GetSize()
}

// GetByteSize returns the size of the tensor in bytes.
func (t *Tensor) GetByteSize() (int64, error) {
	return t.tensor.GetByteSize()
}

// GetElementType returns the data type of the tensor.
func (t *Tensor) GetElementType() (DataType, error) {
	dataType, err := t.tensor.GetElementType()
	if err != nil {
		return 0, err
	}
	return DataType(dataType), nil
}

// SetShape reshapes the tensor to the new shape.
func (t *Tensor) SetShape(shape []int64) error {
	return t.tensor.SetShape(shape)
}
