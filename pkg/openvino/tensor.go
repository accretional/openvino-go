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

func (t *Tensor) GetShape() ([]int32, error) {
	return t.tensor.GetShape()
}
