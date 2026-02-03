package openvino

import (
	"context"

	"github.com/accretional/openvino-go/internal/cgo"
)

type InferRequest struct {
	request *cgo.InferRequest
}

func (cm *CompiledModel) CreateInferRequest() (*InferRequest, error) {
	request, err := cm.compiled.CreateInferRequest()
	if err != nil {
		return nil, err
	}
	return &InferRequest{request: request}, nil
}

func (ir *InferRequest) Close() {
	if ir.request != nil {
		ir.request.Destroy()
	}
}

func (ir *InferRequest) SetInputTensor(name string, data interface{}, shape []int64, dataType DataType) error {
	return ir.request.SetInputTensor(name, data, shape, dataType)
}

func (ir *InferRequest) SetInputTensorByIndex(index int32, data interface{}, shape []int64, dataType DataType) error {
	return ir.request.SetInputTensorByIndex(index, data, shape, dataType)
}

func (ir *InferRequest) Infer() error {
	return ir.request.Infer()
}

func (ir *InferRequest) InferWithContext(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	err := ir.request.Infer()

	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return err
	}
}
