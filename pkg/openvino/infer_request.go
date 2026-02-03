package openvino

import (
	"context"
	"errors"

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

// StartAsync starts asynchronous inference. The inference runs in the background.
// Use Wait() or WaitFor() to wait for completion.
func (ir *InferRequest) StartAsync() error {
	return ir.request.StartAsync()
}

// Wait waits for asynchronous inference to complete. This blocks until inference is done.
func (ir *InferRequest) Wait() error {
	return ir.request.Wait()
}

// WaitFor waits for asynchronous inference to complete with a timeout.
// Returns true if inference completed, false if timeout occurred.
func (ir *InferRequest) WaitFor(timeoutMs int64) (bool, error) {
	return ir.request.WaitFor(timeoutMs)
}

// InferAsync starts asynchronous inference and waits for completion.
// This is a convenience method that combines StartAsync() and Wait().
func (ir *InferRequest) InferAsync() error {
	if err := ir.StartAsync(); err != nil {
		return err
	}
	return ir.Wait()
}

// InferAsyncWithContext starts asynchronous inference and waits for completion with context support.
// This is a convenience method that combines StartAsync() and Wait() with context cancellation.
func (ir *InferRequest) InferAsyncWithContext(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	if err := ir.StartAsync(); err != nil {
		return err
	}

	// Wait for completion, checking context periodically
	done := make(chan error, 1)
	go func() {
		done <- ir.Wait()
	}()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		return err
	}
}

// GetInputTensor retrieves an input tensor by name.
func (ir *InferRequest) GetInputTensor(name string) (*Tensor, error) {
	tensor, err := ir.request.GetInputTensor(name)
	if err != nil {
		return nil, err
	}
	return &Tensor{tensor: tensor}, nil
}

// GetInputTensorByIndex retrieves an input tensor by index.
func (ir *InferRequest) GetInputTensorByIndex(index int32) (*Tensor, error) {
	tensor, err := ir.request.GetInputTensorByIndex(index)
	if err != nil {
		return nil, err
	}
	return &Tensor{tensor: tensor}, nil
}

// SetInputTensors sets a batch of input tensors by name.
// Requires model with batch dimension; number of tensors must match batch size.
func (ir *InferRequest) SetInputTensors(name string, tensors []*Tensor) error {
	cgoTensors := make([]*cgo.Tensor, len(tensors))
	for i, t := range tensors {
		if t == nil {
			return errors.New("tensor cannot be nil")
		}
		cgoTensors[i] = t.tensor
	}
	return ir.request.SetInputTensors(name, cgoTensors)
}

// SetInputTensorsByIndex sets a batch of input tensors by index.
// Requires model with batch dimension; number of tensors must match batch size.
func (ir *InferRequest) SetInputTensorsByIndex(index int32, tensors []*Tensor) error {
	cgoTensors := make([]*cgo.Tensor, len(tensors))
	for i, t := range tensors {
		if t == nil {
			return errors.New("tensor cannot be nil")
		}
		cgoTensors[i] = t.tensor
	}
	return ir.request.SetInputTensorsByIndex(index, cgoTensors)
}

// SetOutputTensor pre-allocates an output tensor by name for zero-copy output.
func (ir *InferRequest) SetOutputTensor(name string, tensor *Tensor) error {
	return ir.request.SetOutputTensor(name, tensor.tensor)
}

// SetOutputTensorByIndex pre-allocates an output tensor by index for zero-copy output.
func (ir *InferRequest) SetOutputTensorByIndex(index int32, tensor *Tensor) error {
	return ir.request.SetOutputTensorByIndex(index, tensor.tensor)
}

func (ir *InferRequest) Cancel() error {
	return ir.request.Cancel()
}

func (ir *InferRequest) GetTensor(name string) (*Tensor, error) {
	tensor, err := ir.request.GetTensor(name)
	if err != nil {
		return nil, err
	}
	return &Tensor{tensor: tensor}, nil
}

func (ir *InferRequest) SetTensor(name string, tensor *Tensor) error {
	return ir.request.SetTensor(name, tensor.tensor)
}

// SetCallback sets a callback function that is called when async inference completes.
func (ir *InferRequest) SetCallback(callback func(error)) error {
	return ir.request.SetCallback(callback)
}
