package openvino

import "github.com/accretional/openvino-go/internal/cgo"

// VariableState represents a variable state in a stateful model.
// VariableState is only available for models that contain ReadValue and Assign operations.
// This is typically used for RNNs, LSTMs, and other stateful models.
//
// VariableState allows you to:
// - Query and manage state between inference passes
// - Reset state to default values
// - Get and set state tensors for state propagation
//
// Note: For general hierarchical data without stateful operations,
// use multiple inference passes with manual state management in Go.
type VariableState struct {
	state *cgo.VariableState
}

// QueryState queries all variable states from the infer request.
func (ir *InferRequest) QueryState() ([]*VariableState, error) {
	cgoStates, err := ir.request.QueryState()
	if err != nil {
		return nil, err
	}

	states := make([]*VariableState, len(cgoStates))
	for i, cgoState := range cgoStates {
		states[i] = &VariableState{state: cgoState}
	}

	return states, nil
}

// ResetState resets all variable states to their default values.
func (ir *InferRequest) ResetState() error {
	return ir.request.ResetState()
}

// GetName returns the name of the variable state.
func (vs *VariableState) GetName() (string, error) {
	return vs.state.GetName()
}

// GetState returns the current state tensor.
func (vs *VariableState) GetState() (*Tensor, error) {
	tensor, err := vs.state.GetState()
	if err != nil {
		return nil, err
	}
	return &Tensor{tensor: tensor}, nil
}

// SetState sets the state tensor for the next inference.
func (vs *VariableState) SetState(tensor *Tensor) error {
	return vs.state.SetState(tensor.tensor)
}

// Reset resets the variable state to its default value.
func (vs *VariableState) Reset() error {
	return vs.state.Reset()
}

// Close releases the VariableState resources.
func (vs *VariableState) Close() {
	if vs.state != nil {
		vs.state.Destroy()
	}
}
