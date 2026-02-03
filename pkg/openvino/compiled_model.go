package openvino

import "github.com/accretional/openvino-go/internal/cgo"

type CompiledModel struct {
	compiled *cgo.CompiledModel
}

func (c *Core) CompileModel(model *Model, device string, options ...CompileOption) (*CompiledModel, error) {
	props := make(map[string]string)
	for _, opt := range options {
		opt(props)
	}

	var compiled *cgo.CompiledModel
	var err error

	if len(props) > 0 {
		compiled, err = c.core.CompileModelWithProperties(model.model, device, props)
	} else {
		compiled, err = c.core.CompileModel(model.model, device)
	}

	if err != nil {
		return nil, err
	}
	return &CompiledModel{compiled: compiled}, nil
}

func (cm *CompiledModel) Close() {
	if cm.compiled != nil {
		cm.compiled.Destroy()
	}
}
