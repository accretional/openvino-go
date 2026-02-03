package openvino

import "github.com/accretional/openvino-go/internal/cgo"

type Model struct {
	model *cgo.Model
}

func (c *Core) ReadModel(modelPath string) (*Model, error) {
	model, err := c.core.ReadModel(modelPath)
	if err != nil {
		return nil, err
	}
	return &Model{model: model}, nil
}

func (m *Model) Close() {
	if m.model != nil {
		m.model.Destroy()
	}
}

func (m *Model) GetInputs() ([]PortInfo, error) {
	cgoPorts, err := m.model.GetInputs()
	if err != nil {
		return nil, err
	}

	ports := make([]PortInfo, len(cgoPorts))
	for i, p := range cgoPorts {
		ports[i] = PortInfo{
			Name:     p.Name,
			Shape:    p.Shape,
			DataType: DataType(p.DataType),
		}
	}
	return ports, nil
}

func (m *Model) GetOutputs() ([]PortInfo, error) {
	cgoPorts, err := m.model.GetOutputs()
	if err != nil {
		return nil, err
	}

	ports := make([]PortInfo, len(cgoPorts))
	for i, p := range cgoPorts {
		ports[i] = PortInfo{
			Name:     p.Name,
			Shape:    p.Shape,
			DataType: DataType(p.DataType),
		}
	}
	return ports, nil
}
