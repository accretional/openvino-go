package openvino

import "github.com/accretional/openvino-go/internal/cgo"

type Core struct {
	core *cgo.Core
}

func NewCore() (*Core, error) {
	core, err := cgo.CreateCore()
	if err != nil {
		return nil, err
	}
	return &Core{core: core}, nil
}

func (c *Core) Close() {
	if c.core != nil {
		c.core.Destroy()
	}
}

func (c *Core) GetAvailableDevices() ([]string, error) {
	return c.core.GetAvailableDevices()
}
