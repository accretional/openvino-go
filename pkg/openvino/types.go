package openvino

import "github.com/accretional/openvino-go/internal/cgo"

type DataType = cgo.DataType

const (
	DataTypeFloat32 = cgo.DataTypeFloat32
	DataTypeInt64   = cgo.DataTypeInt64
	DataTypeInt32   = cgo.DataTypeInt32
	DataTypeUint8   = cgo.DataTypeUint8
)

type PortInfo struct {
	Name     string
	Shape    []int32
	DataType DataType
}

type PerformanceMode string

const (
	PerformanceModeLatency    PerformanceMode = "LATENCY"
	PerformanceModeThroughput PerformanceMode = "THROUGHPUT"
)
