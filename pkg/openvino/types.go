package openvino

import "github.com/accretional/openvino-go/internal/cgo"

type DataType = cgo.DataType

const (
	DataTypeFloat32  = cgo.DataTypeFloat32
	DataTypeInt64    = cgo.DataTypeInt64
	DataTypeInt32    = cgo.DataTypeInt32
	DataTypeUint8    = cgo.DataTypeUint8
	DataTypeFloat64  = cgo.DataTypeFloat64
	DataTypeInt8     = cgo.DataTypeInt8
	DataTypeUint16   = cgo.DataTypeUint16
	DataTypeInt16    = cgo.DataTypeInt16
	DataTypeUint32   = cgo.DataTypeUint32
	DataTypeUint64   = cgo.DataTypeUint64
	DataTypeFloat16  = cgo.DataTypeFloat16
	DataTypeBFloat16 = cgo.DataTypeBFloat16
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
