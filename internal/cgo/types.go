package cgo

type DataType int32

const (
	DataTypeFloat32 DataType = 0
	DataTypeInt64   DataType = 1
	DataTypeInt32   DataType = 2
	DataTypeUint8   DataType = 3
)

type PortInfo struct {
	Name     string
	Shape    []int32
	DataType DataType
}
