package cgo

type DataType int32

const (
	DataTypeFloat32  DataType = 0
	DataTypeInt64    DataType = 1
	DataTypeInt32    DataType = 2
	DataTypeUint8    DataType = 3
	DataTypeFloat64  DataType = 4
	DataTypeInt8     DataType = 5
	DataTypeUint16   DataType = 6
	DataTypeInt16    DataType = 7
	DataTypeUint32   DataType = 8
	DataTypeUint64   DataType = 9
	DataTypeFloat16  DataType = 10
	DataTypeBFloat16 DataType = 11
)

type PortInfo struct {
	Name     string
	Shape    []int32
	DataType DataType
}
