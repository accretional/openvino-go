package openvino

import (
	"testing"
)

func TestNewTensor(t *testing.T) {
	tensor, err := NewTensor(DataTypeFloat32, []int64{1, 3, 224, 224})
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	defer tensor.Close()

	shape, err := tensor.GetShape()
	if err != nil {
		t.Fatalf("GetShape failed: %v", err)
	}
	if len(shape) != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != 224 || shape[3] != 224 {
		t.Errorf("Expected shape [1, 3, 224, 224], got %v", shape)
	}

	dataType, err := tensor.GetElementType()
	if err != nil {
		t.Fatalf("GetElementType failed: %v", err)
	}
	if dataType != DataTypeFloat32 {
		t.Errorf("Expected DataTypeFloat32, got %v", dataType)
	}

	size, err := tensor.GetSize()
	if err != nil {
		t.Fatalf("GetSize failed: %v", err)
	}
	expectedSize := int64(1 * 3 * 224 * 224)
	if size != expectedSize {
		t.Errorf("Expected size %d, got %d", expectedSize, size)
	}

	byteSize, err := tensor.GetByteSize()
	if err != nil {
		t.Fatalf("GetByteSize failed: %v", err)
	}
	expectedByteSize := expectedSize * 4 // float32 = 4 bytes
	if byteSize != expectedByteSize {
		t.Errorf("Expected byte size %d, got %d", expectedByteSize, byteSize)
	}
}

func TestNewTensorWithData(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0, 4.0}
	tensor, err := NewTensorWithData(DataTypeFloat32, []int64{1, 4}, data)
	if err != nil {
		t.Fatalf("NewTensorWithData failed: %v", err)
	}
	defer tensor.Close()

	result, err := tensor.GetDataAsFloat32()
	if err != nil {
		t.Fatalf("GetDataAsFloat32 failed: %v", err)
	}
	if len(result) != len(data) {
		t.Errorf("Expected %d elements, got %d", len(data), len(result))
	}
	for i := range data {
		if result[i] != data[i] {
			t.Errorf("Expected data[%d] = %f, got %f", i, data[i], result[i])
		}
	}
}

func TestNewTensorWithInt64Data(t *testing.T) {
	data := []int64{10, 20, 30, 40}
	tensor, err := NewTensorWithData(DataTypeInt64, []int64{1, 4}, data)
	if err != nil {
		t.Fatalf("NewTensorWithData failed: %v", err)
	}
	defer tensor.Close()

	result, err := tensor.GetDataAsInt64()
	if err != nil {
		t.Fatalf("GetDataAsInt64 failed: %v", err)
	}
	if len(result) != len(data) {
		t.Errorf("Expected %d elements, got %d", len(data), len(result))
	}
	for i := range data {
		if result[i] != data[i] {
			t.Errorf("Expected data[%d] = %d, got %d", i, data[i], result[i])
		}
	}
}

func TestTensor_SetShape(t *testing.T) {
	tensor, err := NewTensor(DataTypeFloat32, []int64{1, 4})
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	defer tensor.Close()

	err = tensor.SetShape([]int64{2, 8})
	if err != nil {
		t.Fatalf("SetShape failed: %v", err)
	}

	shape, err := tensor.GetShape()
	if err != nil {
		t.Fatalf("GetShape failed: %v", err)
	}
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 8 {
		t.Errorf("Expected shape [2, 8], got %v", shape)
	}
}

func TestNewTensorWithMultipleDataTypes(t *testing.T) {
	tests := []struct {
		name     string
		dataType DataType
		data     interface{}
		shape    []int64
	}{
		{"Float64", DataTypeFloat64, []float64{1.5, 2.5}, []int64{1, 2}},
		{"Int32", DataTypeInt32, []int32{1, 2, 3}, []int64{1, 3}},
		{"Uint8", DataTypeUint8, []uint8{10, 20, 30}, []int64{1, 3}},
		{"Int8", DataTypeInt8, []int8{-10, 20, -30}, []int64{1, 3}},
		{"Uint16", DataTypeUint16, []uint16{100, 200}, []int64{1, 2}},
		{"Int16", DataTypeInt16, []int16{-100, 200}, []int64{1, 2}},
		{"Uint32", DataTypeUint32, []uint32{1000, 2000}, []int64{1, 2}},
		{"Uint64", DataTypeUint64, []uint64{10000, 20000}, []int64{1, 2}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, err := NewTensorWithData(tt.dataType, tt.shape, tt.data)
			if err != nil {
				t.Fatalf("NewTensorWithData failed: %v", err)
			}
			defer tensor.Close()

			dataType, err := tensor.GetElementType()
			if err != nil {
				t.Fatalf("GetElementType failed: %v", err)
			}
			if dataType != tt.dataType {
				t.Errorf("Expected %v, got %v", tt.dataType, dataType)
			}
		})
	}
}
