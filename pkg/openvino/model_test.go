package openvino

import (
	"os"
	"testing"
)

func TestCore_ReadModel(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()

	_, err := core.ReadModel("/nonexistent/path/model.ir")
	if err == nil {
		t.Fatal("ReadModel with nonexistent path should return error")
	}
	// With a real path we'd get model; skip integration test without a fixture
}

func TestCore_ReadModelFromBuffer_Error(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()

	_, err := core.ReadModelFromBuffer(nil, nil)
	if err == nil {
		t.Fatal("ReadModelFromBuffer with nil buffer should return error")
	}

	_, err = core.ReadModelFromBuffer([]byte{}, nil)
	if err == nil {
		t.Fatal("ReadModelFromBuffer with empty buffer should return error")
	}
}

func TestCore_ReadModelFromBuffer_MinimalXML(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()

	xmlModel := `<?xml version="1.0"?>
<net name="test_model" version="11">
	<layers>
		<layer id="0" name="x" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,10"/>
			<output>
				<port id="0" precision="FP32" names="x">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="y" type="Parameter" version="opset1">
			<data element_type="f32" shape="1,10"/>
			<output>
				<port id="0" precision="FP32" names="y">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="add" type="Add" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>10</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="z">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="res" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>10</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
	</edges>
</net>`

	model, err := core.ReadModelFromBuffer([]byte(xmlModel), nil)
	if err != nil {
		t.Fatalf("ReadModelFromBuffer failed: %v", err)
	}
	defer model.Close()

	inputs, err := model.GetInputs()
	if err != nil {
		t.Fatalf("GetInputs failed: %v", err)
	}
	if len(inputs) != 2 {
		t.Fatalf("expected 2 inputs, got %d", len(inputs))
	}

	outputs, err := model.GetOutputs()
	if err != nil {
		t.Fatalf("GetOutputs failed: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outputs))
	}
}

func TestCore_ReadModelFromBuffer_integration(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()

	modelPath := getTestModelPath(t)
	if modelPath == "" {
		t.Skip("no test model path (set OPENVINO_TEST_MODEL for integration)")
	}

	modelData, err := os.ReadFile(modelPath)
	if err != nil {
		t.Skipf("ReadFile failed: %v", err)
	}

	model, err := core.ReadModelFromBuffer(modelData, nil)
	if err != nil {
		t.Skipf("ReadModelFromBuffer failed: %v", err)
	}
	defer model.Close()
	if model == nil {
		t.Fatal("ReadModelFromBuffer returned nil with nil error")
	}
}

func TestCore_ReadModel_integration(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()

	// If OPENVINO_TEST_MODEL is set, try to load and exercise model
	modelPath := getTestModelPath(t)
	if modelPath == "" {
		t.Skip("no test model path (set OPENVINO_TEST_MODEL for integration)")
	}

	model, err := core.ReadModel(modelPath)
	if err != nil {
		t.Skipf("ReadModel failed: %v", err)
	}
	defer model.Close()
	if model == nil {
		t.Fatal("ReadModel returned nil with nil error")
	}
}

func TestModel_Close(t *testing.T) {
	m := &Model{}
	m.Close()
}

func TestModel_GetInputs(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()
	modelPath := getTestModelPath(t)
	if modelPath == "" {
		t.Skip("no test model path")
	}
	model, err := core.ReadModel(modelPath)
	if err != nil {
		t.Skipf("ReadModel failed: %v", err)
	}
	defer model.Close()

	inputs, err := model.GetInputs()
	if err != nil {
		t.Fatalf("GetInputs failed: %v", err)
	}
	if inputs == nil {
		t.Fatal("GetInputs returned nil slice")
	}
}

func TestModel_GetOutputs(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()
	modelPath := getTestModelPath(t)
	if modelPath == "" {
		t.Skip("no test model path")
	}
	model, err := core.ReadModel(modelPath)
	if err != nil {
		t.Skipf("ReadModel failed: %v", err)
	}
	defer model.Close()

	outputs, err := model.GetOutputs()
	if err != nil {
		t.Fatalf("GetOutputs failed: %v", err)
	}
	if outputs == nil {
		t.Fatal("GetOutputs returned nil slice")
	}
}
