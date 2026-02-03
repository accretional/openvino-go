package openvino

import "testing"

func TestPerformanceHint(t *testing.T) {
	props := make(map[string]string)
	PerformanceHint(PerformanceModeThroughput)(props)
	if props["PERFORMANCE_HINT"] != "THROUGHPUT" {
		t.Errorf("expected THROUGHPUT, got %s", props["PERFORMANCE_HINT"])
	}
}

func TestNumStreams(t *testing.T) {
	props := make(map[string]string)
	NumStreams(4)(props)
	if props["NUM_STREAMS"] != "4" {
		t.Errorf("expected 4, got %s", props["NUM_STREAMS"])
	}
}

func TestInferenceNumThreads(t *testing.T) {
	props := make(map[string]string)
	InferenceNumThreads(8)(props)
	if props["INFERENCE_NUM_THREADS"] != "8" {
		t.Errorf("expected 8, got %s", props["INFERENCE_NUM_THREADS"])
	}
}
