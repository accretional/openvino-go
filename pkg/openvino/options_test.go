package openvino

import "testing"

func TestPerformanceHint(t *testing.T) {
	props := make(map[string]string)
	PerformanceHint(PerformanceModeThroughput)(props)
	if props["PERFORMANCE_HINT"] != "THROUGHPUT" {
		t.Errorf("expected THROUGHPUT, got %s", props["PERFORMANCE_HINT"])
	}
}

func TestPerformanceHint_Latency(t *testing.T) {
	props := make(map[string]string)
	PerformanceHint(PerformanceModeLatency)(props)
	if props["PERFORMANCE_HINT"] != "LATENCY" {
		t.Errorf("expected LATENCY, got %s", props["PERFORMANCE_HINT"])
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

func TestCompileOptions_combined(t *testing.T) {
	props := make(map[string]string)
	PerformanceHint(PerformanceModeLatency)(props)
	NumStreams(2)(props)
	InferenceNumThreads(4)(props)
	if props["PERFORMANCE_HINT"] != "LATENCY" {
		t.Errorf("PERFORMANCE_HINT = %s, want LATENCY", props["PERFORMANCE_HINT"])
	}
	if props["NUM_STREAMS"] != "2" {
		t.Errorf("NUM_STREAMS = %s, want 2", props["NUM_STREAMS"])
	}
	if props["INFERENCE_NUM_THREADS"] != "4" {
		t.Errorf("INFERENCE_NUM_THREADS = %s, want 4", props["INFERENCE_NUM_THREADS"])
	}
}

func TestPerformanceMode_constants(t *testing.T) {
	if PerformanceModeLatency != "LATENCY" {
		t.Errorf("PerformanceModeLatency = %q, want LATENCY", PerformanceModeLatency)
	}
	if PerformanceModeThroughput != "THROUGHPUT" {
		t.Errorf("PerformanceModeThroughput = %q, want THROUGHPUT", PerformanceModeThroughput)
	}
}
