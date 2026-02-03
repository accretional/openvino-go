package openvino

import "fmt"

type CompileOption func(map[string]string)

func PerformanceHint(mode PerformanceMode) CompileOption {
	return func(props map[string]string) {
		props["PERFORMANCE_HINT"] = string(mode)
	}
}

func NumStreams(n int) CompileOption {
	return func(props map[string]string) {
		props["NUM_STREAMS"] = fmt.Sprintf("%d", n)
	}
}

func InferenceNumThreads(n int) CompileOption {
	return func(props map[string]string) {
		props["INFERENCE_NUM_THREADS"] = fmt.Sprintf("%d", n)
	}
}
