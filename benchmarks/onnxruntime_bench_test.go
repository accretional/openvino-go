package bench_test

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	openvino "github.com/accretional/openvino-go/pkg/openvino"
	ort "github.com/yalue/onnxruntime_go"
)

const defaultORTLibPath = "/usr/local/lib/onnxruntime/libonnxruntime.so"

func getORTLibPath() string {
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		return p
	}
	return defaultORTLibPath
}

// initORT initializes the ONNX Runtime environment if not already initialized.
func initORT(b *testing.B) {
	b.Helper()
	if ort.IsInitialized() {
		return
	}
	ort.SetSharedLibraryPath(getORTLibPath())
	if err := ort.InitializeEnvironment(); err != nil {
		b.Fatalf("ort.InitializeEnvironment: %v", err)
	}
}

// destroyORT tears down the ONNX Runtime environment.
func destroyORT(b *testing.B) {
	b.Helper()
	if err := ort.DestroyEnvironment(); err != nil {
		b.Fatalf("ort.DestroyEnvironment: %v", err)
	}
}

// ortInputInfo stores ONNX Runtime tensor info derived from the shared model metadata.
type ortInputInfo struct {
	Name  string
	Shape ort.Shape
	IsInt bool // true for int64, false for float32
}

func getORTInputInfos() []ortInputInfo {
	infos := make([]ortInputInfo, len(benchInputs))
	for i, inp := range benchInputs {
		// Create a copy of shape to avoid any modification issues
		shapeCopy := make([]int64, len(inp.Shape))
		copy(shapeCopy, inp.Shape)
		infos[i] = ortInputInfo{
			Name:  inp.Name,
			Shape: ort.NewShape(shapeCopy...),
			IsInt: inp.DataType == openvino.DataTypeInt64 || inp.DataType == openvino.DataTypeInt32,
		}
	}
	return infos
}

func getORTOutputShapes() []ort.Shape {
	shapes := make([]ort.Shape, len(benchOutputs))
	for i, out := range benchOutputs {
		// Create a copy of shape to avoid any modification issues
		shapeCopy := make([]int64, len(out.Shape))
		copy(shapeCopy, out.Shape)
		shapes[i] = ort.NewShape(shapeCopy...)
	}
	return shapes
}

func getORTInputNames() []string {
	names := make([]string, len(benchInputs))
	for i, inp := range benchInputs {
		names[i] = inp.Name
	}
	return names
}

func getORTOutputNames() []string {
	names := make([]string, len(benchOutputs))
	for i, out := range benchOutputs {
		names[i] = out.Name
	}
	return names
}

// ortTensors holds pre-allocated input and output tensors for ONNX Runtime benchmarks.
type ortTensors struct {
	inputs  []ort.ArbitraryTensor
	outputs []ort.ArbitraryTensor
	// Keep typed references for data access
	outputTensors []*ort.Tensor[float32]
}

func createORTTensors(b *testing.B) *ortTensors {
	b.Helper()
	infos := getORTInputInfos()
	outShapes := getORTOutputShapes()

	inputs := make([]ort.ArbitraryTensor, len(infos))
	for i, info := range infos {
		var err error
		if info.IsInt {
			data := makeDummyInt64(info.Shape.FlattenedSize())
			inputs[i], err = ort.NewTensor(info.Shape, data)
		} else {
			data := makeDummyFloat32(info.Shape.FlattenedSize())
			inputs[i], err = ort.NewTensor(info.Shape, data)
		}
		if err != nil {
			b.Fatalf("ort.NewTensor(%s): %v", info.Name, err)
		}
	}

	outputs := make([]ort.ArbitraryTensor, len(outShapes))
	outputTensors := make([]*ort.Tensor[float32], len(outShapes))
	for i, shape := range outShapes {
		out, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			b.Fatalf("ort.NewEmptyTensor(output %d): %v", i, err)
		}
		outputs[i] = out
		outputTensors[i] = out
	}

	return &ortTensors{inputs: inputs, outputs: outputs, outputTensors: outputTensors}
}

func (t *ortTensors) Destroy() {
	for _, v := range t.inputs {
		v.Destroy()
	}
	for _, v := range t.outputs {
		v.Destroy()
	}
}

// BenchmarkONNXRuntime_Load measures session creation time.
func BenchmarkONNXRuntime_Load(b *testing.B) {
	initORT(b)
	defer destroyORT(b)

	tensors := createORTTensors(b)
	defer tensors.Destroy()

	inputNames := getORTInputNames()
	outputNames := getORTOutputNames()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		session, err := ort.NewAdvancedSession(benchModelPath,
			inputNames, outputNames,
			tensors.inputs, tensors.outputs, nil)
		if err != nil {
			b.Fatalf("NewAdvancedSession: %v", err)
		}
		session.Destroy()
	}
}

// BenchmarkONNXRuntime_Infer measures single inference with session pre-created.
func BenchmarkONNXRuntime_Infer(b *testing.B) {
	initORT(b)
	defer destroyORT(b)

	tensors := createORTTensors(b)
	defer tensors.Destroy()

	session, err := ort.NewAdvancedSession(benchModelPath,
		getORTInputNames(), getORTOutputNames(),
		tensors.inputs, tensors.outputs, nil)
	if err != nil {
		b.Fatalf("NewAdvancedSession: %v", err)
	}
	defer session.Destroy()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := session.Run(); err != nil {
			b.Fatalf("Run: %v", err)
		}
	}
}

// BenchmarkONNXRuntime_InferParallel measures parallel inference with separate sessions.
func BenchmarkONNXRuntime_InferParallel(b *testing.B) {
	initORT(b)
	defer destroyORT(b)

	inputNames := getORTInputNames()
	outputNames := getORTOutputNames()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		tensors := createORTTensors(b)
		defer tensors.Destroy()

		session, err := ort.NewAdvancedSession(benchModelPath,
			inputNames, outputNames,
			tensors.inputs, tensors.outputs, nil)
		if err != nil {
			b.Fatalf("NewAdvancedSession: %v", err)
		}
		defer session.Destroy()

		for pb.Next() {
			if err := session.Run(); err != nil {
				b.Fatalf("Run: %v", err)
			}
		}
	})
}

// createORTTensorsWithBatch creates tensors with a specific batch size.
func createORTTensorsWithBatch(b *testing.B, batchSize int64) *ortTensors {
	b.Helper()
	infos := getORTInputInfos()

	inputs := make([]ort.ArbitraryTensor, len(infos))
	for i, info := range infos {
		// Create a proper copy of the shape to avoid modifying the original
		origDims := []int64(info.Shape)
		dims := make([]int64, len(origDims))
		copy(dims, origDims)
		if len(dims) > 0 {
			dims[0] = batchSize
		}
		shape := ort.NewShape(dims...)
		size := shape.FlattenedSize()

		var err error
		if info.IsInt {
			data := makeDummyInt64(size)
			inputs[i], err = ort.NewTensor(shape, data)
		} else {
			data := makeDummyFloat32(size)
			inputs[i], err = ort.NewTensor(shape, data)
		}
		if err != nil {
			b.Fatalf("ort.NewTensor(%s): %v", info.Name, err)
		}
	}

	// Output shapes also need batch adjustment
	outShapes := getORTOutputShapes()
	outputs := make([]ort.ArbitraryTensor, len(outShapes))
	outputTensors := make([]*ort.Tensor[float32], len(outShapes))
	for i, shape := range outShapes {
		origDims := []int64(shape)
		dims := make([]int64, len(origDims))
		copy(dims, origDims)
		if len(dims) > 0 {
			dims[0] = batchSize
		}
		newShape := ort.NewShape(dims...)
		out, err := ort.NewEmptyTensor[float32](newShape)
		if err != nil {
			b.Fatalf("ort.NewEmptyTensor(output %d): %v", i, err)
		}
		outputs[i] = out
		outputTensors[i] = out
	}

	return &ortTensors{inputs: inputs, outputs: outputs, outputTensors: outputTensors}
}

// createORTTensorsWithSeqLen creates tensors with a specific sequence length.
func createORTTensorsWithSeqLen(b *testing.B, seqLen int64) *ortTensors {
	b.Helper()
	infos := getORTInputInfos()

	inputs := make([]ort.ArbitraryTensor, len(infos))
	for i, info := range infos {
		// Create a proper copy of the shape to avoid modifying the original
		origDims := []int64(info.Shape)
		dims := make([]int64, len(origDims))
		copy(dims, origDims)
		if len(dims) > 1 {
			dims[1] = seqLen
		}
		shape := ort.NewShape(dims...)
		size := shape.FlattenedSize()

		var err error
		if info.IsInt {
			data := makeDummyInt64(size)
			inputs[i], err = ort.NewTensor(shape, data)
		} else {
			data := makeDummyFloat32(size)
			inputs[i], err = ort.NewTensor(shape, data)
		}
		if err != nil {
			b.Fatalf("ort.NewTensor(%s): %v", info.Name, err)
		}
	}

	// Output shapes also need seq length adjustment
	outShapes := getORTOutputShapes()
	outputs := make([]ort.ArbitraryTensor, len(outShapes))
	outputTensors := make([]*ort.Tensor[float32], len(outShapes))
	for i, shape := range outShapes {
		origDims := []int64(shape)
		dims := make([]int64, len(origDims))
		copy(dims, origDims)
		if len(dims) > 1 {
			dims[1] = seqLen
		}
		newShape := ort.NewShape(dims...)
		out, err := ort.NewEmptyTensor[float32](newShape)
		if err != nil {
			b.Fatalf("ort.NewEmptyTensor(output %d): %v", i, err)
		}
		outputs[i] = out
		outputTensors[i] = out
	}

	return &ortTensors{inputs: inputs, outputs: outputs, outputTensors: outputTensors}
}

// BenchmarkONNXRuntime_BatchSize tests inference with different batch sizes.
// Requires model with dynamic batch dimension; skips if model has fixed batch.
func BenchmarkONNXRuntime_BatchSize(b *testing.B) {
	initORT(b)
	defer destroyORT(b)

	// Test if model supports dynamic batch by trying batch=2
	tensors := createORTTensorsWithBatch(b, 2)
	session, err := ort.NewAdvancedSession(benchModelPath,
		getORTInputNames(), getORTOutputNames(),
		tensors.inputs, tensors.outputs, nil)
	if err != nil {
		tensors.Destroy()
		b.Skip("Model does not support dynamic batch size")
	}
	err = session.Run()
	session.Destroy()
	tensors.Destroy()
	if err != nil {
		b.Skip("Model does not support dynamic batch size")
	}

	batchSizes := []int64{1, 2, 4, 8, 16}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("batch_%d", batchSize), func(b *testing.B) {
			tensors := createORTTensorsWithBatch(b, batchSize)
			defer tensors.Destroy()

			session, err := ort.NewAdvancedSession(benchModelPath,
				getORTInputNames(), getORTOutputNames(),
				tensors.inputs, tensors.outputs, nil)
			if err != nil {
				b.Fatalf("NewAdvancedSession: %v", err)
			}
			defer session.Destroy()

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := session.Run(); err != nil {
					b.Fatalf("Run: %v", err)
				}
			}
		})
	}
}

// BenchmarkONNXRuntime_SeqLen tests inference with different sequence lengths.
// Requires model with dynamic sequence dimension; skips if model has fixed dimensions.
func BenchmarkONNXRuntime_SeqLen(b *testing.B) {
	initORT(b)
	defer destroyORT(b)

	// Check if model has a sequence dimension
	if len(benchInputs) == 0 || len(benchInputs[0].Shape) < 2 {
		b.Skip("Model does not have sequence dimension")
	}

	// Test if model supports dynamic sequence length
	tensors := createORTTensorsWithSeqLen(b, 64)
	session, err := ort.NewAdvancedSession(benchModelPath,
		getORTInputNames(), getORTOutputNames(),
		tensors.inputs, tensors.outputs, nil)
	if err != nil {
		tensors.Destroy()
		b.Skip("Model does not support dynamic sequence length")
	}
	err = session.Run()
	session.Destroy()
	tensors.Destroy()
	if err != nil {
		b.Skip("Model does not support dynamic sequence length")
	}

	seqLengths := []int64{32, 64, 128, 256, 512}

	for _, seqLen := range seqLengths {
		b.Run(fmt.Sprintf("seq_%d", seqLen), func(b *testing.B) {
			tensors := createORTTensorsWithSeqLen(b, seqLen)
			defer tensors.Destroy()

			session, err := ort.NewAdvancedSession(benchModelPath,
				getORTInputNames(), getORTOutputNames(),
				tensors.inputs, tensors.outputs, nil)
			if err != nil {
				b.Fatalf("NewAdvancedSession: %v", err)
			}
			defer session.Destroy()

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := session.Run(); err != nil {
					b.Fatalf("Run: %v", err)
				}
			}
		})
	}
}

// BenchmarkONNXRuntime_FirstInference measures cold-start latency.
func BenchmarkONNXRuntime_FirstInference(b *testing.B) {
	initORT(b)
	defer destroyORT(b)

	inputNames := getORTInputNames()
	outputNames := getORTOutputNames()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		tensors := createORTTensors(b)

		session, err := ort.NewAdvancedSession(benchModelPath,
			inputNames, outputNames,
			tensors.inputs, tensors.outputs, nil)
		if err != nil {
			tensors.Destroy()
			b.Fatalf("NewAdvancedSession: %v", err)
		}
		b.StartTimer()

		// Measure only the first inference
		if err := session.Run(); err != nil {
			b.Fatalf("Run: %v", err)
		}

		b.StopTimer()
		session.Destroy()
		tensors.Destroy()
		b.StartTimer()
	}
}

// BenchmarkONNXRuntime_Throughput measures maximum inferences per second.
func BenchmarkONNXRuntime_Throughput(b *testing.B) {
	initORT(b)
	defer destroyORT(b)

	inputNames := getORTInputNames()
	outputNames := getORTOutputNames()

	numWorkers := runtime.NumCPU()
	var wg sync.WaitGroup
	var totalInferences atomic.Int64
	var firstErr atomic.Value

	b.ReportAllocs()
	b.ResetTimer()

	duration := time.Second * 5 // Run for 5 seconds
	start := time.Now()

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tensors := createORTTensors(b)
			defer tensors.Destroy()

			session, err := ort.NewAdvancedSession(benchModelPath,
				inputNames, outputNames,
				tensors.inputs, tensors.outputs, nil)
			if err != nil {
				firstErr.CompareAndSwap(nil, err)
				return
			}
			defer session.Destroy()

			for time.Since(start) < duration {
				if err := session.Run(); err != nil {
					firstErr.CompareAndSwap(nil, err)
					return
				}
				totalInferences.Add(1)
			}
		}()
	}

	wg.Wait()
	elapsed := time.Since(start)

	if v := firstErr.Load(); v != nil {
		b.Fatalf("throughput error: %v", v)
	}

	throughput := float64(totalInferences.Load()) / elapsed.Seconds()
	b.ReportMetric(throughput, "inferences/sec")
}

// BenchmarkONNXRuntime_Memory reports memory usage during inference.
func BenchmarkONNXRuntime_Memory(b *testing.B) {
	initORT(b)
	defer destroyORT(b)

	tensors := createORTTensors(b)
	defer tensors.Destroy()

	session, err := ort.NewAdvancedSession(benchModelPath,
		getORTInputNames(), getORTOutputNames(),
		tensors.inputs, tensors.outputs, nil)
	if err != nil {
		b.Fatalf("NewAdvancedSession: %v", err)
	}
	defer session.Destroy()

	// Force GC and get baseline
	runtime.GC()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := session.Run(); err != nil {
			b.Fatalf("Run: %v", err)
		}
	}
	b.StopTimer()

	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)

	b.ReportMetric(float64(memAfter.Alloc-memBefore.Alloc)/1024/1024, "MB_delta")
	b.ReportMetric(float64(memAfter.TotalAlloc-memBefore.TotalAlloc)/1024/1024, "MB_total_alloc")
}

// BenchmarkONNXRuntime_Threads tests performance with different thread counts.
func BenchmarkONNXRuntime_Threads(b *testing.B) {
	initORT(b)
	defer destroyORT(b)

	threadCounts := []int{1, 2, 4, 8}

	for _, numThreads := range threadCounts {
		b.Run(fmt.Sprintf("threads_%d", numThreads), func(b *testing.B) {
			tensors := createORTTensors(b)
			defer tensors.Destroy()

			// Create session options with thread count
			opts, err := ort.NewSessionOptions()
			if err != nil {
				b.Fatalf("NewSessionOptions: %v", err)
			}
			defer opts.Destroy()
			opts.SetIntraOpNumThreads(numThreads)

			session, err := ort.NewAdvancedSession(benchModelPath,
				getORTInputNames(), getORTOutputNames(),
				tensors.inputs, tensors.outputs, opts)
			if err != nil {
				b.Fatalf("NewAdvancedSession: %v", err)
			}
			defer session.Destroy()

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := session.Run(); err != nil {
					b.Fatalf("Run: %v", err)
				}
			}
		})
	}
}

// BenchmarkONNXRuntime_ConcurrentSessions tests multiple independent sessions.
func BenchmarkONNXRuntime_ConcurrentSessions(b *testing.B) {
	initORT(b)
	defer destroyORT(b)

	sessionCounts := []int{1, 2, 4, 8}
	inputNames := getORTInputNames()
	outputNames := getORTOutputNames()

	for _, numSessions := range sessionCounts {
		b.Run(fmt.Sprintf("sessions_%d", numSessions), func(b *testing.B) {
			type ortSession struct {
				tensors *ortTensors
				session *ort.AdvancedSession
			}
			sessions := make([]ortSession, numSessions)

			for i := 0; i < numSessions; i++ {
				tensors := createORTTensors(b)
				session, err := ort.NewAdvancedSession(benchModelPath,
					inputNames, outputNames,
					tensors.inputs, tensors.outputs, nil)
				if err != nil {
					tensors.Destroy()
					b.Fatalf("NewAdvancedSession: %v", err)
				}
				sessions[i] = ortSession{tensors: tensors, session: session}
			}

			defer func() {
				for _, s := range sessions {
					s.session.Destroy()
					s.tensors.Destroy()
				}
			}()

			b.ReportAllocs()
			b.ResetTimer()

			var wg sync.WaitGroup
			var firstErr atomic.Value
			inferCount := b.N / numSessions
			if inferCount == 0 {
				inferCount = 1
			}

			for _, s := range sessions {
				wg.Add(1)
				go func(session *ort.AdvancedSession) {
					defer wg.Done()
					for i := 0; i < inferCount; i++ {
						if err := session.Run(); err != nil {
							firstErr.CompareAndSwap(nil, err)
							return
						}
					}
				}(s.session)
			}

			wg.Wait()
			if v := firstErr.Load(); v != nil {
				b.Fatalf("concurrent sessions error: %v", v)
			}
		})
	}
}
