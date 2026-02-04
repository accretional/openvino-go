package bench_test

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	openvino "github.com/accretional/openvino-go/pkg/openvino"
)

// setOpenVINOInputs sets all model inputs on the infer request with dummy data.
func setOpenVINOInputs(b *testing.B, req *openvino.InferRequest) {
	b.Helper()
	for _, inp := range benchInputs {
		var data interface{}
		switch inp.DataType {
		case openvino.DataTypeInt64:
			data = makeDummyInt64(inp.Size)
		default:
			data = makeDummyFloat32(inp.Size)
		}
		if err := req.SetInputTensor(inp.Name, data, inp.Shape, inp.DataType); err != nil {
			b.Fatalf("SetInputTensor(%s) failed: %v", inp.Name, err)
		}
	}
}

// BenchmarkOpenVINO_Load measures model read + compile time.
// Uses shared Core as creating/destroying Core repeatedly causes stability issues.
func BenchmarkOpenVINO_Load(b *testing.B) {
	if sharedCore == nil {
		b.Fatal("sharedCore not initialized")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model, err := sharedCore.ReadModel(benchModelPath)
		if err != nil {
			b.Fatalf("ReadModel: %v", err)
		}
		compiled, err := sharedCore.CompileModel(model, "CPU")
		if err != nil {
			model.Close()
			b.Fatalf("CompileModel: %v", err)
		}
		compiled.Close()
		model.Close()
	}
}

// BenchmarkOpenVINO_Infer measures single inference with model pre-loaded.
func BenchmarkOpenVINO_Infer(b *testing.B) {
	if sharedCore == nil {
		b.Fatal("sharedCore not initialized")
	}

	model, err := sharedCore.ReadModel(benchModelPath)
	if err != nil {
		b.Fatalf("ReadModel: %v", err)
	}
	defer model.Close()

	compiled, err := sharedCore.CompileModel(model, "CPU")
	if err != nil {
		b.Fatalf("CompileModel: %v", err)
	}
	defer compiled.Close()

	req, err := compiled.CreateInferRequest()
	if err != nil {
		b.Fatalf("CreateInferRequest: %v", err)
	}
	defer req.Close()

	setOpenVINOInputs(b, req)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := req.Infer(); err != nil {
			b.Fatalf("Infer: %v", err)
		}
	}
}

// BenchmarkOpenVINO_InferParallel measures parallel inference from the same compiled model.
func BenchmarkOpenVINO_InferParallel(b *testing.B) {
	if sharedCore == nil {
		b.Fatal("sharedCore not initialized")
	}

	model, err := sharedCore.ReadModel(benchModelPath)
	if err != nil {
		b.Fatalf("ReadModel: %v", err)
	}
	defer model.Close()

	compiled, err := sharedCore.CompileModel(model, "CPU")
	if err != nil {
		b.Fatalf("CompileModel: %v", err)
	}
	defer compiled.Close()

	b.ReportAllocs()
	b.ResetTimer()

	var firstErr atomic.Value
	b.RunParallel(func(pb *testing.PB) {
		req, err := compiled.CreateInferRequest()
		if err != nil {
			firstErr.CompareAndSwap(nil, err)
			return
		}
		defer req.Close()

		for _, inp := range benchInputs {
			var data interface{}
			switch inp.DataType {
			case openvino.DataTypeInt64:
				data = makeDummyInt64(inp.Size)
			default:
				data = makeDummyFloat32(inp.Size)
			}
			if err := req.SetInputTensor(inp.Name, data, inp.Shape, inp.DataType); err != nil {
				firstErr.CompareAndSwap(nil, err)
				return
			}
		}

		for pb.Next() {
			if err := req.Infer(); err != nil {
				firstErr.CompareAndSwap(nil, err)
				return
			}
		}
	})
	if v := firstErr.Load(); v != nil {
		b.Fatalf("parallel infer error: %v", v)
	}
}

// setOpenVINOInputsWithBatch sets inputs with a specific batch size.
func setOpenVINOInputsWithBatch(b *testing.B, req *openvino.InferRequest, batchSize int64) {
	b.Helper()
	for _, inp := range benchInputs {
		// Modify shape to use the specified batch size
		shape := make([]int64, len(inp.Shape))
		copy(shape, inp.Shape)
		if len(shape) > 0 {
			shape[0] = batchSize
		}
		size := batchSize
		for i := 1; i < len(shape); i++ {
			size *= shape[i]
		}

		var data interface{}
		switch inp.DataType {
		case openvino.DataTypeInt64:
			data = makeDummyInt64(size)
		default:
			data = makeDummyFloat32(size)
		}
		if err := req.SetInputTensor(inp.Name, data, shape, inp.DataType); err != nil {
			b.Fatalf("SetInputTensor(%s) failed: %v", inp.Name, err)
		}
	}
}

// setOpenVINOInputsWithSeqLen sets inputs with a specific sequence length (second dimension).
func setOpenVINOInputsWithSeqLen(b *testing.B, req *openvino.InferRequest, seqLen int64) {
	b.Helper()
	for _, inp := range benchInputs {
		// Modify shape to use the specified sequence length
		shape := make([]int64, len(inp.Shape))
		copy(shape, inp.Shape)
		if len(shape) > 1 {
			shape[1] = seqLen
		}
		size := int64(1)
		for _, d := range shape {
			size *= d
		}

		var data interface{}
		switch inp.DataType {
		case openvino.DataTypeInt64:
			data = makeDummyInt64(size)
		default:
			data = makeDummyFloat32(size)
		}
		if err := req.SetInputTensor(inp.Name, data, shape, inp.DataType); err != nil {
			b.Fatalf("SetInputTensor(%s) failed: %v", inp.Name, err)
		}
	}
}

// BenchmarkOpenVINO_BatchSize tests inference with different batch sizes.
// Requires model with dynamic batch dimension; skips if model has fixed batch.
func BenchmarkOpenVINO_BatchSize(b *testing.B) {
	if sharedCore == nil {
		b.Fatal("sharedCore not initialized")
	}

	// Check if model supports dynamic batch (first dim is -1 or configurable)
	if len(benchInputs) > 0 && len(benchInputs[0].Shape) > 0 {
		// If original shape has batch=1, model likely doesn't support dynamic batch
		// We'll try batch=2 first to check
		model, err := sharedCore.ReadModel(benchModelPath)
		if err != nil {
			b.Fatalf("ReadModel: %v", err)
		}
		compiled, err := sharedCore.CompileModel(model, "CPU")
		if err != nil {
			model.Close()
			b.Fatalf("CompileModel: %v", err)
		}
		req, err := compiled.CreateInferRequest()
		if err != nil {
			compiled.Close()
			model.Close()
			b.Fatalf("CreateInferRequest: %v", err)
		}
		// Try setting batch=2
		testShape := make([]int64, len(benchInputs[0].Shape))
		copy(testShape, benchInputs[0].Shape)
		if len(testShape) > 0 {
			testShape[0] = 2
		}
		size := int64(1)
		for _, d := range testShape {
			size *= d
		}
		testData := makeDummyFloat32(size)
		err = req.SetInputTensor(benchInputs[0].Name, testData, testShape, benchInputs[0].DataType)
		req.Close()
		compiled.Close()
		model.Close()
		if err != nil {
			b.Skip("Model does not support dynamic batch size")
		}
	}

	batchSizes := []int64{1, 2, 4, 8, 16}

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("batch_%d", batchSize), func(b *testing.B) {
			model, err := sharedCore.ReadModel(benchModelPath)
			if err != nil {
				b.Fatalf("ReadModel: %v", err)
			}
			defer model.Close()

			compiled, err := sharedCore.CompileModel(model, "CPU")
			if err != nil {
				b.Fatalf("CompileModel: %v", err)
			}
			defer compiled.Close()

			req, err := compiled.CreateInferRequest()
			if err != nil {
				b.Fatalf("CreateInferRequest: %v", err)
			}
			defer req.Close()

			setOpenVINOInputsWithBatch(b, req, batchSize)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := req.Infer(); err != nil {
					b.Fatalf("Infer: %v", err)
				}
			}
		})
	}
}

// BenchmarkOpenVINO_SeqLen tests inference with different sequence lengths.
// Requires model with dynamic sequence dimension; skips if model has fixed dimensions.
func BenchmarkOpenVINO_SeqLen(b *testing.B) {
	if sharedCore == nil {
		b.Fatal("sharedCore not initialized")
	}

	// Check if model has a sequence dimension (at least 2D input)
	if len(benchInputs) == 0 || len(benchInputs[0].Shape) < 2 {
		b.Skip("Model does not have sequence dimension")
	}

	// Check if model supports dynamic sequence length
	model, err := sharedCore.ReadModel(benchModelPath)
	if err != nil {
		b.Fatalf("ReadModel: %v", err)
	}
	compiled, err := sharedCore.CompileModel(model, "CPU")
	if err != nil {
		model.Close()
		b.Fatalf("CompileModel: %v", err)
	}
	req, err := compiled.CreateInferRequest()
	if err != nil {
		compiled.Close()
		model.Close()
		b.Fatalf("CreateInferRequest: %v", err)
	}
	// Try setting a different sequence length
	testShape := make([]int64, len(benchInputs[0].Shape))
	copy(testShape, benchInputs[0].Shape)
	testShape[1] = 64 // Try a different seq length
	size := int64(1)
	for _, d := range testShape {
		size *= d
	}
	testData := makeDummyFloat32(size)
	err = req.SetInputTensor(benchInputs[0].Name, testData, testShape, benchInputs[0].DataType)
	req.Close()
	compiled.Close()
	model.Close()
	if err != nil {
		b.Skip("Model does not support dynamic sequence length")
	}

	seqLengths := []int64{32, 64, 128, 256, 512}

	for _, seqLen := range seqLengths {
		b.Run(fmt.Sprintf("seq_%d", seqLen), func(b *testing.B) {
			model, err := sharedCore.ReadModel(benchModelPath)
			if err != nil {
				b.Fatalf("ReadModel: %v", err)
			}
			defer model.Close()

			compiled, err := sharedCore.CompileModel(model, "CPU")
			if err != nil {
				b.Fatalf("CompileModel: %v", err)
			}
			defer compiled.Close()

			req, err := compiled.CreateInferRequest()
			if err != nil {
				b.Fatalf("CreateInferRequest: %v", err)
			}
			defer req.Close()

			setOpenVINOInputsWithSeqLen(b, req, seqLen)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := req.Infer(); err != nil {
					b.Fatalf("Infer: %v", err)
				}
			}
		})
	}
}

// BenchmarkOpenVINO_FirstInference measures cold-start latency (first inference after model load).
func BenchmarkOpenVINO_FirstInference(b *testing.B) {
	if sharedCore == nil {
		b.Fatal("sharedCore not initialized")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		model, err := sharedCore.ReadModel(benchModelPath)
		if err != nil {
			b.Fatalf("ReadModel: %v", err)
		}

		compiled, err := sharedCore.CompileModel(model, "CPU")
		if err != nil {
			model.Close()
			b.Fatalf("CompileModel: %v", err)
		}

		req, err := compiled.CreateInferRequest()
		if err != nil {
			compiled.Close()
			model.Close()
			b.Fatalf("CreateInferRequest: %v", err)
		}

		setOpenVINOInputs(b, req)
		b.StartTimer()

		// Measure only the first inference
		if err := req.Infer(); err != nil {
			b.Fatalf("Infer: %v", err)
		}

		b.StopTimer()
		req.Close()
		compiled.Close()
		model.Close()
		b.StartTimer()
	}
}

// BenchmarkOpenVINO_Throughput measures maximum inferences per second.
func BenchmarkOpenVINO_Throughput(b *testing.B) {
	if sharedCore == nil {
		b.Fatal("sharedCore not initialized")
	}

	model, err := sharedCore.ReadModel(benchModelPath)
	if err != nil {
		b.Fatalf("ReadModel: %v", err)
	}
	defer model.Close()

	compiled, err := sharedCore.CompileModel(model, "CPU")
	if err != nil {
		b.Fatalf("CompileModel: %v", err)
	}
	defer compiled.Close()

	// Use multiple workers for throughput
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
			req, err := compiled.CreateInferRequest()
			if err != nil {
				firstErr.CompareAndSwap(nil, err)
				return
			}
			defer req.Close()

			for _, inp := range benchInputs {
				var data interface{}
				switch inp.DataType {
				case openvino.DataTypeInt64:
					data = makeDummyInt64(inp.Size)
				default:
					data = makeDummyFloat32(inp.Size)
				}
				if err := req.SetInputTensor(inp.Name, data, inp.Shape, inp.DataType); err != nil {
					firstErr.CompareAndSwap(nil, err)
					return
				}
			}

			for time.Since(start) < duration {
				if err := req.Infer(); err != nil {
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

// BenchmarkOpenVINO_Memory reports memory usage during inference.
func BenchmarkOpenVINO_Memory(b *testing.B) {
	if sharedCore == nil {
		b.Fatal("sharedCore not initialized")
	}

	model, err := sharedCore.ReadModel(benchModelPath)
	if err != nil {
		b.Fatalf("ReadModel: %v", err)
	}
	defer model.Close()

	compiled, err := sharedCore.CompileModel(model, "CPU")
	if err != nil {
		b.Fatalf("CompileModel: %v", err)
	}
	defer compiled.Close()

	req, err := compiled.CreateInferRequest()
	if err != nil {
		b.Fatalf("CreateInferRequest: %v", err)
	}
	defer req.Close()

	setOpenVINOInputs(b, req)

	// Force GC and get baseline
	runtime.GC()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := req.Infer(); err != nil {
			b.Fatalf("Infer: %v", err)
		}
	}
	b.StopTimer()

	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)

	b.ReportMetric(float64(memAfter.Alloc-memBefore.Alloc)/1024/1024, "MB_delta")
	b.ReportMetric(float64(memAfter.TotalAlloc-memBefore.TotalAlloc)/1024/1024, "MB_total_alloc")
}

// BenchmarkOpenVINO_Threads tests performance with different thread counts.
func BenchmarkOpenVINO_Threads(b *testing.B) {
	if sharedCore == nil {
		b.Fatal("sharedCore not initialized")
	}

	threadCounts := []int{1, 2, 4, 8}

	for _, numThreads := range threadCounts {
		b.Run(fmt.Sprintf("threads_%d", numThreads), func(b *testing.B) {
			model, err := sharedCore.ReadModel(benchModelPath)
			if err != nil {
				b.Fatalf("ReadModel: %v", err)
			}
			defer model.Close()

			// Compile with specific thread count
			compiled, err := sharedCore.CompileModel(model, "CPU",
				openvino.InferenceNumThreads(numThreads))
			if err != nil {
				b.Fatalf("CompileModel: %v", err)
			}
			defer compiled.Close()

			req, err := compiled.CreateInferRequest()
			if err != nil {
				b.Fatalf("CreateInferRequest: %v", err)
			}
			defer req.Close()

			setOpenVINOInputs(b, req)

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := req.Infer(); err != nil {
					b.Fatalf("Infer: %v", err)
				}
			}
		})
	}
}

// BenchmarkOpenVINO_ConcurrentSessions tests multiple independent sessions.
func BenchmarkOpenVINO_ConcurrentSessions(b *testing.B) {
	if sharedCore == nil {
		b.Fatal("sharedCore not initialized")
	}

	sessionCounts := []int{1, 2, 4, 8}

	for _, numSessions := range sessionCounts {
		b.Run(fmt.Sprintf("sessions_%d", numSessions), func(b *testing.B) {
			// Create multiple compiled models and requests
			type session struct {
				model    *openvino.Model
				compiled *openvino.CompiledModel
				req      *openvino.InferRequest
			}
			sessions := make([]session, numSessions)

			for i := 0; i < numSessions; i++ {
				model, err := sharedCore.ReadModel(benchModelPath)
				if err != nil {
					b.Fatalf("ReadModel: %v", err)
				}
				compiled, err := sharedCore.CompileModel(model, "CPU")
				if err != nil {
					model.Close()
					b.Fatalf("CompileModel: %v", err)
				}
				req, err := compiled.CreateInferRequest()
				if err != nil {
					compiled.Close()
					model.Close()
					b.Fatalf("CreateInferRequest: %v", err)
				}
				setOpenVINOInputs(b, req)
				sessions[i] = session{model: model, compiled: compiled, req: req}
			}

			defer func() {
				for _, s := range sessions {
					s.req.Close()
					s.compiled.Close()
					s.model.Close()
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
				go func(req *openvino.InferRequest) {
					defer wg.Done()
					for i := 0; i < inferCount; i++ {
						if err := req.Infer(); err != nil {
							firstErr.CompareAndSwap(nil, err)
							return
						}
					}
				}(s.req)
			}

			wg.Wait()
			if v := firstErr.Load(); v != nil {
				b.Fatalf("concurrent sessions error: %v", v)
			}
		})
	}
}
