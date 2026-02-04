// Package openvino provides idiomatic Go bindings for Intel OpenVINO Runtime.
//
// OpenVINO (Open Visual Inference and Neural network Optimization) is Intel's
// toolkit for optimizing and deploying AI inference. This package enables Go
// applications to leverage OpenVINO's optimized inference engine for neural
// network architectures including transformers, CNNs, RNNs, and GNNs.
//
// Basic synchronous usage:
//
//	core, err := openvino.NewCore()
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer core.Close()
//
//	model, err := core.ReadModel("model.xml")
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer model.Close()
//
//	compiled, err := core.CompileModel(model, "CPU",
//		openvino.PerformanceHint(openvino.PerformanceModeThroughput),
//		openvino.NumStreams(4),
//	)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer compiled.Close()
//
//	request, err := compiled.CreateInferRequest()
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer request.Close()
//
//	// Set input tensor and run inference
//	err = request.SetInputTensor("input", data, shape, openvino.DataTypeFloat32)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	err = request.Infer()
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	output, err := request.GetOutputTensor("output")
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer output.Close()
//
//	result, err := output.GetDataAsFloat32()
//	if err != nil {
//		log.Fatal(err)
//	}
//
// Asynchronous inference:
//
//	// Start async inference
//	err = request.StartAsync()
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Do other work while inference runs...
//
//	// Wait for completion
//	err = request.Wait()
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Or use convenience method
//	err = request.InferAsync()
//
//	// Or with timeout
//	completed, err := request.WaitFor(5000) // 5 seconds
//	if !completed {
//		log.Fatal("inference timed out")
//	}
//
//	// Or with context support
//	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
//	defer cancel()
//	err = request.InferAsyncWithContext(ctx)
package openvino
