// Text Embedding Async Example for openvino-go
// This demonstrates how to use asynchronous inference for batch text embedding processing
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/accretional/openvino-go/pkg/openvino"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: text-embedding-async <model.xml|model.onnx> <text1> [text2] [text3] ...")
		fmt.Println("\nExample:")
		fmt.Println("  text-embedding-async model.onnx \"Hello, world!\" \"How are you?\" \"Good morning!\"")
		fmt.Println("\nNote:")
		fmt.Println("  - Model should be a text embedding model (e.g., sentence-transformers)")
		fmt.Println("  - This example demonstrates async inference for processing multiple texts concurrently")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	texts := os.Args[2:]

	if !strings.HasSuffix(modelPath, ".xml") && !strings.HasSuffix(modelPath, ".onnx") {
		log.Fatalf("Model must be a .xml (OpenVINO IR) or .onnx file, got: %s", modelPath)
	}

	fmt.Printf("Text Embedding Async Example\n")
	fmt.Printf("============================\n\n")
	fmt.Printf("Model: %s\n", modelPath)
	fmt.Printf("Texts to process: %d\n\n", len(texts))

	// Step 1: Create OpenVINO Core
	fmt.Println("Creating OpenVINO Core...")
	core, err := openvino.NewCore()
	if err != nil {
		log.Fatalf("Failed to create core: %v", err)
	}
	defer core.Close()

	devices, err := core.GetAvailableDevices()
	if err != nil {
		log.Fatalf("Failed to get devices: %v", err)
	}
	fmt.Printf("Available devices: %v\n", devices)

	// Step 2: Load model
	fmt.Printf("\nLoading model from: %s\n", modelPath)
	model, err := core.ReadModel(modelPath)
	if err != nil {
		log.Fatalf("Failed to read model: %v", err)
	}
	defer model.Close()

	inputs, err := model.GetInputs()
	if err != nil {
		log.Fatalf("Failed to get input info: %v", err)
	}

	outputs, err := model.GetOutputs()
	if err != nil {
		log.Fatalf("Failed to get output info: %v", err)
	}

	// Step 3: Compile model with throughput optimization for async inference
	fmt.Println("\nCompiling model for CPU with throughput optimization...")
	device := "CPU"
	if len(devices) > 0 {
		device = devices[0]
	}

	compiledModel, err := core.CompileModel(model, device,
		openvino.PerformanceHint(openvino.PerformanceModeThroughput),
		openvino.NumStreams(4), // Multiple streams for parallel async requests
	)
	if err != nil {
		log.Fatalf("Failed to compile model: %v", err)
	}
	defer compiledModel.Close()
	fmt.Printf("Model compiled successfully for device: %s\n", device)

	// Step 4: Process texts using async inference
	fmt.Printf("\nProcessing %d texts using async inference...\n", len(texts))
	fmt.Println()

	// Create multiple inference requests for parallel processing
	numRequests := min(len(texts), 4) // Use up to 4 parallel requests
	requests := make([]*openvino.InferRequest, numRequests)
	for i := 0; i < numRequests; i++ {
		req, err := compiledModel.CreateInferRequest()
		if err != nil {
			log.Fatalf("Failed to create infer request %d: %v", i, err)
		}
		requests[i] = req
	}
	defer func() {
		for _, req := range requests {
			req.Close()
		}
	}()

	// Process texts concurrently
	startTime := time.Now()
	results := processTextsAsync(requests, texts, inputs, outputs)
	totalTime := time.Since(startTime)

	// Display results
	fmt.Println("\n=== Results ===")
	for i, result := range results {
		if result.err != nil {
			fmt.Printf("Text %d: Error - %v\n", i+1, result.err)
			continue
		}
		fmt.Printf("\nText %d: %q\n", i+1, result.text)
		fmt.Printf("  Embedding dimension: %d\n", len(result.embedding))
		fmt.Printf("  First 5 values: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
			result.embedding[0], result.embedding[1], result.embedding[2],
			result.embedding[3], result.embedding[4])
		fmt.Printf("  Processing time: %v\n", result.processingTime)
	}

	fmt.Printf("\n=== Performance Summary ===\n")
	fmt.Printf("Total texts processed: %d\n", len(texts))
	fmt.Printf("Total time: %v\n", totalTime)
	fmt.Printf("Average time per text: %v\n", totalTime/time.Duration(len(texts)))
	fmt.Printf("Throughput: %.2f texts/second\n", float64(len(texts))/totalTime.Seconds())

	fmt.Println("\nAsync inference example completed successfully!")
}

type embeddingResult struct {
	text           string
	embedding      []float32
	processingTime time.Duration
	err            error
}

func processTextsAsync(requests []*openvino.InferRequest, texts []string, inputs []openvino.PortInfo, outputs []openvino.PortInfo) []embeddingResult {
	results := make([]embeddingResult, len(texts))
	var wg sync.WaitGroup

	// Use a channel to manage available requests
	requestChan := make(chan *openvino.InferRequest, len(requests))
	for _, req := range requests {
		requestChan <- req
	}

	// Process each text
	for i, text := range texts {
		wg.Add(1)
		go func(idx int, txt string) {
			defer wg.Done()

			// Get an available request
			req := <-requestChan
			defer func() {
				requestChan <- req // Return request to pool
			}()

			startTime := time.Now()
			embedding, err := processTextAsync(req, txt, inputs, outputs)
			processingTime := time.Since(startTime)

			results[idx] = embeddingResult{
				text:           txt,
				embedding:      embedding,
				processingTime: processingTime,
				err:            err,
			}
		}(i, text)
	}

	wg.Wait()
	return results
}

func processTextAsync(request *openvino.InferRequest, text string, inputs []openvino.PortInfo, outputs []openvino.PortInfo) ([]float32, error) {
	// Prepare input data
	inputData, attentionMask, err := prepareTextInput(text, inputs)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare input: %w", err)
	}

	// Set input tensors
	err = setInputTensors(request, inputs, inputData, attentionMask)
	if err != nil {
		return nil, fmt.Errorf("failed to set input tensors: %w", err)
	}

	// Start async inference
	err = request.StartAsync()
	if err != nil {
		return nil, fmt.Errorf("failed to start async inference: %w", err)
	}

	// Wait for completion with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Use WaitFor with timeout, or Wait if context is not cancelled
	completed, err := request.WaitFor(30000) // 30 seconds
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	if !completed {
		return nil, fmt.Errorf("inference timed out")
	}

	// Get output tensor
	outputTensor, err := getOutputTensor(request, outputs)
	if err != nil {
		return nil, fmt.Errorf("failed to get output tensor: %w", err)
	}
	defer outputTensor.Close()

	outputData, err := outputTensor.GetDataAsFloat32()
	if err != nil {
		return nil, fmt.Errorf("failed to get output data: %w", err)
	}

	outputShape, err := outputTensor.GetShape()
	if err != nil {
		return nil, fmt.Errorf("failed to get output shape: %w", err)
	}

	// Extract and normalize embedding
	embedding := extractEmbedding(outputData, outputShape, len(inputData))
	normalizedEmbedding := normalizeL2(embedding)

	// Check context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
		return normalizedEmbedding, nil
	}
}

// Helper functions (same as text-embedding example)
func prepareTextInput(text string, inputs []openvino.PortInfo) ([]int64, []int64, error) {
	words := strings.Fields(strings.ToLower(text))
	maxSeqLen := 512

	if len(inputs) > 0 {
		inputShape := inputs[0].Shape
		if len(inputShape) >= 2 && inputShape[1] > 0 {
			maxSeqLen = int(inputShape[1])
		}
	}

	tokenIDs := make([]int64, 0, len(words)+2)
	tokenIDs = append(tokenIDs, 101) // [CLS]

	for _, word := range words {
		tokenID := int64(hashString(word)) % 30000
		if tokenID < 100 {
			tokenID = 1000 + tokenID
		}
		tokenIDs = append(tokenIDs, tokenID)
		if len(tokenIDs) >= maxSeqLen-1 {
			break
		}
	}

	if len(tokenIDs) < maxSeqLen {
		tokenIDs = append(tokenIDs, 102) // [SEP]
	}

	for len(tokenIDs) < maxSeqLen {
		tokenIDs = append(tokenIDs, 0) // [PAD]
	}

	if len(tokenIDs) > maxSeqLen {
		tokenIDs = tokenIDs[:maxSeqLen]
		tokenIDs[maxSeqLen-1] = 102
	}

	attentionMask := make([]int64, maxSeqLen)
	realTokens := len(tokenIDs)
	for i := 0; i < maxSeqLen; i++ {
		if i < realTokens && tokenIDs[i] != 0 {
			attentionMask[i] = 1
		} else {
			attentionMask[i] = 0
		}
	}

	return tokenIDs, attentionMask, nil
}

func setInputTensors(request *openvino.InferRequest, inputs []openvino.PortInfo, inputIDs []int64, attentionMask []int64) error {
	for _, input := range inputs {
		inputName := strings.ToLower(input.Name)
		var data interface{}
		var shape []int64
		var dataType openvino.DataType

		if len(input.Shape) > 0 {
			shape = make([]int64, len(input.Shape))
			for i, dim := range input.Shape {
				if dim == -1 {
					if i == 1 {
						shape[i] = int64(len(inputIDs))
					} else {
						shape[i] = 1
					}
				} else {
					shape[i] = int64(dim)
				}
			}
		} else {
			shape = []int64{1, int64(len(inputIDs))}
		}

		dataType = input.DataType
		if dataType == 0 {
			dataType = openvino.DataTypeInt64
		}

		if strings.Contains(inputName, "input_ids") || strings.Contains(inputName, "input") {
			data = inputIDs
		} else if strings.Contains(inputName, "attention_mask") || strings.Contains(inputName, "attention") {
			data = attentionMask
		} else if strings.Contains(inputName, "token_type_ids") || strings.Contains(inputName, "token_type") {
			tokenTypeIDs := make([]int64, len(inputIDs))
			for i := range tokenTypeIDs {
				tokenTypeIDs[i] = 0
			}
			data = tokenTypeIDs
		} else {
			data = inputIDs
		}

		err := request.SetInputTensor(input.Name, data, shape, dataType)
		if err != nil {
			for idx, inp := range inputs {
				if inp.Name == input.Name {
					err = request.SetInputTensorByIndex(int32(idx), data, shape, dataType)
					break
				}
			}
			if err != nil {
				return fmt.Errorf("failed to set input '%s': %w", input.Name, err)
			}
		}
	}

	return nil
}

func getOutputTensor(request *openvino.InferRequest, outputs []openvino.PortInfo) (*openvino.Tensor, error) {
	if len(outputs) == 0 {
		return nil, fmt.Errorf("no outputs found in model")
	}

	// Prefer sentence_embedding (pooled), then token_embeddings; fall back to first output
	for _, out := range outputs {
		if strings.Contains(strings.ToLower(out.Name), "sentence_embedding") {
			t, err := request.GetOutputTensor(out.Name)
			if err == nil {
				return t, nil
			}
		}
	}
	for _, out := range outputs {
		if strings.Contains(strings.ToLower(out.Name), "token_embeddings") {
			t, err := request.GetOutputTensor(out.Name)
			if err == nil {
				return t, nil
			}
		}
	}

	outputTensor, err := request.GetOutputTensor(outputs[0].Name)
	if err != nil {
		outputTensor, err = request.GetOutputTensorByIndex(0)
		if err != nil {
			return nil, fmt.Errorf("failed to get output tensor: %w", err)
		}
	}
	return outputTensor, nil
}

func extractEmbedding(outputData []float32, outputShape []int32, seqLen int) []float32 {
	if len(outputShape) == 0 {
		return outputData
	}

	if len(outputShape) == 2 {
		embeddingDim := int(outputShape[1])
		if len(outputData) >= embeddingDim {
			return outputData[:embeddingDim]
		}
		return outputData
	}

	if len(outputShape) == 3 {
		batchSize := int(outputShape[0])
		seqLength := int(outputShape[1])
		embeddingDim := int(outputShape[2])

		if batchSize == 0 || seqLength == 0 || embeddingDim == 0 {
			return outputData
		}

		embedding := make([]float32, embeddingDim)
		for i := 0; i < embeddingDim; i++ {
			var sum float32
			count := 0
			for j := 0; j < seqLength && j < seqLen; j++ {
				idx := j*embeddingDim + i
				if idx < len(outputData) {
					sum += outputData[idx]
					count++
				}
			}
			if count > 0 {
				embedding[i] = sum / float32(count)
			}
		}
		return embedding
	}

	return outputData
}

func normalizeL2(vec []float32) []float32 {
	norm := l2Norm(vec)
	if norm == 0 {
		return vec
	}

	normalized := make([]float32, len(vec))
	for i, v := range vec {
		normalized[i] = v / norm
	}
	return normalized
}

func l2Norm(vec []float32) float32 {
	var sum float64
	for _, v := range vec {
		sum += float64(v * v)
	}
	return float32(math.Sqrt(sum))
}

func hashString(s string) uint32 {
	var hash uint32 = 2166136261
	for _, c := range s {
		hash ^= uint32(c)
		hash *= 16777619
	}
	return hash
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
