// Text Embedding example for openvino-go
// This demonstrates how to use OpenVINO for text embedding inference
// with transformer models (BERT, sentence-transformers, etc.)
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"time"

	"github.com/accretional/openvino-go/pkg/openvino"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: text-embedding <model.xml|model.onnx> <text>")
		fmt.Println("\nExample:")
		fmt.Println("  text-embedding model.xml \"Hello, world!\"")
		fmt.Println("\nNote:")
		fmt.Println("  - Model should be a text embedding model (e.g., sentence-transformers)")
		fmt.Println("  - For proper tokenization, you may need to preprocess text separately")
		fmt.Println("  - This example uses a simplified tokenization approach")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	text := strings.Join(os.Args[2:], " ")

	if !strings.HasSuffix(modelPath, ".xml") && !strings.HasSuffix(modelPath, ".onnx") {
		log.Fatalf("Model must be a .xml (OpenVINO IR) or .onnx file, got: %s", modelPath)
	}

	fmt.Printf("Text Embedding Example\n")
	fmt.Printf("======================\n\n")
	fmt.Printf("Model: %s\n", modelPath)
	fmt.Printf("Text: %q\n\n", text)

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

	fmt.Printf("\nLoading model from: %s\n", modelPath)
	model, err := core.ReadModel(modelPath)
	if err != nil {
		log.Fatalf("Failed to read model: %v", err)
	}
	defer model.Close()

	fmt.Println("\n=== Model I/O Information ===")
	inputs, err := model.GetInputs()
	if err != nil {
		log.Fatalf("Failed to get input info: %v", err)
	}
	fmt.Printf("Model has %d input(s):\n", len(inputs))
	for i, input := range inputs {
		fmt.Printf("  Input %d: name='%s', shape=%v, type=%d\n", i, input.Name, input.Shape, input.DataType)
	}

	outputs, err := model.GetOutputs()
	if err != nil {
		log.Fatalf("Failed to get output info: %v", err)
	}
	fmt.Printf("Model has %d output(s):\n", len(outputs))
	for i, output := range outputs {
		fmt.Printf("  Output %d: name='%s', shape=%v, type=%d\n", i, output.Name, output.Shape, output.DataType)
	}

	fmt.Println("\nCompiling model for CPU...")
	device := "CPU"
	if len(devices) > 0 {
		device = devices[0]
	}

	compiledModel, err := core.CompileModel(model, device,
		openvino.PerformanceHint(openvino.PerformanceModeLatency),
	)
	if err != nil {
		log.Fatalf("Failed to compile model: %v", err)
	}
	defer compiledModel.Close()
	fmt.Printf("Model compiled successfully for device: %s\n", device)

	fmt.Println("\nCreating inference request...")
	request, err := compiledModel.CreateInferRequest()
	if err != nil {
		log.Fatalf("Failed to create infer request: %v", err)
	}
	defer request.Close()

	fmt.Println("\nPreparing input data...")
	inputData, attentionMask, err := prepareTextInput(text, inputs)
	if err != nil {
		log.Fatalf("Failed to prepare input: %v", err)
	}

	fmt.Println("Setting input tensors...")
	err = setInputTensors(request, inputs, inputData, attentionMask)
	if err != nil {
		log.Fatalf("Failed to set input tensors: %v", err)
	}

	fmt.Println("Running inference...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	startTime := time.Now()
	err = request.InferWithContext(ctx)
	if err != nil {
		if err == context.DeadlineExceeded {
			log.Fatalf("Inference timed out after 30 seconds")
		}
		log.Fatalf("Failed to run inference: %v", err)
	}
	inferenceTime := time.Since(startTime)
	fmt.Printf("Inference completed in %v\n", inferenceTime)

	fmt.Println("\nExtracting embedding...")
	outputTensor, err := getOutputTensor(request, outputs)
	if err != nil {
		log.Fatalf("Failed to get output tensor: %v", err)
	}
	defer outputTensor.Close()

	outputData, err := outputTensor.GetDataAsFloat32()
	if err != nil {
		log.Fatalf("Failed to get output data: %v", err)
	}

	outputShape, err := outputTensor.GetShape()
	if err != nil {
		log.Fatalf("Failed to get output shape: %v", err)
	}

	fmt.Printf("Output shape: %v\n", outputShape)
	fmt.Printf("Output size: %d elements\n", len(outputData))

	embedding := extractEmbedding(outputData, outputShape, len(inputData))
	fmt.Printf("Embedding dimension: %d\n", len(embedding))

	normalizedEmbedding := normalizeL2(embedding)
	fmt.Printf("Normalized embedding dimension: %d\n", len(normalizedEmbedding))

	fmt.Println("\n=== Results ===")
	fmt.Printf("First 10 embedding values:\n")
	for i := 0; i < len(normalizedEmbedding) && i < 10; i++ {
		fmt.Printf("  [%d] = %.6f\n", i, normalizedEmbedding[i])
	}

	var sum, min, max float32
	min = normalizedEmbedding[0]
	max = normalizedEmbedding[0]
	for _, val := range normalizedEmbedding {
		sum += val
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}
	mean := sum / float32(len(normalizedEmbedding))

	fmt.Printf("\nEmbedding statistics:\n")
	fmt.Printf("  Mean: %.6f\n", mean)
	fmt.Printf("  Min:  %.6f\n", min)
	fmt.Printf("  Max:  %.6f\n", max)
	fmt.Printf("  L2 norm: %.6f\n", l2Norm(normalizedEmbedding))

	fmt.Println("\nText embedding example completed successfully!")
}

// Testing only. For production use, you should use a proper tokenizer (like HuggingFace tokenizer)
func prepareTextInput(text string, inputs []openvino.PortInfo) ([]int64, []int64, error) {
	words := strings.Fields(strings.ToLower(text))
	maxSeqLen := 512

	if len(inputs) > 0 {
		inputShape := inputs[0].Shape
		if len(inputShape) >= 2 && inputShape[1] > 0 {
			maxSeqLen = int(inputShape[1]) // -1 means dynamic; keep default
		}
	}

	tokenIDs := make([]int64, 0, len(words)+2)
	tokenIDs = append(tokenIDs, 101)

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
		tokenIDs = append(tokenIDs, 102)
	}

	for len(tokenIDs) < maxSeqLen {
		tokenIDs = append(tokenIDs, 0)
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

	fmt.Printf("Tokenized text: %d tokens (max: %d)\n", realTokens, maxSeqLen)
	fmt.Printf("First 10 token IDs: %v\n", tokenIDs[:min(10, len(tokenIDs))])

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
			for i, inp := range inputs {
				if inp.Name == input.Name {
					err = request.SetInputTensorByIndex(int32(i), data, shape, dataType)
					break
				}
			}
			if err != nil {
				return fmt.Errorf("failed to set input '%s': %w", input.Name, err)
			}
		}
		fmt.Printf("  Set input '%s': shape=%v, type=%d\n", input.Name, shape, dataType)
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
