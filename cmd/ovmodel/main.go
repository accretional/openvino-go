// ovmodel - OpenVINO Model Downloader CLI
// A simple tool to download models for use with openvino-go bindings
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	defaultModelDir = "models"
	huggingFaceBase = "https://huggingface.co"
)

type ModelSource string

const (
	SourceHuggingFace ModelSource = "huggingface"
	SourceURL         ModelSource = "url"
	SourceModelZoo    ModelSource = "modelzoo"
)

type ModelInfo struct {
	ID       string
	Source   ModelSource
	URL      string
	Format   string // "onnx", "xml", "auto"
	Checksum string // optional SHA256
}

var knownModels = map[string]ModelInfo{
	"test-model": {
		ID:     "test-model",
		Source: SourceURL,
		// This is a very small model that works well for hello-world examples
		URL:    "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
		Format: "onnx",
	},
	"all-MiniLM-L6-v2": {
		ID:     "sentence-transformers/all-MiniLM-L6-v2",
		Source: SourceHuggingFace,
		Format: "onnx",
	},
	"all-mpnet-base-v2": {
		ID:     "sentence-transformers/all-mpnet-base-v2",
		Source: SourceHuggingFace,
		Format: "onnx",
	},
}

func main() {
	var (
		modelID    = flag.String("model", "", "Model ID or alias (e.g., 'all-MiniLM-L6-v2', 'sentence-transformers/all-MiniLM-L6-v2')")
		url        = flag.String("url", "", "Direct URL to download model from")
		outputDir  = flag.String("output", defaultModelDir, "Output directory for models")
		listModels = flag.Bool("list", false, "List available model aliases")
		format     = flag.String("format", "auto", "Model format: onnx, xml, or auto (default: auto)")
		force      = flag.Bool("force", false, "Force re-download even if model exists")
	)
	flag.Parse()

	if *listModels {
		listAvailableModels()
		return
	}

	if *modelID == "" && *url == "" {
		fmt.Fprintf(os.Stderr, "Error: must specify either -model or -url\n\n")
		flag.Usage()
		os.Exit(1)
	}

	modelDir := *outputDir
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Error: failed to create model directory: %v\n", err)
		os.Exit(1)
	}

	var info ModelInfo
	if *url != "" {
		info = ModelInfo{
			ID:     filepath.Base(*url),
			Source: SourceURL,
			URL:    *url,
			Format: *format,
		}
	} else {
		// Check if it's a known alias
		if known, ok := knownModels[*modelID]; ok {
			info = known
		} else {
			// Assume it's a HuggingFace model ID
			info = ModelInfo{
				ID:     *modelID,
				Source: SourceHuggingFace,
				Format: *format,
			}
		}
	}

	if err := downloadModel(info, modelDir, *force); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func listAvailableModels() {
	fmt.Println("Available model aliases:")
	fmt.Println()
	for alias, info := range knownModels {
		fmt.Printf("  %-30s -> %s (%s)\n", alias, info.ID, info.Source)
	}
	fmt.Println()
	fmt.Println("You can also use any HuggingFace model ID directly:")
	fmt.Println("  ovmodel -model sentence-transformers/all-MiniLM-L6-v2")
	fmt.Println()
	fmt.Println("Or download from a direct URL:")
	fmt.Println("  ovmodel -url https://example.com/model.onnx")
}

func downloadModel(info ModelInfo, modelDir string, force bool) error {
	fmt.Printf("Downloading model: %s\n", info.ID)

	var modelPath string
	var err error

	switch info.Source {
	case SourceHuggingFace:
		modelPath, err = downloadFromHuggingFace(info, modelDir, force)
	case SourceURL:
		modelPath, err = downloadFromURL(info, modelDir, force)
	case SourceModelZoo:
		return fmt.Errorf("model zoo download not yet implemented")
	default:
		return fmt.Errorf("unknown source: %s", info.Source)
	}

	if err != nil {
		return err
	}

	fmt.Printf("✓ Model saved to: %s\n", modelPath)
	return nil
}

func downloadFromHuggingFace(info ModelInfo, modelDir string, force bool) (string, error) {
	modelID := info.ID
	modelName := strings.ReplaceAll(modelID, "/", "_")

	// Determine file to download based on format
	var fileName string
	switch info.Format {
	case "onnx":
		fileName = "model.onnx"
	case "xml":
		fileName = "model.xml"
	case "auto":
		// Try ONNX first (more common), then XML
		fileName = "model.onnx"
	}

	outputPath := filepath.Join(modelDir, modelName, fileName)

	// Check if already exists
	if !force {
		if _, err := os.Stat(outputPath); err == nil {
			fmt.Printf("✓ Model already exists at %s (use -force to re-download)\n", outputPath)
			return outputPath, nil
		}
	}

	// Create model directory
	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		return "", fmt.Errorf("failed to create directory: %w", err)
	}

	// Try ONNX first, then XML
	urls := []string{
		fmt.Sprintf("%s/%s/resolve/main/%s", huggingFaceBase, modelID, "model.onnx"),
		fmt.Sprintf("%s/%s/resolve/main/onnx/%s", huggingFaceBase, modelID, "model.onnx"),
		fmt.Sprintf("%s/%s/resolve/main/%s", huggingFaceBase, modelID, "model.xml"),
	}

	var lastErr error
	for _, url := range urls {
		fmt.Printf("  Trying: %s\n", url)
		if err := downloadFile(url, outputPath); err == nil {
			fmt.Printf("  ✓ Downloaded from: %s\n", url)
			return outputPath, nil
		} else {
			lastErr = err
			fmt.Printf("  ✗ Failed: %v\n", err)
		}
	}

	return "", fmt.Errorf("failed to download model from any URL: %w", lastErr)
}

func downloadFromURL(info ModelInfo, modelDir string, force bool) (string, error) {
	url := info.URL
	fileName := filepath.Base(url)
	// Remove query parameters
	if idx := strings.Index(fileName, "?"); idx != -1 {
		fileName = fileName[:idx]
	}

	// Special handling for test-model: save as test_model.onnx
	if info.ID == "test-model" {
		fileName = "test_model.onnx"
	}

	outputPath := filepath.Join(modelDir, fileName)

	// Check if already exists
	if !force {
		if _, err := os.Stat(outputPath); err == nil {
			fmt.Printf("✓ Model already exists at %s (use -force to re-download)\n", outputPath)
			return outputPath, nil
		}
	}

	if err := downloadFile(url, outputPath); err != nil {
		return "", fmt.Errorf("failed to download from URL: %w", err)
	}

	return outputPath, nil
}

func downloadFile(url, outputPath string) error {
	client := &http.Client{
		Timeout: 30 * time.Minute,
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "ovmodel/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	// Check content length for progress
	contentLength := resp.ContentLength
	if contentLength > 0 {
		fmt.Printf("  Downloading %s (%.2f MB)...\n", filepath.Base(outputPath), float64(contentLength)/(1024*1024))
	}

	// Create output file
	out, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer out.Close()

	// Download with progress
	var written int64
	buf := make([]byte, 32*1024) // 32KB buffer
	for {
		nr, er := resp.Body.Read(buf)
		if nr > 0 {
			nw, ew := out.Write(buf[0:nr])
			if nw < 0 || nr < nw {
				nw = 0
				if ew == nil {
					ew = fmt.Errorf("invalid write")
				}
			}
			written += int64(nw)
			if ew != nil {
				return ew
			}
			if nr != nw {
				return io.ErrShortWrite
			}

			// Print progress
			if contentLength > 0 && written%(10*1024*1024) == 0 { // Every 10MB
				percent := float64(written) / float64(contentLength) * 100
				fmt.Printf("  Progress: %.1f%%\n", percent)
			}
		}
		if er != nil {
			if er != io.EOF {
				return er
			}
			break
		}
	}

	return nil
}

func verifyChecksum(filePath, expectedHash string) error {
	if expectedHash == "" {
		return nil
	}

	f, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, f); err != nil {
		return err
	}

	actualHash := hex.EncodeToString(hash.Sum(nil))
	if actualHash != expectedHash {
		return fmt.Errorf("checksum mismatch: expected %s, got %s", expectedHash, actualHash)
	}

	return nil
}
