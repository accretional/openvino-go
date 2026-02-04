// runbench runs the benchmarks and outputs formatted comparison tables.
package main

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

type BenchResult struct {
	Name        string
	Runtime     string // "OpenVINO" or "ONNXRuntime"
	SubTest     string // e.g., "batch_4", "threads_2"
	NsPerOp     float64
	AllocsPerOp int64
	BytesPerOp  int64
	Throughput  float64 // inferences/sec if present
	MemoryMB    float64 // MB_delta if present
	Skipped     bool
}

func main() {
	// Parse arguments
	benchtime := "5x"
	filter := "."
	showHelp := false

	for i := 1; i < len(os.Args); i++ {
		arg := os.Args[i]
		if arg == "-h" || arg == "--help" {
			showHelp = true
		} else if strings.HasSuffix(arg, "x") || strings.HasSuffix(arg, "s") {
			benchtime = arg
		} else {
			filter = arg
		}
	}

	if showHelp {
		fmt.Println("Usage: runbench [benchtime] [filter]")
		fmt.Println()
		fmt.Println("Arguments:")
		fmt.Println("  benchtime  Number of iterations (e.g., 5x) or duration (e.g., 1s)")
		fmt.Println("             Default: 5x")
		fmt.Println("  filter     Regex to filter benchmark names (e.g., Infer, Thread)")
		fmt.Println("             Default: . (all benchmarks)")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  runbench              # Run all benchmarks 5 times each")
		fmt.Println("  runbench 10x          # Run all benchmarks 10 times each")
		fmt.Println("  runbench 1s           # Run each benchmark for 1 second")
		fmt.Println("  runbench 5x Infer     # Run only Infer benchmarks")
		fmt.Println("  runbench 5x Thread    # Run only Thread scaling benchmarks")
		return
	}

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║               OpenVINO vs ONNX Runtime Benchmark Comparison                  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	printBenchmarkContext()
	fmt.Printf("Running benchmarks (iterations: %s, filter: %s)...\n\n", benchtime, filter)

	cmd := exec.Command("go", "test", "-bench="+filter, "-benchmem", "-benchtime="+benchtime)
	// Find the benchmarks directory (two levels up from cmd/runbench)
	cmd.Dir = "../.."
	output, err := cmd.CombinedOutput()
	if err != nil {
		// Check if it's just skipped tests
		if !bytes.Contains(output, []byte("PASS")) && !bytes.Contains(output, []byte("SKIP")) {
			fmt.Fprintf(os.Stderr, "Benchmark failed: %v\n%s\n", err, output)
			os.Exit(1)
		}
	}

	results := parseResults(string(output))
	if len(results) == 0 {
		fmt.Println("No benchmark results found.")
		fmt.Println("Make sure the model path is set via MODEL_PATH environment variable.")
		return
	}
	printResults(results)
}

func printBenchmarkContext() {
	fmt.Println("┌─ Benchmark Configuration ────────────────────────────────────────────────────")

	// Model info - check both env vars for consistency with bench_test.go
	modelPath := os.Getenv("BENCH_MODEL")
	if modelPath == "" {
		modelPath = os.Getenv("MODEL_PATH")
	}
	if modelPath == "" {
		modelPath = "../models/all-MiniLM-L6-v2/onnx/model.onnx"
	}
	modelName := filepath.Base(filepath.Dir(filepath.Dir(modelPath)))
	if modelName == "." || modelName == "" {
		modelName = filepath.Base(modelPath)
	}
	fmt.Printf("│  Model:       %s\n", modelName)
	fmt.Printf("│  Model Path:  %s\n", modelPath)

	// System info
	fmt.Printf("│  Platform:    %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("│  CPU Cores:   %d\n", runtime.NumCPU())
	fmt.Printf("│  Go Version:  %s\n", runtime.Version())

	// Try to get OpenVINO version
	ovVersion := getOpenVINOVersion()
	if ovVersion != "" {
		fmt.Printf("│  OpenVINO:    %s\n", ovVersion)
	}

	// Try to get ONNX Runtime version
	ortVersion := getONNXRuntimeVersion()
	if ortVersion != "" {
		fmt.Printf("│  ONNX RT:     %s\n", ortVersion)
	}

	fmt.Println("│")
	fmt.Println("│  Benchmark Types:")
	fmt.Println("│    - Load:               Model loading time")
	fmt.Println("│    - Infer:              Single inference latency")
	fmt.Println("│    - InferParallel:      Parallel inference (using all cores)")
	fmt.Println("│    - FirstInference:     Cold start latency (load + first infer)")
	fmt.Println("│    - Throughput:         Maximum inferences per second")
	fmt.Println("│    - BatchSize:          Scaling with batch size (1, 2, 4, 8)")
	fmt.Println("│    - SeqLen:             Scaling with sequence length")
	fmt.Println("│    - Threads:            Thread count scaling")
	fmt.Println("│    - Memory:             Memory consumption during inference")
	fmt.Println("│    - ConcurrentSessions: Multiple model instances")
	fmt.Println("└──────────────────────────────────────────────────────────────────────────────")
	fmt.Println()
}

func getOpenVINOVersion() string {
	// Try to read from environment or a known location
	if v := os.Getenv("OPENVINO_VERSION"); v != "" {
		return v
	}
	// Try running a command to get version (if available)
	cmd := exec.Command("python3", "-c", "import openvino; print(openvino.__version__)")
	out, err := cmd.Output()
	if err == nil {
		return strings.TrimSpace(string(out))
	}
	return ""
}

func getONNXRuntimeVersion() string {
	if v := os.Getenv("ONNXRUNTIME_VERSION"); v != "" {
		return v
	}
	// Check the shared library
	libPaths := []string{
		"/usr/lib/libonnxruntime.so",
		"/usr/local/lib/libonnxruntime.so",
		os.Getenv("ONNXRUNTIME_LIB_PATH"),
	}
	for _, p := range libPaths {
		if p == "" {
			continue
		}
		if _, err := os.Stat(p); err == nil {
			// Library exists, try to get version from strings command
			cmd := exec.Command("strings", p)
			out, err := cmd.Output()
			if err == nil {
				// Look for version pattern like "1.18.0"
				re := regexp.MustCompile(`\b1\.\d+\.\d+\b`)
				matches := re.FindAllString(string(out), -1)
				if len(matches) > 0 {
					return matches[0]
				}
			}
			return "installed"
		}
	}
	return ""
}

func parseResults(output string) []BenchResult {
	var results []BenchResult

	// Parse each line for benchmark results
	scanner := bufio.NewScanner(strings.NewReader(output))
	for scanner.Scan() {
		line := scanner.Text()

		// Skip non-benchmark lines
		if !strings.HasPrefix(line, "Benchmark") {
			continue
		}

		// Check for skip
		if strings.Contains(line, "--- SKIP") {
			skipPattern := regexp.MustCompile(`--- SKIP: Benchmark(OpenVINO|ONNXRuntime)_(\w+)`)
			if matches := skipPattern.FindStringSubmatch(line); matches != nil {
				results = append(results, BenchResult{
					Runtime: matches[1],
					Name:    matches[2],
					Skipped: true,
				})
			}
			continue
		}

		// Parse benchmark result line
		result := parseBenchLine(line)
		if result != nil {
			results = append(results, *result)
		}
	}

	return results
}

func parseBenchLine(line string) *BenchResult {
	// Pattern: BenchmarkRuntime_TestName/subtest-N  iterations  time ns/op  [custom metrics]  [bytes B/op]  [allocs allocs/op]
	namePattern := regexp.MustCompile(`^Benchmark(OpenVINO|ONNXRuntime)_(\w+)(?:/(\S+))?-\d+`)
	matches := namePattern.FindStringSubmatch(line)
	if matches == nil {
		return nil
	}

	result := &BenchResult{
		Runtime: matches[1],
		Name:    matches[2],
		SubTest: matches[3],
	}

	// Extract ns/op
	nsPattern := regexp.MustCompile(`([\d.]+)\s+ns/op`)
	if m := nsPattern.FindStringSubmatch(line); m != nil {
		result.NsPerOp, _ = strconv.ParseFloat(m[1], 64)
	}

	// Extract B/op
	bPattern := regexp.MustCompile(`(\d+)\s+B/op`)
	if m := bPattern.FindStringSubmatch(line); m != nil {
		result.BytesPerOp, _ = strconv.ParseInt(m[1], 10, 64)
	}

	// Extract allocs/op
	allocPattern := regexp.MustCompile(`(\d+)\s+allocs/op`)
	if m := allocPattern.FindStringSubmatch(line); m != nil {
		result.AllocsPerOp, _ = strconv.ParseInt(m[1], 10, 64)
	}

	// Extract inferences/sec (throughput)
	tpsPattern := regexp.MustCompile(`([\d.]+)\s+inferences/sec`)
	if m := tpsPattern.FindStringSubmatch(line); m != nil {
		result.Throughput, _ = strconv.ParseFloat(m[1], 64)
	}

	// Extract MB_delta (memory)
	memPattern := regexp.MustCompile(`([\d.]+)\s+MB_delta`)
	if m := memPattern.FindStringSubmatch(line); m != nil {
		result.MemoryMB, _ = strconv.ParseFloat(m[1], 64)
	}

	return result
}

func printResults(results []BenchResult) {
	// Group results by test name
	groups := make(map[string][]BenchResult)
	for _, r := range results {
		key := r.Name
		groups[key] = append(groups[key], r)
	}

	// Sort group names in a logical order
	order := []string{"Load", "Infer", "InferParallel", "FirstInference", "Throughput",
		"BatchSize", "SeqLen", "Threads", "Memory", "ConcurrentSessions"}
	var names []string
	seen := make(map[string]bool)
	for _, name := range order {
		if _, ok := groups[name]; ok {
			names = append(names, name)
			seen[name] = true
		}
	}
	// Add any remaining groups not in our predefined order
	for name := range groups {
		if !seen[name] {
			names = append(names, name)
		}
	}

	// Print comparison tables
	for _, name := range names {
		group := groups[name]
		printGroup(name, group)
	}

	// Print summary
	printSummary(results)
}

func printGroup(name string, results []BenchResult) {
	// Print section header
	title := formatTestName(name)
	fmt.Printf("┌─ %s %s\n", title, strings.Repeat("─", 75-len(title)))

	// Check if this has subtests
	hasSubtests := false
	for _, r := range results {
		if r.SubTest != "" {
			hasSubtests = true
			break
		}
	}

	if hasSubtests {
		printSubtestTable(results)
	} else {
		printSimpleComparison(results)
	}
	fmt.Println()
}

func formatTestName(name string) string {
	switch name {
	case "Load":
		return "Model Loading"
	case "Infer":
		return "Single Inference"
	case "InferParallel":
		return "Parallel Inference"
	case "FirstInference":
		return "First Inference (Cold Start)"
	case "Throughput":
		return "Throughput"
	case "BatchSize":
		return "Batch Size Scaling"
	case "SeqLen":
		return "Sequence Length Scaling"
	case "Threads":
		return "Thread Scaling"
	case "Memory":
		return "Memory Usage"
	case "ConcurrentSessions":
		return "Concurrent Sessions"
	default:
		return name
	}
}

func printSimpleComparison(results []BenchResult) {
	var ov, ort *BenchResult
	for i := range results {
		if results[i].Runtime == "OpenVINO" {
			ov = &results[i]
		} else {
			ort = &results[i]
		}
	}

	fmt.Printf("│  %-15s %12s     %-15s %12s\n", "OpenVINO:", formatDuration(ov), "ONNX Runtime:", formatDuration(ort))

	// Show throughput if available
	hasThroughput := ov != nil && ov.Throughput > 0 && ort != nil && ort.Throughput > 0
	if hasThroughput {
		fmt.Printf("│  %-15s %12.0f     %-15s %12.0f    (inferences/sec)\n",
			"", ov.Throughput, "", ort.Throughput)
	}

	// Show memory if available
	if ov != nil && ov.MemoryMB > 0 || ort != nil && ort.MemoryMB > 0 {
		ovMem := "N/A"
		ortMem := "N/A"
		if ov != nil && ov.MemoryMB > 0 {
			ovMem = fmt.Sprintf("%.1f MB", ov.MemoryMB)
		}
		if ort != nil && ort.MemoryMB > 0 {
			ortMem = fmt.Sprintf("%.1f MB", ort.MemoryMB)
		}
		fmt.Printf("│  %-15s %12s     %-15s %12s    (memory delta)\n", "", ovMem, "", ortMem)
	}

	// Determine winner - use throughput for throughput benchmarks, time otherwise
	winner := ""
	if ov != nil && ort != nil && !ov.Skipped && !ort.Skipped {
		if hasThroughput {
			// Compare by throughput (higher is better)
			winner = compareThroughput(ov.Throughput, ort.Throughput)
		} else {
			// Compare by time (lower is better)
			winner = compareLatency(ov.NsPerOp, ort.NsPerOp)
		}
	}

	if winner != "" {
		fmt.Printf("│  Result: %s\n", winner)
	}
}

// compareLatency compares two latency values (lower is better)
// Returns a description of which is faster, or "comparable" if within 2%
func compareLatency(ovNs, ortNs float64) string {
	if ovNs == 0 || ortNs == 0 {
		return ""
	}

	ratio := ovNs / ortNs
	if ratio > 1 {
		// ORT is faster
		if ratio < 1.02 {
			return "Comparable performance"
		}
		return fmt.Sprintf("ONNX Runtime %.1fx faster", ratio)
	} else {
		// OV is faster
		ratio = ortNs / ovNs
		if ratio < 1.02 {
			return "Comparable performance"
		}
		return fmt.Sprintf("OpenVINO %.1fx faster", ratio)
	}
}

// compareThroughput compares two throughput values (higher is better)
// Returns a description of which has higher throughput, or "comparable" if within 2%
func compareThroughput(ovTps, ortTps float64) string {
	if ovTps == 0 || ortTps == 0 {
		return ""
	}

	ratio := ovTps / ortTps
	if ratio > 1 {
		// OV has higher throughput
		if ratio < 1.02 {
			return "Comparable performance"
		}
		return fmt.Sprintf("OpenVINO %.1fx higher throughput", ratio)
	} else {
		// ORT has higher throughput
		ratio = ortTps / ovTps
		if ratio < 1.02 {
			return "Comparable performance"
		}
		return fmt.Sprintf("ONNX Runtime %.1fx higher throughput", ratio)
	}
}

// getWinnerByLatency returns "ov", "ort", or "" (tie) based on latency comparison
// Uses 2% threshold for determining comparable performance
func getWinnerByLatency(ovNs, ortNs float64) string {
	if ovNs == 0 || ortNs == 0 {
		return ""
	}

	ratio := ovNs / ortNs
	if ratio > 1.02 {
		return "ort" // ORT is faster by more than 2%
	} else if ratio < 0.98 {
		return "ov" // OV is faster by more than 2%
	}
	return "" // Within 2%, comparable
}

// getWinnerByThroughput returns "ov", "ort", or "" (tie) based on throughput comparison
// Uses 2% threshold for determining comparable performance
func getWinnerByThroughput(ovTps, ortTps float64) string {
	if ovTps == 0 || ortTps == 0 {
		return ""
	}

	ratio := ovTps / ortTps
	if ratio > 1.02 {
		return "ov" // OV has higher throughput by more than 2%
	} else if ratio < 0.98 {
		return "ort" // ORT has higher throughput by more than 2%
	}
	return "" // Within 2%, comparable
}

func printSubtestTable(results []BenchResult) {
	// Group by subtest
	subtests := make(map[string]map[string]*BenchResult)
	var subtestOrder []string

	for i := range results {
		r := &results[i]
		if _, ok := subtests[r.SubTest]; !ok {
			subtests[r.SubTest] = make(map[string]*BenchResult)
			subtestOrder = append(subtestOrder, r.SubTest)
		}
		subtests[r.SubTest][r.Runtime] = r
	}

	// Sort subtests naturally
	sort.Slice(subtestOrder, func(i, j int) bool {
		return naturalLess(subtestOrder[i], subtestOrder[j])
	})

	fmt.Printf("│  %-18s %14s %14s %18s\n", "Variant", "OpenVINO", "ONNX Runtime", "Comparison")
	fmt.Printf("│  %s\n", strings.Repeat("─", 68))

	for _, subtest := range subtestOrder {
		runtimes := subtests[subtest]
		ov := runtimes["OpenVINO"]
		ort := runtimes["ONNXRuntime"]

		ovTime := formatDuration(ov)
		ortTime := formatDuration(ort)

		comparison := ""
		if ov != nil && ort != nil && !ov.Skipped && !ort.Skipped {
			comparison = compareLatencyShort(ov.NsPerOp, ort.NsPerOp)
		}

		fmt.Printf("│  %-18s %14s %14s %18s\n", subtest, ovTime, ortTime, comparison)
	}
}

// compareLatencyShort returns a short comparison string for table display
func compareLatencyShort(ovNs, ortNs float64) string {
	if ovNs == 0 || ortNs == 0 {
		return ""
	}

	ratio := ovNs / ortNs
	if ratio > 1 {
		// ORT is faster
		if ratio < 1.02 {
			return "~comparable"
		}
		return fmt.Sprintf("ORT %.1fx faster", ratio)
	} else {
		// OV is faster
		ratio = ortNs / ovNs
		if ratio < 1.02 {
			return "~comparable"
		}
		return fmt.Sprintf("OV %.1fx faster", ratio)
	}
}

func printSummary(results []BenchResult) {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                   SUMMARY                                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════╝")

	// Count wins
	ovWins := 0
	ortWins := 0
	ties := 0

	// Group by test
	groups := make(map[string][]BenchResult)
	for _, r := range results {
		if r.Skipped {
			continue
		}
		key := r.Name + "/" + r.SubTest
		groups[key] = append(groups[key], r)
	}

	for key, group := range groups {
		var ov, ort *BenchResult
		for i := range group {
			if group[i].Runtime == "OpenVINO" {
				ov = &group[i]
			} else {
				ort = &group[i]
			}
		}
		if ov != nil && ort != nil {
			// For throughput benchmarks, compare by throughput (higher is better)
			if strings.HasPrefix(key, "Throughput") && ov.Throughput > 0 && ort.Throughput > 0 {
				winner := getWinnerByThroughput(ov.Throughput, ort.Throughput)
				switch winner {
				case "ov":
					ovWins++
				case "ort":
					ortWins++
				default:
					ties++
				}
			} else {
				// Compare by latency (lower is better)
				winner := getWinnerByLatency(ov.NsPerOp, ort.NsPerOp)
				switch winner {
				case "ov":
					ovWins++
				case "ort":
					ortWins++
				default:
					ties++
				}
			}
		}
	}

	total := ovWins + ortWins + ties
	fmt.Println()
	fmt.Printf("  Test Results:  OpenVINO wins %d/%d  |  ONNX Runtime wins %d/%d", ovWins, total, ortWins, total)
	if ties > 0 {
		fmt.Printf("  |  Comparable: %d", ties)
	}
	fmt.Println()

	// Find key metrics
	var ovInfer, ortInfer, ovThroughput, ortThroughput *BenchResult
	for i := range results {
		r := &results[i]
		if r.Name == "Infer" && r.SubTest == "" {
			if r.Runtime == "OpenVINO" {
				ovInfer = r
			} else {
				ortInfer = r
			}
		}
		if r.Name == "Throughput" {
			if r.Runtime == "OpenVINO" {
				ovThroughput = r
			} else {
				ortThroughput = r
			}
		}
	}

	fmt.Println()
	fmt.Println("  Key Metrics:")
	if ovInfer != nil && ortInfer != nil {
		fmt.Printf("    Single Inference:  OpenVINO %s  vs  ONNX Runtime %s\n",
			formatDuration(ovInfer), formatDuration(ortInfer))
	}
	if ovThroughput != nil && ortThroughput != nil {
		fmt.Printf("    Throughput:        OpenVINO %.0f inf/s  vs  ONNX Runtime %.0f inf/s\n",
			ovThroughput.Throughput, ortThroughput.Throughput)
	}
	fmt.Println()
}

func formatDuration(r *BenchResult) string {
	if r == nil {
		return "N/A"
	}
	if r.Skipped {
		return "SKIP"
	}

	ns := r.NsPerOp
	switch {
	case ns >= 1e9:
		return fmt.Sprintf("%.2fs", ns/1e9)
	case ns >= 1e6:
		return fmt.Sprintf("%.2fms", ns/1e6)
	case ns >= 1e3:
		return fmt.Sprintf("%.2fµs", ns/1e3)
	default:
		return fmt.Sprintf("%.0fns", ns)
	}
}

func getAllocs(r *BenchResult) int64 {
	if r == nil {
		return 0
	}
	return r.AllocsPerOp
}

func naturalLess(a, b string) bool {
	// Extract numbers for natural sorting
	numA := extractNumber(a)
	numB := extractNumber(b)
	if numA != numB {
		return numA < numB
	}
	return a < b
}

func extractNumber(s string) int {
	re := regexp.MustCompile(`\d+`)
	match := re.FindString(s)
	if match == "" {
		return 0
	}
	n, _ := strconv.Atoi(match)
	return n
}
