package openvino

import "github.com/accretional/openvino-go/internal/cgo"

type ProfilingInfoStatus = cgo.ProfilingInfoStatus

const (
	ProfilingInfoStatusNotRun       = cgo.ProfilingInfoStatusNotRun
	ProfilingInfoStatusOptimizedOut = cgo.ProfilingInfoStatusOptimizedOut
	ProfilingInfoStatusExecuted     = cgo.ProfilingInfoStatusExecuted
)

type ProfilingInfo struct {
	Status   ProfilingInfoStatus
	RealTime int64
	CPUTime  int64
	NodeName string
	ExecType string
	NodeType string
}

func (ir *InferRequest) GetProfilingInfo() ([]ProfilingInfo, error) {
	cgoInfos, err := ir.request.GetProfilingInfo()
	if err != nil {
		return nil, err
	}

	infos := make([]ProfilingInfo, len(cgoInfos))
	for i, cgoInfo := range cgoInfos {
		infos[i] = ProfilingInfo{
			Status:   cgoInfo.Status,
			RealTime: cgoInfo.RealTime,
			CPUTime:  cgoInfo.CPUTime,
			NodeName: cgoInfo.NodeName,
			ExecType: cgoInfo.ExecType,
			NodeType: cgoInfo.NodeType,
		}
	}

	return infos, nil
}
