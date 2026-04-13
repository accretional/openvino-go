.PHONY: help setup build test tidy setup-linux setup-mac build-linux build-mac

.DEFAULT_GOAL := help

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

help:
	@echo "openvino-go"
	@echo "  make setup        OpenVINO + build tools (Linux: apt; macOS: toolkit)"
	@echo "  make build        C++ wrapper for this OS"
	@echo "  make test         go test with CGO"
	@echo "  make tidy         go mod tidy"
	@echo "  make setup-linux | setup-mac | build-linux | build-mac  (direct scripts)"

setup:
	@case "$$(uname -s)" in \
		Linux) "$(ROOT)scripts/setup-linux.sh" ;; \
		Darwin) "$(ROOT)scripts/setup-mac.sh" ;; \
		*) echo "make setup: only Linux and macOS are supported" >&2; exit 1 ;; \
	esac

setup-linux:
	"$(ROOT)scripts/setup-linux.sh"

setup-mac:
	"$(ROOT)scripts/setup-mac.sh"

build:
	@case "$$(uname -s)" in \
		Linux) "$(ROOT)scripts/build-linux.sh" ;; \
		Darwin) "$(ROOT)scripts/build-mac.sh" ;; \
		*) echo "make build: only Linux and macOS are supported" >&2; exit 1 ;; \
	esac

build-linux:
	"$(ROOT)scripts/build-linux.sh"

build-mac:
	"$(ROOT)scripts/build-mac.sh"

test:
	CGO_ENABLED=1 go test ./... -v -count=1

tidy:
	go mod tidy
