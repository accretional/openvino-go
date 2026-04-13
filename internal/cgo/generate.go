package cgo

//go:generate sh -c "cd ../.. && if [ \"$(uname -s)\" = \"Darwin\" ]; then ./scripts/build-mac.sh; else ./scripts/build-linux.sh; fi"
