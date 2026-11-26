# The modelmap Mac app. Needs Xcode, xcodegen (brew install xcodegen), uv and
# Poetry.
#
#   make run      stage Python, build, and open the app
#   make test     the Swift unit tests
#   make clean    remove everything these targets write

# pipefail, so a test run piped through rg still fails the target.
SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c

VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' ../pyproject.toml)
BUILD   := $(shell git rev-list --count HEAD)
APP     := build/Build/Products/Release/modelmap.app
XCODE   := xcodebuild -project ModelMap.xcodeproj -scheme ModelMap -derivedDataPath build \
           -destination 'platform=macOS,arch=arm64' \
           MARKETING_VERSION=$(VERSION) CURRENT_PROJECT_VERSION=$(BUILD)

.PHONY: python project build run test clean

python:
	scripts/stage_python.sh

project:
	xcodegen generate --quiet

build: python project
	$(XCODE) -configuration Release -quiet build
	@echo "Built $(APP) ($(VERSION))"

run: build
	open $(APP)

test: python project
	$(XCODE) -configuration Debug test 2>&1 | rg 'Test run with|TEST (SUCCEEDED|FAILED)|error:|failed'


clean:
	rm -rf build Python ModelMap.xcodeproj
