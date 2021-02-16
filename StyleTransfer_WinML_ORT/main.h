#pragma once

#define NOMINMAX
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING 1 // The C++ Standard doesn't provide equivalent non-deprecated functionality yet.

#include <iostream>
#include <vcruntime.h>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include "Windows.AI.MachineLearning.Native.h"
 
#include <onnxruntime_cxx_api.h>
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.h>
#include <string>
#include <fstream>

#include <Windows.h>
