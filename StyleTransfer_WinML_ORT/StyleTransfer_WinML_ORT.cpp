// StyleTransfer_WinML_ORT.cpp : This file contains the function which dynamically setect ORT EP or DML basesd on platform
#include "main.h"
#include <MemoryBuffer.h>

using namespace winrt;
using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Storage;
using namespace std;
using namespace cv;

// Global variables
hstring modelPath= L"C:\\temp\\style.onnx";
string deviceName = "default";
hstring imagePath;
LearningModel model = nullptr;
LearningModelDeviceKind deviceKind = LearningModelDeviceKind::DirectX;
LearningModelSession session = nullptr;
LearningModelBinding binding = nullptr;
VideoFrame imageFrame = nullptr;
string labelsFilePath;
std::string settings_str;


int main()
{
    /*Read Image -- Common between WinML/ORT
    * WinML prefers code in Bitmap but ORT needs buffer access
    * Below is example of converting bitmap code to tensor buffer 
    */
    VideoFrame inputImage = nullptr;
    hstring imagePath = L"C:\\temp\\test.png";
    hstring input_node_names = L"input1";
    hstring output_node_names = L"output1";
    StorageFile file = StorageFile::GetFileFromPathAsync(imagePath).get();
    auto stream = file.OpenAsync(FileAccessMode::Read).get();    
    BitmapDecoder decoder = BitmapDecoder::CreateAsync(stream).get();
    SoftwareBitmap softwareBitmap = decoder.GetSoftwareBitmapAsync().get();
    inputImage = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
    BYTE* pData = nullptr;
    UINT32 size = 0;
    float* pCPUTensor;
    uint32_t uCapacity;
    BitmapBuffer spBitmapBuffer(softwareBitmap.LockBuffer(BitmapBufferAccessMode::Read));
    winrt::Windows::Foundation::IMemoryBufferReference reference = spBitmapBuffer.CreateReference();
    auto spByteAccess = reference.as<::Windows::Foundation::IMemoryBufferByteAccess>();
    spByteAccess->GetBuffer(&pData, &size);
    uint32_t height = softwareBitmap.PixelHeight();
    uint32_t width = softwareBitmap.PixelWidth();
    uint32_t channels = 3;
    std::vector<int64_t> shape = { 1, channels, height , width };

    size_t input_tensor_size = width * height * channels;
    TensorFloat tf = TensorFloat::Create(shape);
    com_ptr<ITensorNative> itn = tf.as<ITensorNative>();
    itn->GetBuffer(reinterpret_cast<BYTE**>(&pCPUTensor), &uCapacity);
    std::vector<const char*> output_ptr_names(output_node_names.begin(), output_node_names.end());
    std::vector<const char*> input_ptr_names(input_node_names.begin(), input_node_names.end());
 
    for (UINT32 i = 0; i < size; i += 4)
    {
        // suppose the model expects BGR image.
        // index 0 is B, 1 is G, 2 is R, 3 is alpha(dropped).
        UINT32 pixelInd = i / 4;
        pCPUTensor[pixelInd] = (float)pData[i + 2];
        pCPUTensor[(height * width) + pixelInd] = (float)pData[i + 1];
        pCPUTensor[(height * width * 2) + pixelInd] = (float)pData[i];
    }
    /*
    * WinML - Input Image
    * ORT - CPUTensor
    */
    printf("image loaded");
    //Load model for WinML
    model = LearningModel::LoadFromFilePath(modelPath);
    //WinML code
    session = LearningModelSession{ model, LearningModelDevice(deviceKind) };
    binding = LearningModelBinding{ session };
    binding.Bind(input_node_names, ImageFeatureValue::CreateFromVideoFrame(inputImage));
    binding.Bind(L"output1", TensorFloat::Create(shape));
    printf("Model loaded");
    auto results = session.Evaluate(binding, L"RunId");
     
    auto resultTensor = results.Outputs().Lookup(L"output1").as<TensorFloat>();
    auto resultVector = resultTensor.GetAsVectorView();
    //ORT code path
    Ort::Env env;
    Ort::SessionOptions session_options;
    //Add code for ORT EP Nuget using https://github.com/microsoft/onnxruntime/blob/master/BUILD.md
    //The DNNL, TensorRT, and OpenVINO providers are built as shared libraries vs being statically linked 
    //into the main onnxruntime. This enables them to be loaded only when needed, and if the dependent 
    //libraries of the provider are not installed onnxruntime will still run fine, 
    //it just will not be able to use that provider
    Ort::Session session(env, modelPath.c_str(), Ort::SessionOptions{ nullptr });

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, pCPUTensor, input_tensor_size, shape.data(), 4);
    assert(input_tensor.IsTensor());
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_ptr_names.data(), &input_tensor, 1, output_ptr_names.data(), 1);
    std::cout << "Code completed!\n";

    //Output tensor and results can be parsed with same datatype
}

