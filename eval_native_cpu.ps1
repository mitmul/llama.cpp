param(
    [string]$BuildDir = "build-cpu-relwithdebinfo",
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Config = "RelWithDebInfo"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$Env:PREV_PATH=$env:PATH

$nativeEap = $ErrorActionPreference
try {
    # Native commands can write useful status to stderr; don't treat that as a terminating PowerShell error.
    $ErrorActionPreference = "Continue"

    cmake -S . -B $BuildDir `
        -G "Visual Studio 18 2026" `
        -A x64 `
        -DGGML_BACKEND_DL=OFF `
        -DGGML_CPU=ON `
        -DGGML_BLAS=OFF `
        -DGGML_OPENVINO=OFF `
        -DGGML_CUDA=OFF `
        -DGGML_HIP=OFF `
        -DGGML_VULKAN=OFF `
        -DGGML_METAL=OFF `
        -DGGML_SYCL=OFF `
        -DGGML_MUSA=OFF `
        -DGGML_WEBGPU=OFF `
        -DGGML_OPENCL=OFF `
        -DGGML_CANN=OFF `
        -DGGML_ZDNN=OFF `
        -DGGML_ZENDNN=OFF `
        -DGGML_RPC=OFF `
        -DGGML_HEXAGON=OFF `
        -DGGML_NATIVE=ON `
        -DGGML_MAX_DIMS=128 `
        -DLLAMA_CURL=OFF 

    if ($LASTEXITCODE -ne 0) {
        throw "cmake configure failed (exit=$LASTEXITCODE)"
    }

    cmake --build $BuildDir --config $Config -- /m
    if ($LASTEXITCODE -ne 0) {
        throw "cmake build failed (exit=$LASTEXITCODE)"
    }
    Write-Host "`nCPU-only build complete: $BuildDir ($Config)" -ForegroundColor Green
    
    # .\build-cpu-relwithdebinfo\bin\RelWithDebInfo\llama-eval-callback.exe `
    & "$BuildDir\\bin\\$Config\\llama-eval-callback.exe" `
    -m .\plamo-2-translate_Q4_0.gguf `
    -c 2048 `
    -n 8 `
    -lv 1 `
    -p "<|plamo:bos|><|plamo:op|>dataset\ntranslation\n<|plamo:op|>input\nThis is a white pen.\n<|plamo:op|>output\n" `
    2>&1 > tmp/plamo-2-translate-eval-callback_native_cpu.txt
} finally {
    $ErrorActionPreference = $nativeEap
}

$Env:PATH=$Env:PREV_PATH