param(
    [string]$BuildDir = "build-openvino-cpu-relwithdebinfo",
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Config = "RelWithDebInfo",
    [string]$SetupVars = "C:\\Users\\NECPC-USER\\Downloads\\openvino_genai_windows_2025.4.0.0_x86_64\\openvino_genai_windows_2025.4.0.0_x86_64\\setupvars.ps1"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$Env:PREV_PATH=$env:PATH

if ($SetupVars) {
    if (Test-Path $SetupVars) {
        & $SetupVars | Out-Host
    } else {
        throw "OpenVINO setupvars.ps1 not found: $SetupVars"
    }
} elseif (-not $env:OpenVINO_DIR) {
    Write-Warning "OpenVINO_DIR is not set. Run setupvars.ps1 or pass -SetupVars."
}

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
        -DGGML_OPENVINO=ON `
        -DGGML_CPU_REPACK=OFF `
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
        -DGGML_OPENVINO_DEVICE="CPU" `
        -DGGML_OPENVINO_DEBUG_OUTPUT=1 `
        -DGGML_OPENVINO_DUMP_IR=1 `
        -DLLAMA_CURL=OFF

    if ($LASTEXITCODE -ne 0) {
        throw "cmake configure failed (exit=$LASTEXITCODE)"
    }

    cmake --build $BuildDir --config $Config -- /m
    if ($LASTEXITCODE -ne 0) {
        throw "cmake build failed (exit=$LASTEXITCODE)"
    }
    Write-Host "`nOpenVINO CPU build complete: $BuildDir ($Config)" -ForegroundColor Green

    $OutputFile = "tmp/plamo-2-translate-cli_openvino_cpu.txt"

    # .\build-openvino-cpu-relwithdebinfo\bin\RelWithDebInfo\llama-eval-callback.exe
    & "$BuildDir\\bin\\$Config\\llama-cli.exe" `
    -m .\plamo-2-translate_Q4_0.gguf `
    -c 2048 `
    -n 8 `
    -lv 3 `
    -p "<|plamo:bos|><|plamo:op|>dataset\ntranslation\n<|plamo:op|>input\nThis is a white pen.\n<|plamo:op|>output\n" `
    2>&1 | Out-File -FilePath $OutputFile -Encoding UTF8
} finally {
    $ErrorActionPreference = $nativeEap
}

$Env:PATH=$Env:PREV_PATH