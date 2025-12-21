C:\Users\NECPC-USER\Downloads\openvino_genai_windows_2025.4.0.0_x86_64\openvino_genai_windows_2025.4.0.0_x86_64\setupvars.ps1

cmake --build build-openvino-relwithdebinfo --config RelWithDebInfo -- /m

$ovBase="C:\Users\NECPC-USER\Downloads\openvino_genai_windows_2025.4.0.0_x86_64\openvino_genai_windows_2025.4.0.0_x86_64\runtime"
$env:Path="$ovBase\bin\intel64\Release;$ovBase\3rdparty\tbb\bin;$(Resolve-Path .\build-openvino-relwithdebinfo\bin\RelWithDebInfo);$env:Path"

$env:GGML_OPENVINO_SSM_ONLY="1"

.\build-openvino-relwithdebinfo\bin\RelWithDebInfo\llama-cli.exe `
-m .\plamo2-1b-alignment-id1397-dpo-auto.gguf `
-c 2048 `
-n 1024 `
--no-warmup `
-fit on `
-v `
--log-file run-openvino-relwithdebinfo_test.log `
-p "<|plamo:bos|><|plamo:op|>user\nこんにちは\n<|plamo:op|>assistant\n" `
--single-turn
