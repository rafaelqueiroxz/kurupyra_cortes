$ErrorActionPreference = "Stop"

Write-Host "Kurupyra Cortes - instalacao Windows/NVIDIA CUDA"

if (-not (Test-Path ".\venv")) {
    py -3.14 -m venv .\venv
}

.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

Write-Host "Instalando PyTorch CUDA pelo indice oficial cu128..."
.\venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host "Instalando demais dependencias..."
.\venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Validando CUDA..."
@'
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA disponivel:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit(
        "CUDA nao ficou disponivel. No Windows/Python 3.14, instale Python 3.12, recrie a venv "
        "e rode novamente este script. Nao rode o Whisper em CPU para podcasts longos."
    )
'@ | .\venv\Scripts\python.exe -

Write-Host "Instalacao concluida. Execute: .\venv\Scripts\python.exe main.py"
