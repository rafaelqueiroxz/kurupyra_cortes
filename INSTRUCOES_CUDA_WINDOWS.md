# Kurupyra Cortes - Windows/NVIDIA

Use o instalador:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_windows_cuda.ps1
```

O ponto critico e instalar o PyTorch CUDA antes do `requirements.txt`:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Depois valide:

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'SEM CUDA')"
```

Se `torch.cuda.is_available()` retornar `False` no Python 3.14, use Python 3.12:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```
