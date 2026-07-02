import json
import logging
import os
import time
from pathlib import Path

import ffmpeg
import torch
import whisper


log = logging.getLogger("Modulo1")

BASE_DIR = Path(__file__).parent
PASTA_ENTRADA = BASE_DIR / "podcasts_brutos"
PASTA_SAIDA = BASE_DIR / "output"

WHISPER_MODEL = os.getenv("KURUPYRA_WHISPER_MODEL", "medium")
WHISPER_LANGUAGE = os.getenv("KURUPYRA_WHISPER_LANGUAGE", "pt")
REQUIRE_CUDA = os.getenv("KURUPYRA_REQUIRE_CUDA", "1").lower() not in {"0", "false", "nao", "no"}
TORCH_CUDA_COMMAND = (
    "python -m pip install --upgrade --force-reinstall torch torchvision torchaudio "
    "--index-url https://download.pytorch.org/whl/cu128"
)

_MODELO_WHISPER = None
_MODELO_DEVICE = None


def validar_cuda() -> str:
    if torch.cuda.is_available():
        nome_gpu = torch.cuda.get_device_name(0)
        log.info("[CUDA] GPU detectada para Whisper: %s", nome_gpu)
        return "cuda"

    mensagem = (
        "CUDA indisponivel no PyTorch. Para evitar o gargalo de CPU, instale o PyTorch CUDA "
        f"com: {TORCH_CUDA_COMMAND}. Se continuar falso no Python 3.14, crie o ambiente com "
        "Python 3.12 e repita a instalacao CUDA."
    )
    if REQUIRE_CUDA:
        raise RuntimeError(mensagem)

    log.warning("[CUDA] %s Rodando em CPU porque KURUPYRA_REQUIRE_CUDA=0.", mensagem)
    return "cpu"


def carregar_modelo_whisper():
    global _MODELO_WHISPER, _MODELO_DEVICE

    device = validar_cuda()
    if _MODELO_WHISPER is not None and _MODELO_DEVICE == device:
        return _MODELO_WHISPER, _MODELO_DEVICE

    log.info("[Whisper] Carregando modelo '%s' em device='%s'.", WHISPER_MODEL, device)
    _MODELO_WHISPER = whisper.load_model(WHISPER_MODEL, device=device)
    _MODELO_DEVICE = device
    return _MODELO_WHISPER, _MODELO_DEVICE


def extrair_audio(caminho_video: Path, caminho_audio: Path) -> bool:
    log.info("[FFmpeg] Extraindo audio: %s", caminho_video.name)
    try:
        (
            ffmpeg.input(str(caminho_video))
            .output(
                str(caminho_audio),
                ac=1,
                ar=16000,
                acodec="pcm_s16le",
                threads=0,
                loglevel="error",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        tamanho_mb = caminho_audio.stat().st_size / 1_000_000
        log.info("[FFmpeg] Audio WAV pronto: %s (%.1f MB)", caminho_audio.name, tamanho_mb)
        return True
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        log.error("[FFmpeg] Falha ao extrair audio: %s", stderr)
        return False


def transcrever_audio(caminho_audio: Path, nome_base: str) -> dict | None:
    modelo, device = carregar_modelo_whisper()
    inicio = time.time()
    log.info("[Whisper] Transcrevendo com word_timestamps=True: %s", caminho_audio.name)

    try:
        resultado = modelo.transcribe(
            str(caminho_audio),
            language=WHISPER_LANGUAGE,
            word_timestamps=True,
            verbose=False,
            fp16=(device == "cuda"),
            condition_on_previous_text=False,
            temperature=0.0,
            beam_size=5,
            best_of=5,
        )
    except Exception as exc:
        log.error("[Whisper] Falha na transcricao: %s", exc, exc_info=True)
        return None

    palavras = []
    segmentos = []
    for segmento in resultado.get("segments", []):
        segmentos.append(
            {
                "inicio": round(float(segmento.get("start", 0.0)), 3),
                "fim": round(float(segmento.get("end", 0.0)), 3),
                "texto": str(segmento.get("text", "")).strip(),
            }
        )
        for palavra_info in segmento.get("words", []):
            texto = str(palavra_info.get("word", "")).strip()
            if not texto:
                continue
            palavras.append(
                {
                    "palavra": texto,
                    "inicio": round(float(palavra_info.get("start", 0.0)), 3),
                    "fim": round(float(palavra_info.get("end", 0.0)), 3),
                }
            )

    duracao = palavras[-1]["fim"] if palavras else 0.0
    dados = {
        "arquivo_origem": nome_base,
        "modelo_whisper": WHISPER_MODEL,
        "device": device,
        "idioma_detectado": resultado.get("language", WHISPER_LANGUAGE),
        "texto_completo": resultado.get("text", "").strip(),
        "total_palavras": len(palavras),
        "duracao_estimada_segundos": duracao,
        "tempo_transcricao_segundos": round(time.time() - inicio, 2),
        "segmentos": segmentos,
        "palavras": palavras,
    }

    log.info(
        "[Whisper] Concluido: %s palavras | %.1fs de video | %.1fs processando",
        len(palavras),
        duracao,
        dados["tempo_transcricao_segundos"],
    )
    return dados


def salvar_transcricao(dados: dict, caminho_json: Path) -> None:
    caminho_json.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_json, "w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2)
    log.info("[JSON] Transcricao salva: %s", caminho_json)


def processar_video(caminho_video: Path) -> Path | None:
    caminho_video = Path(caminho_video)
    nome_base = caminho_video.stem
    pasta_saida_video = PASTA_SAIDA / nome_base
    caminho_audio = pasta_saida_video / f"{nome_base}.wav"
    caminho_json = pasta_saida_video / "transcricao_completa.json"

    if caminho_json.exists():
        log.info("[Skip] Transcricao existente para '%s'.", nome_base)
        return caminho_json

    pasta_saida_video.mkdir(parents=True, exist_ok=True)
    if not caminho_video.exists():
        log.error("[Entrada] Video nao encontrado: %s", caminho_video)
        return None

    try:
        if not extrair_audio(caminho_video, caminho_audio):
            return None

        dados = transcrever_audio(caminho_audio, nome_base)
        if not dados:
            return None

        salvar_transcricao(dados, caminho_json)
        return caminho_json
    finally:
        if caminho_audio.exists():
            try:
                caminho_audio.unlink()
                log.info("[Cleanup] WAV temporario removido.")
            except OSError as exc:
                log.warning("[Cleanup] Nao foi possivel remover WAV: %s", exc)


def monitorar_pasta(intervalo_segundos: int = 30) -> None:
    PASTA_ENTRADA.mkdir(parents=True, exist_ok=True)
    log.info("[Monitor] Modulo 1 monitorando: %s", PASTA_ENTRADA)

    while True:
        videos = sorted(list(PASTA_ENTRADA.glob("*.mp4")) + list(PASTA_ENTRADA.glob("*.MP4")))
        if not videos:
            log.info("[Monitor] Nenhum video encontrado. Aguardando %ss.", intervalo_segundos)
        for video in videos:
            processar_video(video)
        time.sleep(intervalo_segundos)


def processar_video_direto(caminho_video: str | Path) -> Path | None:
    return processar_video(Path(caminho_video))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s")
    monitorar_pasta()
