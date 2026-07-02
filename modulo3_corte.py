import json
import logging
import re
from pathlib import Path

from PIL import Image

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

from moviepy.editor import CompositeVideoClip, ImageClip, VideoFileClip


log = logging.getLogger("Modulo3")

LARGURA_ALVO = 1080
ALTURA_ALVO = 1920
ASPECTO_ALVO = LARGURA_ALVO / ALTURA_ALVO
OPACIDADE_WATERMARK = 0.40


def nome_seguro(texto: str, fallback: str) -> str:
    texto = texto or fallback
    texto = re.sub(r"[^\w\s-]", "", texto, flags=re.UNICODE)
    texto = re.sub(r"\s+", "_", texto.strip())
    return (texto or fallback)[:70]


def cortar_para_vertical(clip):
    aspecto_origem = clip.w / clip.h
    if aspecto_origem > ASPECTO_ALVO:
        largura_crop = int(clip.h * ASPECTO_ALVO)
        x1 = max(0, int((clip.w - largura_crop) / 2))
        clip_cropado = clip.crop(x1=x1, y1=0, x2=x1 + largura_crop, y2=clip.h)
        log.info("[Crop] Horizontal/largo: %sx%s -> crop x=%s w=%s.", clip.w, clip.h, x1, largura_crop)
    else:
        altura_crop = int(clip.w / ASPECTO_ALVO)
        y1 = max(0, int((clip.h - altura_crop) / 2))
        clip_cropado = clip.crop(x1=0, y1=y1, x2=clip.w, y2=y1 + altura_crop)
        log.info("[Crop] Vertical/estreito: %sx%s -> crop y=%s h=%s.", clip.w, clip.h, y1, altura_crop)
    return clip_cropado.resize((LARGURA_ALVO, ALTURA_ALVO))


def criar_clip_watermark(caminho_watermark: Path, duracao: float):
    if not caminho_watermark or not Path(caminho_watermark).exists():
        log.warning("[Watermark] Arquivo nao encontrado. Pulando marca d'agua.")
        return None

    clip = None
    try:
        clip = ImageClip(str(caminho_watermark))
        if clip.w > 400:
            clip = clip.resize(width=400)
        return (
            clip.set_duration(duracao)
            .set_opacity(OPACIDADE_WATERMARK)
            .set_position(("center", 80))
        )
    except Exception as exc:
        if clip is not None:
            clip.close()
        log.warning("[Watermark] Falha ao carregar watermark: %s", exc)
        return None


def processar_corte(
    caminho_video_original: Path,
    corte: dict,
    pasta_saida: Path,
    caminho_watermark: Path | None,
    indice: int,
) -> Path | None:
    start = float(corte["start_time"])
    end = float(corte["end_time"])
    titulo_safe = nome_seguro(corte.get("titulo_youtube", ""), f"corte_{indice:02d}")
    caminho_temp = pasta_saida / f"corte_{indice:02d}_{titulo_safe}_sem_audio.mp4"

    video = None
    subclip = None
    clip_vertical = None
    watermark_clip = None
    clip_final = None

    try:
        log.info("[Corte %s] Renderizando %.1fs -> %.1fs | %s", indice, start, end, titulo_safe)
        video = VideoFileClip(str(caminho_video_original), audio=False)
        fim_real = min(end, float(video.duration))
        if fim_real <= start:
            log.warning("[Corte %s] Intervalo invalido apos limitar pela duracao do video.", indice)
            return None

        subclip = video.subclip(start, fim_real)
        clip_vertical = cortar_para_vertical(subclip)
        watermark_clip = criar_clip_watermark(caminho_watermark, clip_vertical.duration)

        if watermark_clip is not None:
            clip_final = CompositeVideoClip([clip_vertical, watermark_clip], size=(LARGURA_ALVO, ALTURA_ALVO))
            clip_final = clip_final.set_duration(clip_vertical.duration).set_fps(clip_vertical.fps)
        else:
            clip_final = clip_vertical

        clip_final.write_videofile(
            str(caminho_temp),
            codec="libx264",
            audio=False,
            fps=clip_vertical.fps,
            preset="fast",
            ffmpeg_params=["-threads", "0", "-crf", "23", "-pix_fmt", "yuv420p"],
            logger=None,
        )
        log.info("[Corte %s] Video temporario gerado: %s", indice, caminho_temp)
        return caminho_temp

    except Exception as exc:
        log.error("[Corte %s] Erro ao processar: %s", indice, exc, exc_info=True)
        return None
    finally:
        for objeto in [clip_final, watermark_clip, clip_vertical, subclip, video]:
            if objeto is not None:
                try:
                    objeto.close()
                except Exception:
                    pass


def processar_todos_os_cortes(
    caminho_video: Path,
    caminho_cortes_json: Path,
    pasta_saida: Path,
    caminho_watermark: Path | None = None,
    watermark: Path | None = None,
) -> list[dict]:
    caminho_video = Path(caminho_video)
    caminho_cortes_json = Path(caminho_cortes_json)
    pasta_saida = Path(pasta_saida)
    caminho_watermark = caminho_watermark or watermark

    log.info("[Modulo 3] Processando cortes: %s", caminho_cortes_json)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    if not caminho_video.exists():
        log.error("[Modulo 3] Video original nao encontrado: %s", caminho_video)
        return []

    with open(caminho_cortes_json, encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    cortes = dados.get("cortes", [])
    resultados: list[dict] = []
    for indice, corte in enumerate(cortes, 1):
        caminho_temp = processar_corte(
            caminho_video_original=caminho_video,
            corte=corte,
            pasta_saida=pasta_saida,
            caminho_watermark=caminho_watermark,
            indice=indice,
        )
        if caminho_temp:
            info = dict(corte)
            info.update(
                {
                    "indice": indice,
                    "caminho_temp": str(caminho_temp),
                    "caminho_video_original": str(caminho_video),
                }
            )
            resultados.append(info)

    log.info("[Modulo 3] %s/%s cortes renderizados.", len(resultados), len(cortes))
    return resultados


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s")
    if len(sys.argv) < 3:
        print("Uso: python modulo3_corte.py <video.mp4> <cortes_sugeridos.json>")
        raise SystemExit(1)
    processar_todos_os_cortes(
        caminho_video=Path(sys.argv[1]),
        caminho_cortes_json=Path(sys.argv[2]),
        pasta_saida=Path(sys.argv[2]).parent / "cortes_temp",
        caminho_watermark=Path("assets/watermark.png"),
    )
