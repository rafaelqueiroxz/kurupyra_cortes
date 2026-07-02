import json
import logging
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

os.environ.setdefault("IMAGEMAGICK_BINARY", "magick")

from moviepy.editor import VideoClip, VideoFileClip


log = logging.getLogger("Modulo4")

LARGURA_VIDEO = 1080
ALTURA_VIDEO = 1920
COR_ATIVA = (255, 215, 0)
COR_NORMAL = (255, 255, 255)
COR_BORDA = (0, 0, 0)
TAMANHO_FONTE = 82
TAMANHO_MINIMO = 46
ESPESSURA_BORDA = 7
MAX_CHARS_LINHA = 18
MAX_LINHAS = 2
POSICAO_Y = 0.58
ESPACO_PALAVRAS = 14
ESPACO_LINHAS = 12

FONTES_PREFERIDAS = [
    "Impact.ttf",
    "impact.ttf",
    "Arial Black.ttf",
    "ariblk.ttf",
    "Montserrat-Black.ttf",
    "MontserratBlack.ttf",
    "arialbd.ttf",
]

_CACHE_FONTES: dict[tuple[str, int], ImageFont.ImageFont] = {}


def encontrar_fonte(tamanho: int) -> ImageFont.ImageFont:
    pastas = [
        Path(__file__).parent / "assets" / "fonts",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
        Path("."),
    ]
    for nome in FONTES_PREFERIDAS:
        for pasta in pastas:
            caminho = pasta / nome
            chave = (str(caminho), tamanho)
            if chave in _CACHE_FONTES:
                return _CACHE_FONTES[chave]
            if caminho.exists():
                try:
                    fonte = ImageFont.truetype(str(caminho), tamanho)
                    _CACHE_FONTES[chave] = fonte
                    return fonte
                except Exception:
                    continue

    log.warning("[Fonte] Impact/Arial Black nao encontrada. Usando fallback PIL.")
    try:
        return ImageFont.load_default(size=tamanho)
    except TypeError:
        return ImageFont.load_default()


def texto_limpo(palavra: str) -> str:
    return re.sub(r"\s+", " ", str(palavra)).strip()


def quebrar_em_linhas(palavras: list[dict]) -> list[list[dict]]:
    linhas: list[list[dict]] = []
    linha_atual: list[dict] = []

    for palavra in palavras:
        texto = texto_limpo(palavra["palavra"])
        candidato = " ".join([texto_limpo(p["palavra"]) for p in linha_atual + [palavra]])
        if linha_atual and len(candidato) > MAX_CHARS_LINHA and len(linhas) < MAX_LINHAS - 1:
            linhas.append(linha_atual)
            linha_atual = [palavra]
        else:
            linha_atual.append(palavra)

    if linha_atual:
        linhas.append(linha_atual)
    return linhas[:MAX_LINHAS]


def agrupar_legendas(palavras: list[dict]) -> list[dict]:
    legendas: list[dict] = []
    grupo: list[dict] = []

    for palavra in palavras:
        if not texto_limpo(palavra.get("palavra", "")):
            continue
        candidato = grupo + [palavra]
        linhas = quebrar_em_linhas(candidato)
        total_linhas = len(linhas)
        maior_linha = max((len(" ".join(texto_limpo(p["palavra"]) for p in linha)) for linha in linhas), default=0)
        duracao = float(candidato[-1]["fim"]) - float(candidato[0]["inicio"])

        if grupo and (total_linhas > MAX_LINHAS or maior_linha > MAX_CHARS_LINHA + 8 or duracao > 2.8):
            legendas.append({"palavras": grupo, "inicio": grupo[0]["inicio"], "fim": grupo[-1]["fim"]})
            grupo = [palavra]
        else:
            grupo = candidato

    if grupo:
        legendas.append({"palavras": grupo, "inicio": grupo[0]["inicio"], "fim": grupo[-1]["fim"]})

    return legendas


def medir(draw: ImageDraw.ImageDraw, texto: str, fonte: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), texto, font=fonte, stroke_width=ESPESSURA_BORDA)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def escolher_fonte_para_linhas(draw: ImageDraw.ImageDraw, linhas: list[list[dict]], largura: int) -> ImageFont.ImageFont:
    tamanho = TAMANHO_FONTE
    while tamanho >= TAMANHO_MINIMO:
        fonte = encontrar_fonte(tamanho)
        cabe = True
        for linha in linhas:
            texto = " ".join(texto_limpo(p["palavra"]) for p in linha)
            largura_texto, _ = medir(draw, texto, fonte)
            palavra_mais_longa = max((medir(draw, texto_limpo(p["palavra"]), fonte)[0] for p in linha), default=0)
            if largura_texto > largura * 0.88 or palavra_mais_longa > largura * 0.82:
                cabe = False
                break
        if cabe:
            return fonte
        tamanho -= 4
    return encontrar_fonte(TAMANHO_MINIMO)


def palavra_ativa(legenda: dict, tempo_atual: float) -> int:
    palavras = legenda["palavras"]
    for indice, palavra in enumerate(palavras):
        if float(palavra["inicio"]) <= tempo_atual <= float(palavra["fim"]):
            return indice
    if not palavras:
        return -1
    distancias = [abs(float(p["inicio"]) - tempo_atual) for p in palavras]
    return int(distancias.index(min(distancias)))


def renderizar_overlay_legenda(legenda: dict, tempo_atual: float, largura: int, altura: int) -> np.ndarray:
    img = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    linhas = quebrar_em_linhas(legenda["palavras"])
    fonte = escolher_fonte_para_linhas(draw, linhas, largura)
    ativo_global = palavra_ativa(legenda, tempo_atual)

    alturas_linhas = []
    larguras_linhas = []
    for linha in linhas:
        medidas = [medir(draw, texto_limpo(p["palavra"]), fonte) for p in linha]
        larguras_linhas.append(sum(w for w, _ in medidas) + ESPACO_PALAVRAS * max(0, len(medidas) - 1))
        alturas_linhas.append(max((h for _, h in medidas), default=0))

    altura_total = sum(alturas_linhas) + ESPACO_LINHAS * max(0, len(linhas) - 1)
    y = int(altura * POSICAO_Y - altura_total / 2)
    indice_global = 0

    for indice_linha, linha in enumerate(linhas):
        x = int((largura - larguras_linhas[indice_linha]) / 2)
        for palavra in linha:
            texto = texto_limpo(palavra["palavra"]).upper()
            cor = COR_ATIVA if indice_global == ativo_global else COR_NORMAL
            draw.text(
                (x, y),
                texto,
                font=fonte,
                fill=(*cor, 255),
                stroke_width=ESPESSURA_BORDA,
                stroke_fill=(*COR_BORDA, 255),
            )
            w, _ = medir(draw, texto, fonte)
            x += w + ESPACO_PALAVRAS
            indice_global += 1
        y += alturas_linhas[indice_linha] + ESPACO_LINHAS

    return np.array(img)


def adicionar_legendas_ao_video(caminho_video_temp: Path, palavras_corte: list[dict], caminho_saida: Path) -> bool:
    legendas = agrupar_legendas(palavras_corte)
    log.info("[Legendas] %s palavras -> %s blocos de legenda.", len(palavras_corte), len(legendas))

    video = None
    clip_com_legenda = None
    try:
        video = VideoFileClip(str(caminho_video_temp), audio=False)
        duracao = video.duration
        fps = video.fps

        def make_frame(t):
            frame = video.get_frame(t)
            legenda_ativa = None
            for legenda in legendas:
                if float(legenda["inicio"]) <= t <= float(legenda["fim"]):
                    legenda_ativa = legenda
                    break
            if legenda_ativa is None:
                return frame

            overlay = renderizar_overlay_legenda(legenda_ativa, t, LARGURA_VIDEO, ALTURA_VIDEO)
            alpha = overlay[:, :, 3:4].astype(float) / 255.0
            composto = frame.astype(float) * (1 - alpha) + overlay[:, :, :3].astype(float) * alpha
            return composto.astype(np.uint8)

        clip_com_legenda = VideoClip(make_frame, duration=duracao).set_fps(fps)
        clip_com_legenda.write_videofile(
            str(caminho_saida),
            codec="libx264",
            audio=False,
            fps=fps,
            preset="fast",
            ffmpeg_params=["-threads", "0", "-crf", "23", "-pix_fmt", "yuv420p"],
            logger=None,
        )
        log.info("[Legendas] Video legendado salvo: %s", caminho_saida)
        return True
    except Exception as exc:
        log.error("[Legendas] Falha ao renderizar legendas: %s", exc, exc_info=True)
        return False
    finally:
        for objeto in [clip_com_legenda, video]:
            if objeto is not None:
                try:
                    objeto.close()
                except Exception:
                    pass


def executar_ffmpeg(cmd: list[str], contexto: str) -> bool:
    try:
        resultado = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        log.error("[%s] FFmpeg nao encontrado no PATH.", contexto)
        return False

    if resultado.returncode != 0:
        log.error("[%s] FFmpeg falhou: %s", contexto, resultado.stderr.strip())
        return False
    return True


def injetar_audio_ffmpeg(
    caminho_video_sem_audio: Path,
    caminho_video_original: Path,
    start_time: float,
    end_time: float,
    caminho_final: Path,
) -> bool:
    caminho_audio_temp = caminho_video_sem_audio.parent / f"{caminho_video_sem_audio.stem}_audio_temp.aac"
    log.info("[Audio] Extraindo e sincronizando audio %.1fs -> %.1fs.", start_time, end_time)

    try:
        cmd_extrair = [
            "ffmpeg",
            "-y",
            "-threads",
            "0",
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-i",
            str(caminho_video_original),
            "-vn",
            "-acodec",
            "aac",
            "-b:a",
            "192k",
            "-loglevel",
            "error",
            str(caminho_audio_temp),
        ]
        if not executar_ffmpeg(cmd_extrair, "Audio"):
            return False

        cmd_combinar = [
            "ffmpeg",
            "-y",
            "-threads",
            "0",
            "-i",
            str(caminho_video_sem_audio),
            "-i",
            str(caminho_audio_temp),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            "-loglevel",
            "error",
            str(caminho_final),
        ]
        return executar_ffmpeg(cmd_combinar, "Mux")
    finally:
        if caminho_audio_temp.exists():
            try:
                caminho_audio_temp.unlink()
            except OSError:
                pass


def filtrar_palavras_do_corte(todas_palavras: list[dict], start: float, end: float) -> list[dict]:
    palavras_corte: list[dict] = []
    for palavra in todas_palavras:
        inicio = float(palavra["inicio"])
        fim = float(palavra["fim"])
        if inicio >= start and fim <= end:
            palavras_corte.append(
                {
                    "palavra": palavra["palavra"],
                    "inicio": round(inicio - start, 3),
                    "fim": round(fim - start, 3),
                }
            )
    return palavras_corte


def nome_seguro(texto: str, fallback: str) -> str:
    texto = texto or fallback
    texto = re.sub(r"[^\w\s-]", "", texto, flags=re.UNICODE)
    texto = re.sub(r"\s+", "_", texto.strip())
    return (texto or fallback)[:70]


def processar_legenda_e_audio(info_corte: dict, todas_palavras: list[dict], pasta_cortes_prontos: Path) -> Path | None:
    caminho_temp = Path(info_corte["caminho_temp"])
    caminho_original = Path(info_corte["caminho_video_original"])
    start = float(info_corte["start_time"])
    end = float(info_corte["end_time"])
    titulo_safe = nome_seguro(info_corte.get("titulo_youtube", ""), f"corte_{info_corte['indice']:02d}")

    pasta_cortes_prontos.mkdir(parents=True, exist_ok=True)
    caminho_com_legenda = caminho_temp.parent / f"{titulo_safe}_legendas.mp4"
    caminho_final = pasta_cortes_prontos / f"{titulo_safe}_FINAL.mp4"

    palavras_corte = filtrar_palavras_do_corte(todas_palavras, start, end)
    if palavras_corte:
        sucesso_legenda = adicionar_legendas_ao_video(caminho_temp, palavras_corte, caminho_com_legenda)
        video_para_audio = caminho_com_legenda if sucesso_legenda else caminho_temp
    else:
        log.warning("[Modulo 4] Nenhuma palavra no intervalo %.1f-%.1f. Gerando sem legenda.", start, end)
        video_para_audio = caminho_temp

    sucesso_audio = injetar_audio_ffmpeg(video_para_audio, caminho_original, start, end, caminho_final)

    for temporario in [caminho_temp, caminho_com_legenda]:
        if temporario.exists():
            try:
                temporario.unlink()
            except OSError as exc:
                log.warning("[Cleanup] Nao foi possivel remover %s: %s", temporario, exc)

    if sucesso_audio:
        log.info("[Modulo 4] Corte final pronto: %s", caminho_final)
        return caminho_final
    return None


def processar_todos_com_legendas(
    lista_cortes_info: list[dict],
    caminho_transcricao_json: Path,
    pasta_cortes_prontos: Path,
) -> list[dict]:
    with open(caminho_transcricao_json, encoding="utf-8") as arquivo:
        transcricao = json.load(arquivo)
    todas_palavras = transcricao.get("palavras", [])

    resultados: list[dict] = []
    for info in lista_cortes_info:
        caminho_final = processar_legenda_e_audio(info, todas_palavras, Path(pasta_cortes_prontos))
        if caminho_final:
            info["caminho_final"] = str(caminho_final)
            resultados.append(info)

    log.info("[Modulo 4] %s/%s cortes finalizados.", len(resultados), len(lista_cortes_info))
    return resultados


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s")
    print("Modulo 4 deve ser chamado pelo main.py.")
