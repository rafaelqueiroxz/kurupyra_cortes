import logging
import shutil
import sys
from pathlib import Path

from modulo1_transcricao import processar_video_direto
from modulo2_diretor import analisar_transcricao
from modulo3_corte import processar_todos_os_cortes
from modulo4_legendas import processar_todos_com_legendas


BASE_DIR = Path(__file__).parent
PASTA_BRUTOS = BASE_DIR / "podcasts_brutos"
PASTA_PROCESSADOS = PASTA_BRUTOS / "_processados"
PASTA_OUTPUT = BASE_DIR / "output"
PASTA_CORTES_TEMP = BASE_DIR / "cortes_temp"
PASTA_CORTES_FINAL = BASE_DIR / "cortes_prontos"
PASTA_ASSETS = BASE_DIR / "assets"
CAMINHO_WATERMARK = PASTA_ASSETS / "watermark.png"


def configurar_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s",
        handlers=[
            logging.FileHandler(BASE_DIR / "main_pipeline.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


log = logging.getLogger("MAIN")


def garantir_diretorios() -> None:
    for pasta in [
        PASTA_BRUTOS,
        PASTA_PROCESSADOS,
        PASTA_OUTPUT,
        PASTA_CORTES_TEMP,
        PASTA_CORTES_FINAL,
        PASTA_ASSETS,
    ]:
        pasta.mkdir(parents=True, exist_ok=True)

    if not CAMINHO_WATERMARK.exists():
        log.warning("[Assets] Watermark nao encontrada. Os cortes sairao sem marca d'agua.")


def arquivar_video_processado(caminho_video: Path) -> None:
    if not caminho_video.exists():
        return

    destino = PASTA_PROCESSADOS / caminho_video.name
    contador = 1
    while destino.exists():
        destino = PASTA_PROCESSADOS / f"{caminho_video.stem}_{contador}{caminho_video.suffix}"
        contador += 1

    shutil.move(str(caminho_video), str(destino))
    log.info("[Arquivo] Video bruto movido para: %s", destino)


def processar_video_completo(caminho_video: Path) -> int:
    log.info("=" * 72)
    log.info("[Pipeline] Iniciando: %s", caminho_video.name)
    log.info("=" * 72)

    try:
        # Módulo 1 — Transcrição
        caminho_json = processar_video_direto(caminho_video)
        if not caminho_json:
            log.error("[Pipeline] Modulo 1 falhou. Pulando video.")
            return 0

        # Módulo 2 — Seleção Multi-Agent (Two-Pass)
        caminho_cortes_json = analisar_transcricao(caminho_json)
        if not caminho_cortes_json:
            log.error("[Pipeline] Modulo 2 falhou. Pulando video.")
            return 0

        # Módulo 3 — Corte e composição visual
        lista_cortes = processar_todos_os_cortes(
            caminho_video=caminho_video,
            caminho_cortes_json=caminho_cortes_json,
            pasta_saida=PASTA_CORTES_TEMP,
            caminho_watermark=CAMINHO_WATERMARK,
        )
        if not lista_cortes:
            log.warning("[Pipeline] Nenhum corte bruto foi gerado.")
            return 0

        # Módulo 4 — Legendagem; resultado final salvo em cortes_prontos
        lista_finais = processar_todos_com_legendas(
            lista_cortes_info=lista_cortes,
            caminho_transcricao_json=caminho_json,
            pasta_cortes_prontos=PASTA_CORTES_FINAL,
        )
        if not lista_finais:
            log.warning("[Pipeline] Nenhum corte finalizado com legenda/audio.")
            return 0

        log.info(
            "[Pipeline] Concluido: %s cortes finais salvos em '%s'.",
            len(lista_finais),
            PASTA_CORTES_FINAL,
        )
        arquivar_video_processado(caminho_video)
        return len(lista_finais)

    except Exception as exc:
        log.error("[Pipeline] Erro critico em %s: %s", caminho_video.name, exc, exc_info=True)
        return 0


def listar_videos_pendentes() -> list[Path]:
    extensoes = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.mkv", "*.MKV")
    videos: list[Path] = []
    for padrao in extensoes:
        videos.extend(PASTA_BRUTOS.glob(padrao))
    return sorted(v for v in videos if v.is_file())


def executar_pipeline_em_lote() -> None:
    configurar_logging()
    garantir_diretorios()

    log.info("=" * 72)
    log.info("KURUPYRA CORTES - pipeline em lote iniciado")
    log.info("=" * 72)

    # Lista capturada UMA ÚNICA VEZ. Nenhum glob é chamado novamente durante o loop.
    videos = listar_videos_pendentes()
    if not videos:
        log.info("[Batch] Nenhum video encontrado em '%s'. Encerrando.", PASTA_BRUTOS)
        print("Todos os videos processados.")
        sys.exit(0)

    log.info("[Batch] %s video(s) encontrado(s) para processar.", len(videos))
    total_cortes = 0

    for video in videos:
        # Guard obrigatório: o arquivo pode ter sido movido por uma iteração anterior.
        if not video.exists():
            log.warning("[Batch] Arquivo ja nao existe (foi movido?). Pulando: %s", video.name)
            continue
        total_cortes += processar_video_completo(video)

    log.info(
        "[Batch] Processamento concluido. Videos: %s | Cortes gerados: %s",
        len(videos),
        total_cortes,
    )
    print("Todos os videos processados.")
    sys.exit(0)


if __name__ == "__main__":
    executar_pipeline_em_lote()