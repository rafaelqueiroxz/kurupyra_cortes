"""
modulo2_diretor.py — Kurupyra Cortes
Arquitetura: Multi-Hook Viral Engine (Google Gemini API)
- Camada 1: Caçador de 5 tipos de gancho viral
- Camada 2: Editor-Chefe com pontuação 1–10
- Duração flexível por tipo de gancho
- 3 variações de título por corte

ATUALIZAÇÃO IMPORTANTE (correção de bug crítico):
- Migrado do SDK legado `google.generativeai` (descontinuado em 30/11/2025)
  para o SDK oficial atual `google-genai`.
- Removido o modelo `gemini-2.0-flash` (aposentado pelo Google em 03/03/2026)
  do valor padrão. Agora usa `gemini-2.5-flash` por padrão.
- A API key NUNCA fica hardcoded — vem de variável de ambiente / .env.
- Erros de chamada de API agora vão pro log real (arquivo), não pro print().
- Retentativas agora usam backoff exponencial com jitter, e distinguem
  erro de rate limit (429) de outros erros.
"""

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError


load_dotenv()  # carrega variáveis do arquivo .env, se existir

log = logging.getLogger("Modulo2")

# ─── CONFIGURAÇÃO ─────────────────────────────────────────────────────────────

GEMINI_MODEL = os.getenv("KURUPYRA_GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CHUNK_SEGUNDOS = int(os.getenv("KURUPYRA_CHUNK_SEGUNDOS", 5 * 60))   # 5 minutos por chunk
OVERLAP_SEGUNDOS = int(os.getenv("KURUPYRA_OVERLAP_SEGUNDOS", 45))
NOTA_MINIMA = float(os.getenv("KURUPYRA_NOTA_MINIMA", 7))            # Cortes com nota < isso são reprovados
MAX_MOMENTOS_POR_CHUNK = int(os.getenv("KURUPYRA_MAX_MOMENTOS_CHUNK", 8))

# Retentativas / backoff
MAX_TENTATIVAS = int(os.getenv("KURUPYRA_MAX_TENTATIVAS", 4))
BACKOFF_BASE_SEGUNDOS = float(os.getenv("KURUPYRA_BACKOFF_BASE", 4))
BACKOFF_MAX_SEGUNDOS = float(os.getenv("KURUPYRA_BACKOFF_MAX", 40))

# Duração (segundos) aceita por tipo de gancho
JANELA_POR_TIPO: dict[str, tuple[float, float]] = {
    "pergunta_forte":       (45.0, 75.0),
    "afirmacao_bombástica": (25.0, 55.0),
    "revelacao":            (50.0, 90.0),
    "numero_chocante":      (30.0, 60.0),
    "virada_narrativa":     (55.0, 90.0),
}
JANELA_PADRAO = (45.0, 75.0)   # Fallback se tipo desconhecido


# ─── PROMPTS ──────────────────────────────────────────────────────────────────

PROMPT_CACADOR_SISTEMA = """
Você é um especialista em viralidade de conteúdo para redes sociais,
com foco em podcasts brasileiros e shorts do YouTube.
Você conhece profundamente os padrões que fazem um trecho viralizar:
ganchos fortes, emoção, surpresa, curiosidade e conclusões impactantes.
Você responde SEMPRE em JSON puro, sem markdown, sem texto extra.
""".strip()

PROMPT_CACADOR_USUARIO = """
Analise a transcrição abaixo e identifique os MOMENTOS VIRAIS mais fortes.

=== REGRA ABSOLUTA DE TIMESTAMPS ===
Os timestamps estão em SEGUNDOS TOTAIS ACUMULADOS.
NÃO converta para minutos. Use o número exato escrito no texto.
Exemplo: [09:30 | 570.5s] → start_time = 570.5, NUNCA 9.5.
=====================================

=== OS 5 TIPOS DE GANCHO VIRAL ===
1. pergunta_forte       — Pergunta polêmica, curiosa ou surpreendente que instiga a resposta
2. afirmacao_bombástica — Afirmação chocante ou controversa que para qualquer um ("Eu perdi R$2M numa semana")
3. revelacao            — Confissão, segredo ou informação inédita nunca dita antes
4. numero_chocante      — Estatística, dado ou número que surpreende ("90% das pessoas erram isso")
5. virada_narrativa     — Momento em que a história vira de direção inesperadamente

=== CRITÉRIOS DE QUALIDADE ===
- O gancho deve prender nos primeiros 3 segundos
- Frases fracas, burocráticas ou de transição devem ser IGNORADAS
- Prefira momentos com emoção: raiva, admiração, surpresa, humor, tristeza
- Identifique a "hook sentence": a frase exata de abertura que prende o espectador
- Retorne NO MÁXIMO {max_momentos} momentos: escolha só os mais fortes deste trecho,
  não force uma lista grande. Qualidade > quantidade.

TRANSCRIÇÃO:
{texto_transcricao}

RESPONDA APENAS COM JSON VÁLIDO NESTE FORMATO (sem markdown, sem texto extra):
{{
  "momentos": [
    {{
      "tipo": "pergunta_forte | afirmacao_bombástica | revelacao | numero_chocante | virada_narrativa",
      "start_time": <segundos totais exatos>,
      "hook_sentence": "Frase exata ou resumo fiel que abre o momento",
      "titulo_emocional": "TÍTULO QUE PROVOCA EMOÇÃO 😱",
      "titulo_curioso": "Título que gera curiosidade sem entregar tudo...",
      "titulo_direto": "FATO DIRETO: O Que Aconteceu"
    }}
  ]
}}
""".strip()

PROMPT_EDITOR_SISTEMA = """
Você é o Editor-Chefe de um canal de cortes de podcast viral com 10 milhões de seguidores.
Você aprova apenas cortes que têm potencial real de viralizar.
Você conhece a diferença entre um momento "ok" e um momento que faz o algoritmo explodir.
Você responde SEMPRE em JSON puro, sem markdown, sem texto extra.
""".strip()

PROMPT_EDITOR_USUARIO = """
Avalie este trecho de podcast para viralidade como short do YouTube.

TIPO DE GANCHO: {tipo_gancho}
HOOK DE ABERTURA: "{hook_sentence}"

--- TRECHO COMPLETO ---
{texto_trecho}
-----------------------

Pontue cada critério de 0 a 10 e calcule a nota final (média ponderada):

- curiosidade    (peso 3): Os primeiros 5 segundos prendem? Dá vontade de continuar?
- autonomia      (peso 2): Faz sentido sem contexto externo? Funciona sozinho?
- emocao         (peso 3): Provoca reação forte — surpresa, raiva, admiração, humor?
- conclusao      (peso 2): Termina de forma satisfatória, com impacto ou punchline?

RESPONDA APENAS COM JSON VÁLIDO (sem markdown, sem texto extra):
{{
  "curiosidade": <0-10>,
  "autonomia": <0-10>,
  "emocao": <0-10>,
  "conclusao": <0-10>,
  "nota_final": <média ponderada arredondada, 0-10>,
  "justificativa": "Análise detalhada em 2-3 frases explicando a nota"
}}
""".strip()


# ─── CLIENTE GEMINI (SDK NOVO: google-genai) ──────────────────────────────────

def criar_cliente() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY nao encontrada. Defina a variavel de ambiente "
            "GEMINI_API_KEY (ex.: em um arquivo .env na raiz do projeto). "
            "Nunca deixe a chave escrita direto no codigo-fonte."
        )
    return genai.Client(api_key=GEMINI_API_KEY)


def chamar_modelo(
    cliente: genai.Client,
    system: str,
    user: str,
    max_tokens: int = 2000,
    contexto: str = "Gemini",
) -> str | None:
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.1,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        # Gemini 2.5 usa tokens de "thinking" por padrao, que consomem o mesmo
        # orcamento de max_output_tokens. Para uma tarefa de extracao/classificacao
        # direta como esta, o thinking so consumia o budget inteiro e deixava a
        # resposta final vazia (finish_reason=MAX_TOKENS, texto=""). Desligamos.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            resposta = cliente.models.generate_content(
                model=GEMINI_MODEL,
                contents=user,
                config=config,
            )
            texto = (resposta.text or "").strip()
            if not texto:
                finish_reason = None
                try:
                    finish_reason = resposta.candidates[0].finish_reason
                except Exception:
                    pass
                log.warning(
                    "[%s] Resposta vazia da API (tentativa %s/%s). finish_reason=%s. "
                    "Se for MAX_TOKENS, aumente max_tokens desta chamada.",
                    contexto, tentativa, MAX_TENTATIVAS, finish_reason,
                )
            return texto

        except ClientError as exc:
            # 429 = rate limit / quota. 4xx em geral = algo estrutural (ex.: modelo
            # inexistente, key invalida) que backoff nao resolve sozinho.
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            log.error(
                "[%s] Erro de cliente na chamada Gemini (tentativa %s/%s, status=%s): %s",
                contexto, tentativa, MAX_TENTATIVAS, status, exc, exc_info=True,
            )
            if status not in (429, "429"):
                # Erro nao relacionado a rate limit (ex.: 404 modelo nao existe,
                # 401/403 key invalida): tentar de novo nao vai resolver.
                return None

        except ServerError as exc:
            log.error(
                "[%s] Erro do servidor Gemini (tentativa %s/%s): %s",
                contexto, tentativa, MAX_TENTATIVAS, exc, exc_info=True,
            )

        except Exception as exc:
            log.error(
                "[%s] Erro inesperado na chamada Gemini (tentativa %s/%s): %s",
                contexto, tentativa, MAX_TENTATIVAS, exc, exc_info=True,
            )

        if tentativa < MAX_TENTATIVAS:
            espera = min(BACKOFF_BASE_SEGUNDOS * (2 ** (tentativa - 1)), BACKOFF_MAX_SEGUNDOS)
            espera += random.uniform(0, espera * 0.25)  # jitter
            log.info("[%s] Aguardando %.1fs antes da proxima tentativa.", contexto, espera)
            time.sleep(espera)

    log.error("[%s] Todas as %s tentativas falharam. Desistindo desta chamada.", contexto, MAX_TENTATIVAS)
    return None


# ─── UTILITÁRIOS ──────────────────────────────────────────────────────────────

def carregar_transcricao(caminho_json: Path) -> dict[str, Any]:
    with open(caminho_json, encoding="utf-8") as arquivo:
        return json.load(arquivo)


def texto_para_chunks(palavras: list[dict], segundos_por_chunk: int = CHUNK_SEGUNDOS) -> list[list[dict]]:
    if not palavras:
        return []

    chunks: list[list[dict]] = []
    inicio_chunk = float(palavras[0]["inicio"])

    while inicio_chunk <= float(palavras[-1]["fim"]):
        fim_chunk = inicio_chunk + segundos_por_chunk
        chunk = [p for p in palavras if inicio_chunk <= float(p["inicio"]) < fim_chunk]
        if chunk:
            chunks.append(chunk)
        proximo_inicio = fim_chunk - OVERLAP_SEGUNDOS
        if proximo_inicio <= inicio_chunk:
            break
        inicio_chunk = proximo_inicio

    log.info(
        "[Chunks] %s palavras divididas em %s blocos de ate %smin com %ss de overlap.",
        len(palavras), len(chunks), segundos_por_chunk // 60, OVERLAP_SEGUNDOS,
    )
    return chunks


def palavras_para_texto_com_timestamp(palavras: list[dict]) -> str:
    linhas: list[str] = []
    linha_atual: list[str] = []
    ultimo_marcador = -999.0

    for palavra in palavras:
        inicio = float(palavra["inicio"])
        if inicio - ultimo_marcador >= 10 or not linha_atual:
            if linha_atual:
                linhas.append(" ".join(linha_atual))
                linha_atual = []
            minutos = int(inicio // 60)
            segundos = int(inicio % 60)
            linha_atual.append(f"[{minutos:02d}:{segundos:02d} | {inicio:.1f}s]")
            ultimo_marcador = inicio
        linha_atual.append(str(palavra["palavra"]))

    if linha_atual:
        linhas.append(" ".join(linha_atual))
    return "\n".join(linhas)


def extrair_texto_do_intervalo(palavras: list[dict], start: float, end: float) -> str:
    trecho = [
        str(p["palavra"])
        for p in palavras
        if float(p["inicio"]) >= start and float(p["fim"]) <= end
    ]
    return " ".join(trecho).strip()


def extrair_json_da_resposta(texto: str) -> str | None:
    texto = texto.replace("```json", "").replace("```", "").strip()
    if not texto:
        return None

    try:
        json.loads(texto)
        return texto
    except json.JSONDecodeError:
        pass

    inicio = texto.find("{")
    if inicio == -1:
        return None

    profundidade = 0
    dentro_string = False
    escape = False
    for indice, char in enumerate(texto[inicio:], start=inicio):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            dentro_string = not dentro_string
            continue
        if dentro_string:
            continue
        if char == "{":
            profundidade += 1
        elif char == "}":
            profundidade -= 1
            if profundidade == 0:
                return texto[inicio: indice + 1]
    return None


def normalizar_hashtags(hashtags: Any) -> list[str]:
    if isinstance(hashtags, str):
        hashtags = re.findall(r"#\w+", hashtags)
    if not isinstance(hashtags, list):
        hashtags = []

    resultado: list[str] = []
    for item in hashtags:
        tag = str(item).strip()
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = f"#{tag}"
        tag = re.sub(r"[^\w#]", "", tag, flags=re.UNICODE)
        if len(tag) > 1 and tag.lower() not in [t.lower() for t in resultado]:
            resultado.append(tag)

    for obrigatoria in ["#podcast", "#cortes", "#shorts"]:
        if obrigatoria.lower() not in [t.lower() for t in resultado]:
            resultado.append(obrigatoria)
    return resultado[:12]


def encontrar_limite_anterior(palavras: list[dict], tempo: float, janela: float = 7.0) -> float:
    candidatos = [
        float(p["fim"])
        for p in palavras
        if tempo - janela <= float(p["fim"]) <= tempo and str(p["palavra"]).rstrip().endswith((".", "!", "?"))
    ]
    return max(candidatos) if candidatos else tempo


def encontrar_limite_posterior(palavras: list[dict], tempo: float, janela: float = 7.0) -> float:
    candidatos = [
        float(p["fim"])
        for p in palavras
        if tempo <= float(p["fim"]) <= tempo + janela and str(p["palavra"]).rstrip().endswith((".", "!", "?"))
    ]
    return min(candidatos) if candidatos else tempo


# ─── VALIDAÇÃO / DEDUPLICAÇÃO ─────────────────────────────────────────────────

def validar_corte(corte: dict, duracao_total: float, indice: int) -> bool:
    campos = ["start_time", "end_time", "titulo_emocional", "tipo"]
    for campo in campos:
        if campo not in corte:
            log.warning("[Validador] Corte %s rejeitado: falta campo '%s'.", indice, campo)
            return False

    try:
        start = float(corte["start_time"])
        end = float(corte["end_time"])
    except (TypeError, ValueError):
        log.warning("[Validador] Corte %s rejeitado: timestamps invalidos.", indice)
        return False

    duracao = end - start
    tipo = corte.get("tipo", "")
    min_dur, max_dur = JANELA_POR_TIPO.get(tipo, JANELA_PADRAO)

    if start < 0 or end <= start or end > duracao_total + 2:
        log.warning("[Validador] Corte %s rejeitado: timestamps fora do intervalo valido.", indice)
        return False
    if duracao < min_dur:
        log.warning("[Validador] Corte %s (%s) rejeitado: muito curto (%.1fs < %.0fs).", indice, tipo, duracao, min_dur)
        return False
    if duracao > max_dur:
        log.warning("[Validador] Corte %s (%s) rejeitado: muito longo (%.1fs > %.0fs).", indice, tipo, duracao, max_dur)
        return False
    return True


def deduplicar_cortes(cortes: list[dict]) -> list[dict]:
    if not cortes:
        return []

    # Mantém o de maior nota em caso de sobreposição
    ordenados = sorted(cortes, key=lambda c: float(c.get("nota_final", 0)), reverse=True)
    resultado: list[dict] = []
    for corte in ordenados:
        start = float(corte["start_time"])
        end = float(corte["end_time"])
        duplicado = False
        for aceito in resultado:
            a_start = float(aceito["start_time"])
            a_end = float(aceito["end_time"])
            sobreposicao = max(0.0, min(end, a_end) - max(start, a_start))
            menor_duracao = min(end - start, a_end - a_start)
            if menor_duracao > 0 and sobreposicao / menor_duracao >= 0.5:
                log.info("[Dedup] Removido por sobreposicao (nota %.1f): %.1f-%.1f.", corte.get("nota_final", 0), start, end)
                duplicado = True
                break
        if not duplicado:
            resultado.append(corte)

    # Re-ordena por start_time para o arquivo final
    return sorted(resultado, key=lambda c: float(c["start_time"]))


# ─── MONTADOR DE BLOCOS ───────────────────────────────────────────────────────

def montar_bloco(momento: dict, palavras: list[dict], duracao_total: float, indice: int) -> dict | None:
    try:
        start = float(momento["start_time"])
    except (KeyError, TypeError, ValueError):
        log.warning("[Montador] Momento %s ignorado: start_time invalido.", indice)
        return None

    if start < 0 or start >= duracao_total:
        log.warning("[Montador] Momento %s ignorado: start_time %.1fs fora do video (%.1fs).", indice, start, duracao_total)
        return None

    tipo = str(momento.get("tipo", "")).strip()
    min_dur, max_dur = JANELA_POR_TIPO.get(tipo, JANELA_PADRAO)
    janela_alvo = (min_dur + max_dur) / 2

    end_alvo = min(start + janela_alvo, duracao_total)
    end_snap = encontrar_limite_posterior(palavras, end_alvo, janela=8.0)
    duracao_snap = end_snap - start

    if duracao_snap < min_dur:
        end_snap = encontrar_limite_posterior(palavras, start + min_dur, janela=12.0)
        duracao_snap = end_snap - start

    if not (min_dur <= duracao_snap <= max_dur):
        log.warning(
            "[Montador] Momento %s (%s): duracao fora da faixa (%.1fs). Descartando.",
            indice, tipo, duracao_snap,
        )
        return None

    log.info(
        "[Montador] Momento %s (%s) montado: %.1f-%.1fs (%.1fs) | \"%s\"",
        indice, tipo, start, end_snap, duracao_snap,
        str(momento.get("hook_sentence", ""))[:60],
    )

    return {
        "start_time": round(start, 3),
        "end_time": round(end_snap, 3),
        "tipo": tipo,
        "hook_sentence": str(momento.get("hook_sentence", "")).strip(),
        "titulo_emocional": str(momento.get("titulo_emocional", "🎙️ DESTAQUE")).strip(),
        "titulo_curioso": str(momento.get("titulo_curioso", "")).strip(),
        "titulo_direto": str(momento.get("titulo_direto", "")).strip(),
        # Alias para compatibilidade com modulo3 e modulo4
        "titulo_youtube": str(momento.get("titulo_emocional", "🎙️ DESTAQUE")).strip(),
        "motivo": f"Gancho viral ({tipo}): '{momento.get('hook_sentence', '')}'",
        "descricao_youtube": "",
        "hashtags": normalizar_hashtags([]),
    }


# ─── CORTE DE SEGURANÇA ───────────────────────────────────────────────────────

def corte_de_seguranca(palavras: list[dict], duracao_total: float) -> dict | None:
    """
    Plano B: encontra o trecho de 60s onde o ritmo de fala mais acelera
    (maior densidade de palavras por segundo), indicando animação ou tensão.

    IMPORTANTE: isto NAO e curadoria de viralidade — e um ultimo recurso
    puramente heuristico para o pipeline nao ficar sem nenhuma saida. Se este
    corte estiver sendo ativado com frequencia, o problema esta nas Camadas 1/2
    (API do Gemini falhando ou nota minima muito rigida), nao aqui.
    """
    JANELA = 60.0
    if duracao_total < JANELA:
        start_seg = float(palavras[0]["inicio"]) if palavras else 0.0
        end_seg = float(palavras[-1]["fim"]) if palavras else duracao_total
    else:
        melhor_start = 0.0
        melhor_densidade = -1.0
        passo = 5.0
        t = 0.0
        while t + JANELA <= duracao_total:
            palavras_janela = [p for p in palavras if t <= float(p["inicio"]) < t + JANELA]
            if palavras_janela:
                primeiros = sum(1 for p in palavras_janela if float(p["inicio"]) < t + 15)
                ultimos = sum(1 for p in palavras_janela if float(p["inicio"]) >= t + 45)
                densidade = (primeiros + ultimos) + ultimos * 0.5
            else:
                densidade = 0.0

            if densidade > melhor_densidade:
                melhor_densidade = densidade
                melhor_start = t
            t += passo

        start_seg = melhor_start
        end_seg = min(duracao_total, melhor_start + JANELA)

    log.warning("[Seguranca] Ativando corte de seguranca: %.1f-%.1fs.", start_seg, end_seg)
    return {
        "start_time": round(start_seg, 3),
        "end_time": round(end_seg, 3),
        "tipo": "virada_narrativa",
        "hook_sentence": "Momento de destaque do podcast",
        "titulo_emocional": "🎙️ MOMENTO QUE PAROU TUDO NO PODCAST",
        "titulo_curioso": "O que aconteceu aqui ninguém esperava...",
        "titulo_direto": "Destaque do Episódio",
        "titulo_youtube": "🎙️ MOMENTO QUE PAROU TUDO NO PODCAST",
        "motivo": "Corte de seguranca: trecho com maior ritmo de fala.",
        "descricao_youtube": "",
        "hashtags": ["#podcast", "#cortes", "#shorts"],
        "nota_final": 5.0,
        "_seguranca": True,
    }


# ─── CAMADA 1: CAÇADOR MULTI-HOOK ─────────────────────────────────────────────

def camada1_cacador_hooks(
    cliente: genai.Client,
    texto_chunk: str,
    chunk_num: int,
    total_chunks: int,
) -> list[dict]:
    cabecalho = f"Chunk {chunk_num}/{total_chunks}. Transcrição com timestamps:\n{texto_chunk}"
    prompt_usuario = PROMPT_CACADOR_USUARIO.format(
        texto_transcricao=cabecalho,
        max_momentos=MAX_MOMENTOS_POR_CHUNK,
    )

    log.info("[Camada1] Chunk %s/%s — caçando ate %s ganchos virais.", chunk_num, total_chunks, MAX_MOMENTOS_POR_CHUNK)
    resposta_raw = chamar_modelo(
        cliente,
        system=PROMPT_CACADOR_SISTEMA,
        user=prompt_usuario,
        max_tokens=4000,
        contexto=f"Camada1-Chunk{chunk_num}",
    )

    if not resposta_raw:
        log.warning("[Camada1] Chunk %s: sem resposta da API. Pulando.", chunk_num)
        return []

    json_str = extrair_json_da_resposta(resposta_raw)
    if not json_str:
        log.warning("[Camada1] Chunk %s: JSON nao encontrado na resposta. Pulando.", chunk_num)
        return []

    try:
        dados = json.loads(json_str)
        momentos = dados.get("momentos", [])
        if not isinstance(momentos, list):
            return []
        log.info("[Camada1] Chunk %s: %s momento(s) identificado(s).", chunk_num, len(momentos))
        return momentos
    except json.JSONDecodeError as exc:
        log.warning("[Camada1] Chunk %s: falha ao parsear JSON: %s", chunk_num, exc)
        return []


# ─── CAMADA 2: EDITOR-CHEFE COM PONTUAÇÃO ────────────────────────────────────

def camada2_editor_chefe(
    cliente: genai.Client,
    texto_trecho: str,
    corte: dict,
) -> float:
    """
    Avalia o trecho e retorna a nota final (0–10).
    Retorna 0.0 em caso de falha (reprovado por segurança).
    """
    start = corte.get("start_time", "?")
    end = corte.get("end_time", "?")

    if not texto_trecho.strip():
        log.warning("[Camada2] Corte %.1f-%.1f: texto vazio. Reprovado.", start, end)
        return 0.0

    prompt_usuario = PROMPT_EDITOR_USUARIO.format(
        tipo_gancho=corte.get("tipo", "desconhecido"),
        hook_sentence=corte.get("hook_sentence", ""),
        texto_trecho=texto_trecho,
    )

    resposta_raw = chamar_modelo(
        cliente,
        system=PROMPT_EDITOR_SISTEMA,
        user=prompt_usuario,
        max_tokens=600,
        contexto=f"Camada2-{start:.0f}-{end:.0f}",
    )

    if not resposta_raw:
        log.warning("[Camada2] Corte %.1f-%.1f: sem resposta. Reprovado.", start, end)
        return 0.0

    json_str = extrair_json_da_resposta(resposta_raw)
    if not json_str:
        log.warning("[Camada2] Corte %.1f-%.1f: JSON invalido. Reprovado.", start, end)
        return 0.0

    try:
        resultado = json.loads(json_str)

        # Calcula média ponderada como verificação cruzada
        pesos = {"curiosidade": 3, "autonomia": 2, "emocao": 3, "conclusao": 2}
        soma_ponderada = sum(float(resultado.get(k, 0)) * v for k, v in pesos.items())
        total_pesos = sum(pesos.values())
        nota_calculada = round(soma_ponderada / total_pesos, 1)

        nota_modelo = float(resultado.get("nota_final", 0))
        nota_final = round((nota_calculada + nota_modelo) / 2, 1)
        justificativa = resultado.get("justificativa", "sem justificativa")

        emoji = "✅ APROVADO" if nota_final >= NOTA_MINIMA else "❌ REPROVADO"
        log.info(
            "[Camada2] %s (%.1f-%.1fs) | Nota: %.1f/10 | %s",
            emoji, start, end, nota_final, justificativa[:80],
        )
        return nota_final

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        log.warning("[Camada2] Corte %.1f-%.1f: erro ao parsear nota: %s", start, end, exc)
        return 0.0


# ─── ORQUESTRADOR PRINCIPAL ───────────────────────────────────────────────────

def analisar_transcricao(caminho_json: Path) -> Path | None:
    caminho_json = Path(caminho_json)
    log.info("[Modulo 2] Analisando transcricao: %s", caminho_json)
    log.info("[Modulo 2] Modelo Gemini em uso: %s", GEMINI_MODEL)

    try:
        transcricao = carregar_transcricao(caminho_json)
    except Exception as exc:
        log.error("[Modulo 2] Falha ao ler transcricao: %s", exc, exc_info=True)
        return None

    palavras = transcricao.get("palavras", [])
    duracao_total = float(transcricao.get("duracao_estimada_segundos", 0.0))
    if not palavras:
        log.error("[Modulo 2] Transcricao sem palavras. Impossivel selecionar cortes.")
        return None

    try:
        cliente = criar_cliente()
    except RuntimeError as exc:
        log.error("[Modulo 2] %s", exc)
        return None

    chunks = texto_para_chunks(palavras)
    todos_momentos: list[dict] = []

    # ── CAMADA 1: Caçador varre todos os chunks ────────────────────────────────
    for indice_chunk, chunk in enumerate(chunks, 1):
        texto_chunk = palavras_para_texto_com_timestamp(chunk)
        momentos = camada1_cacador_hooks(cliente, texto_chunk, indice_chunk, len(chunks))
        for momento in momentos:
            if isinstance(momento, dict):
                momento["_chunk"] = indice_chunk
                todos_momentos.append(momento)
        time.sleep(3)  # Respeita rate limit da tier gratuita

    log.info("[Modulo 2] Camada 1 concluida: %s momento(s) identificado(s).", len(todos_momentos))

    # ── MONTADOR: Constrói cortes deterministicamente ─────────────────────────
    cortes_pre_revisao: list[dict] = []
    for indice, momento in enumerate(todos_momentos, 1):
        bloco = montar_bloco(momento, palavras, duracao_total, indice)
        if bloco and validar_corte(bloco, duracao_total, indice):
            cortes_pre_revisao.append(bloco)

    log.info("[Modulo 2] %s bloco(s) tecnicamente validos passam para a Camada 2.", len(cortes_pre_revisao))

    # ── CAMADA 2: Editor-Chefe pontua cada bloco ──────────────────────────────
    cortes_aprovados: list[dict] = []
    for corte in cortes_pre_revisao:
        texto_trecho = extrair_texto_do_intervalo(palavras, corte["start_time"], corte["end_time"])
        nota = camada2_editor_chefe(cliente, texto_trecho, corte)
        corte["nota_final"] = nota
        if nota >= NOTA_MINIMA:
            cortes_aprovados.append(corte)
        time.sleep(3)  # Respeita rate limit da tier gratuita

    cortes_aprovados.sort(key=lambda c: float(c.get("nota_final", 0)), reverse=True)
    log.info(
        "[Modulo 2] Camada 2 concluida: %s/%s blocos aprovados (nota >= %s).",
        len(cortes_aprovados), len(cortes_pre_revisao), NOTA_MINIMA,
    )

    # ── Fallback de segurança ─────────────────────────────────────────────────
    if not cortes_aprovados:
        log.warning(
            "[Modulo 2] Nenhum bloco aprovado pela IA. Ativando corte de seguranca "
            "(fallback heuristico, NAO e curadoria de viralidade). Se isto estiver "
            "acontecendo com frequencia, revise os logs de erro da API Gemini acima.",
        )
        seguranca = corte_de_seguranca(palavras, duracao_total)
        if seguranca:
            cortes_aprovados.append(seguranca)

    # ── Deduplicação final ────────────────────────────────────────────────────
    cortes_finais = deduplicar_cortes(cortes_aprovados)

    # ── Salva resultado ───────────────────────────────────────────────────────
    caminho_saida = caminho_json.parent / "cortes_sugeridos.json"
    resultado = {
        "arquivo_origem": transcricao.get("arquivo_origem", caminho_json.parent.name),
        "modelo_gemini": GEMINI_MODEL,
        "arquitetura": "multi-hook-viral-engine",
        "nota_minima_aprovacao": NOTA_MINIMA,
        "chunk_segundos": CHUNK_SEGUNDOS,
        "overlap_segundos": OVERLAP_SEGUNDOS,
        "total_momentos_identificados": len(todos_momentos),
        "total_blocos_validos": len(cortes_pre_revisao),
        "total_aprovados_camada2": len(cortes_aprovados),
        "total_cortes_finais": len(cortes_finais),
        "fallback_seguranca_ativado": any(c.get("_seguranca") for c in cortes_finais),
        "cortes": cortes_finais,
    }

    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=2)

    log.info(
        "[Modulo 2] Salvo: %s momentos | %s validos | %s aprovados | %s finais -> %s",
        len(todos_momentos),
        len(cortes_pre_revisao),
        len(cortes_aprovados),
        len(cortes_finais),
        caminho_saida,
    )
    return caminho_saida if cortes_finais else None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s -> %(message)s")
    if len(sys.argv) < 2:
        print("Uso: python modulo2_diretor.py <caminho/transcricao_completa.json>")
        raise SystemExit(1)
    analisar_transcricao(Path(sys.argv[1]))