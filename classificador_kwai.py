"""
CLASSIFICADOR KWAI - VERSÃO OTIMIZADA
====================================
- Reduz inconclusivos usando fallback de melhor categoria com baixa confiança
- Normaliza e valida textos
- Protege credenciais usando variáveis de ambiente
"""

import base64
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta

import faiss
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN", "kuaishousupport.freshdesk.com")
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY", "")

STATUS_DESEJADOS = [4]
PRIORIDADES_DESEJADAS = None
DIAS_ATRAS = None
LIMITE_TICKETS = 500
GROUP_ID = None

# THRESHOLDS
THRESHOLD_POSITIVO = 0.40
THRESHOLD_NEGATIVO = 0.35
MARGEM_MINIMA = 0.0
TOP_K = 10

# Fallbacks para reduzir inconclusivos
MIN_SCORE_FLOOR = 0.28  # abaixo disso vira inconclusivo
ALLOW_LOW_CONFIDENCE_ASSIGN = True

# ============================================================================
# TEMAS
# ============================================================================

TEMAS = {
    "Saque e Retirada": {
        "positivos": [
            "não consigo sacar", "saque rejeitado", "saque cancelado",
            "falha no saque", "erro ao sacar", "saque pendente",
            "saque bloqueado", "impossível sacar", "saque não caiu",
            "saque em análise", "saque travado", "saque negado",
            "saque inválido", "não sai meu dinheiro", "dinheiro não cai",
            "retirada bloqueada", "retirar valores", "transferir saldo",
            "pegar meu dinheiro", "cadê meu saque", "saque sumiu",
            "tirar dinheiro", "sacar grana", "resgatar dinheiro",
            "puxar saldo", "transferência bloqueada", "saldo travado",
            "grana presa", "dinheiro parado", "não solta dinheiro",
            "chave pix inválida saque", "conta bancária rejeitada saque",
            "pix não funciona saque", "banco recusou saque",
            "dados bancários errados", "pix bloqueado",
            "saque demora quanto tempo", "saque atrasado", "esperando saque",
            "atraso saque", "quanto demora sacar", "tempo processamento saque",
            "sacar por pix", "transferir banco", "saque via transferência",
            "retirada por pix", "sacar banco", "erro processar saque",
            "falha retirada", "bug no saque",
        ],
        "negativos": [
            "perdi minha conta", "fui banido permanentemente",
            "não consigo postar video", "meu video foi removido",
            "ganhar moedas assistindo", "como ganhar golds",
            "recarga de diamantes", "shop entrega atrasada",
            "comprei produto nao chegou", "taxa de correios",
            "mudar senha", "denunciar perfil",
        ],
    },
    "Banimento e Suspensão": {
        "positivos": [
            "conta banida", "conta suspensa", "banimento permanente",
            "bloqueio temporário", "suspenso 3 dias", "banido 7 dias",
            "bloqueio definitivo", "conta desativada", "perfil bloqueado",
            "usuário suspenso", "banimento automático", "fui banido",
            "me baniram", "bloquearam minha conta", "suspenderam conta",
            "perfil cancelado", "não consigo mais usar", "bloquearam acesso",
            "conta inativa", "desativaram conta", "banimento injusto",
            "apelar banimento", "contestar bloqueio", "recurso conta banida",
            "reativar conta banida", "reverter banimento",
            "desbloquear conta", "tirar banimento", "remover bloqueio",
            "recuperar conta banida", "pedir revisão banimento",
            "questionar suspensão", "violação detectada",
            "banimento infração", "sanção aplicada",
        ],
        "negativos": [
            "quero dinheiro", "pix nao caiu", "esqueci minha senha",
            "perdi meu chip", "como apagar conta", "deletar perfil",
            "produto com defeito", "reembolso de compra", "ganhar moedas",
            "tarefa sumiu",
        ],
    },
    "Tarefas e Missões": {
        "positivos": [
            "tarefa sumiu", "tarefa desapareceu", "não encontro tarefa",
            "cadê tarefa", "missão sumiu", "tarefa não aparece",
            "tarefa invisível", "missão desapareceu", "tarefa removida",
            "sumiram tarefas", "não consigo fazer tarefa",
            "tarefa não aceita", "bloqueado missão",
            "tarefa não deixa participar", "acesso negado missão",
            "tarefa bugada", "tarefa erro", "missão travando",
            "tarefa não funciona", "erro completar missão",
            "bug tarefa", "tarefa quebrada", "falha tarefa",
            "fiz tarefa não contou", "progresso não salvou",
            "completei não validou", "tarefa zerou",
            "dias sequência perdidos", "missão resetou",
            "perdi progresso", "tarefa não registrou",
            "fiz mas não vale", "recompensa tarefa não veio",
            "prêmio missão não chegou", "completei mas não recebi",
            "bônus tarefa não veio",
        ],
        "negativos": [
            "fui banido", "conta suspensa", "recuperar acesso",
            "senha incorreta", "mudar foto perfil", "pix rejeitado",
            "transferencia pendente", "pagar boleto", "live bloqueada",
            "denunciar video",
        ],
    },
    "Compras e Kwai Shop": {
        "positivos": [
            "comprei kwai shop", "produto não chegou", "item diferente",
            "produto defeito", "cor errada", "tamanho errado",
            "pedido kwai", "compra shop", "produto shop não veio",
            "encomenda shop", "artigo defeituoso", "mercadoria errada",
            "item danificado", "produto quebrado", "embalagem violada",
            "lacre rompido", "qualidade inferior", "material diferente",
            "falsificado produto", "pedido não chegou", "rastreamento parado",
            "prazo entrega estourado", "código rastreio inválido",
            "entrega atrasada", "produto não entregue", "onde está pedido",
            "cadê encomenda", "quero reembolso compra", "devolução produto",
            "cancelar compra", "estornar pagamento", "devolver item",
            "restituir valor", "quero dinheiro volta", "solicitar estorno",
        ],
        "negativos": [
            "quero sacar meu saldo", "retirar dinheiro",
            "transferir para o banco", "tarefa sumiu", "perdi o acesso",
            "banimento injusto", "live bloqueada", "como ganhar moedas",
        ],
    },
    "Recuperação de Acesso": {
        "positivos": [
            "não consigo entrar conta", "recuperar acesso",
            "resgatar conta", "perdi acesso conta", "voltar conta",
            "restaurar conta", "recuperação login", "resgate credenciais",
            "login conta antiga", "acessar perfil antigo", "esqueci senha",
            "não lembro senha", "redefinir senha", "resetar senha",
            "perdi senha", "recuperar senha", "mudar senha esquecida",
            "perdi chip", "troquei número", "celular roubado conta",
            "não tenho telefone", "mudei número", "chip antigo",
            "sem acesso telefone", "telefone perdido", "não lembro email",
            "email antigo sem acesso", "não sei email",
            "esqueci email cadastrado", "código não chega",
            "não recebo sms", "código expirou", "sms não vem",
            "verificação não chega",
        ],
        "negativos": [
            "fui banido por violação", "saque em analise",
            "dinheiro não caiu", "missão bugada", "anúncio falso",
            "produto quebrado", "quero deletar perfil",
        ],
    },
    "Live e Transmissão": {
        "positivos": [
            "live bloqueada", "transmissão suspensa", "não consigo abrir live",
            "erro na live", "live caiu", "banido da live", "fazer live",
            "função live sumiu", "problema na transmissão",
            "live travando", "imagem live ruim", "áudio live mudo",
            "convidado live erro", "live encerrada sozinha",
            "limitação de live", "transmissão ao vivo erro",
            "não posso comentar na live", "chat da live sumiu",
            "pedir autorização live", "requisitos live",
            "permissão live negada",
        ],
        "negativos": [
            "esqueci minha senha", "perdi meu chip", "sacar dinheiro pix",
            "produto não entregue", "tarefa diaria sumiu",
            "banimento de conta permanente", "excluir perfil",
        ],
    },
    "Monetização de Criadores": {
        "positivos": [
            "sou criador não recebi", "pagamento criador atrasado",
            "monetização não caiu", "meu bônus de criador",
            "fundo de criadores", "receita de vídeos",
            "ganhos de conteúdo", "salário de contratado",
            "agência não pagou", "contrato criador kwai",
            "ganhos de visualização", "remuneração de vídeos",
            "meus mimos não recebi", "diamantes criador não chegaram",
            "incentivo criador", "pagamento família kwai",
            "receber por vídeos",
        ],
        "negativos": [
            "sacar meu dinheiro comum", "fazer tarefa de convite",
            "comprar moedas no shop", "recuperar senha",
            "fui banido por nudez", "denunciar vídeo",
        ],
    },
    "Moedas e Kwai Golds": {
        "positivos": [
            "golds não chegaram", "moedas sumiram",
            "trocar golds por dinheiro", "não consigo converter golds",
            "meus kwai golds sumiram", "problema conversão moedas",
            "ganhar mais golds", "moedas de bônus",
            "saldo de moedas errado", "converter moedas em real",
            "falha ao trocar golds", "golds pendentes",
            "recompensa em moedas", "golds de assistir videos",
        ],
        "negativos": [
            "pix não caiu banco", "transferencia saque",
            "entrar na minha conta", "produto shop atrasado",
            "live bloqueada", "excluir videos",
        ],
    },
    "Exclusão e Deletar Conta": {
        "positivos": [
            "quero excluir conta", "deletar perfil", "apagar conta",
            "encerrar conta", "desativar conta", "cancelar perfil",
            "remover conta", "eliminar conta", "como deletar kwai",
            "apagar meus dados", "excluir definitivo", "limpar conta",
            "desvincular tudo", "sair do kwai para sempre",
        ],
        "negativos": [
            "fui banido injustamente", "perdi meu acesso senha",
            "quero sacar saldo", "missão não funciona",
            "denunciar alguém", "bloquear usuario",
        ],
    },
    "Denúncias e Segurança": {
        "positivos": [
            "denunciar usuário", "denunciar perfil", "reportar comportamento",
            "fazer denúncia", "reportar conta", "denunciar vídeo",
            "reportar conteúdo", "acusar usuário", "notificar violação",
            "menor idade app", "criança usando kwai", "denunciar menor",
            "conteúdo pornográfico", "nudez vídeo", "live sexual",
            "discurso ódio", "assédio chat", "ameaças", "bullying",
            "conta fake", "perfil falso", "plágio vídeos",
        ],
        "negativos": [
            "meu saque foi negado", "não consigo entrar senha",
            "perdi meu chip celular", "tarefa sumiu da tela",
            "comprar no kwai shop", "receber meu salario",
        ],
    },
    "Kwai Golds e Recargas (Diamantes)": {
        "positivos": [
            "comprei diamantes não chegou", "recarreguei diamantes",
            "paguei diamantes não veio", "diamantes sumiram",
            "problema na recarga", "diamantes não creditados",
            "falha compra diamantes", "preço recarga errado",
            "recarga pix diamantes", "moedas virtuais presente",
            "diamantes live",
        ],
        "negativos": [
            "quero sacar para o banco", "minha conta foi banida",
            "esqueci meu login", "produto shop quebrado",
            "fazer tarefa de assistir",
        ],
    },
    "Propagandas e Anúncios": {
        "positivos": [
            "anúncio falso", "propaganda enganosa", "anúncio mentiroso",
            "vi propaganda e era golpe", "anúncio de jogo que paga",
            "propaganda abusiva", "anúncio travando app",
            "muita propaganda", "anúncio que não fecha",
            "propaganda irritante", "anúncio de cassino",
            "fraude em anúncio",
        ],
        "negativos": [
            "postar meu vídeo", "falar com suporte",
            "deletar minha conta", "sacar meu saldo",
            "entrar no meu perfil", "minha live caiu",
        ],
    },
    "Problemas Técnicos e App": {
        "positivos": [
            "app fechando", "aplicativo trava", "não abre kwai",
            "erro de conexão", "falha no sistema", "lentidão no app",
            "app pesado", "consumo bateria kwai", "esquentando celular",
            "não carrega vídeo", "vídeo sem som", "áudio atrasado",
            "erro ao postar", "upload falhou", "processamento vídeo travado",
            "atualização deu erro", "bug na câmera", "filtro não funciona",
            "tela preta kwai", "erro servidor",
        ],
        "negativos": [
            "fui banido por regras", "meu saque não caiu",
            "quero apagar perfil", "comprei e não chegou",
            "ganhar dinheiro convite",
        ],
    },
    "Privacidade e Dados": {
        "positivos": [
            "meus dados vazaram", "privacidade da conta",
            "quem viu meu perfil", "esconder meus videos",
            "conta privada", "proteger meus dados", "vincular instagram",
            "vincular tiktok", "mudar meu id", "mudar nome usuario",
            "alterar foto perfil", "mudar biografia", "tirar meu numero",
            "mudar localização",
        ],
        "negativos": [
            "sacar pix", "banimento injusto", "tarefa não paga",
            "produto shop defeito", "código sms não chega",
        ],
    },
    "Engajamento e Seguidores": {
        "positivos": [
            "perdi seguidores", "meus seguidores sumiram",
            "não ganho curtidas", "meu video não tem view",
            "flopado", "conta flopada", "não consigo seguir ninguem",
            "seguindo sozinho", "parar de seguir", "ganhar seguidores",
            "visualizações cairam", "engajamento baixo",
            "meu video nao entrega",
        ],
        "negativos": [
            "fui banido pela diretriz", "não consigo entrar na conta",
            "sacar dinheiro agora", "erro de saque pix", "compra no shop",
        ],
    },
    "Atendimento e Suporte": {
        "positivos": [
            "suporte não responde", "falar com atendente",
            "atendimento humano", "ninguém me ajuda",
            "chat não funciona", "e-mail suporte ignorado",
            "resposta automática", "quero falar com kwai",
            "reclamação sem resposta", "demora atendimento",
            "não resolvem meu problema", "abrir chamado",
            "suporte técnico não retorno", "ouvidoria kwai",
        ],
        "negativos": [
            "postar video novo", "ganhar moedas ouro",
            "produto shop rastreio", "senha errada acesso",
            "fui banido de live",
        ],
    },
    "Família e Grupos": {
        "positivos": [
            "minha família kwai", "sair da família", "entrar em família",
            "líder da família", "problema no grupo",
            "chat da família sumiu", "pontuação família",
            "ranking família", "recompensa de família", "convite família",
            "banido da família",
        ],
        "negativos": [
            "perdi minha conta senha", "saque pix banco",
            "produto shop atrasado", "denunciar pornografia",
            "banimento conta geral",
        ],
    },
    "Direitos Autorais e Plágio": {
        "positivos": [
            "roubaram meu vídeo", "vídeo removido direito autoral",
            "música bloqueada", "direitos autorais áudio",
            "copiaram meu conteúdo", "uso indevido imagem",
            "denunciar plágio", "reivindicação de vídeo",
            "originalidade vídeo", "aviso de copyright",
        ],
        "negativos": [
            "como sacar dinheiro", "tarefa de assistir",
            "esqueci minha senha", "perdi meu chip",
            "comprei moedas golds", "fui banido por spam",
        ],
    },
}

# ============================================================================
# CLASSES
# ============================================================================


class FreshdeskAPI:
    def __init__(self, domain: str, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "FRESHDESK_API_KEY vazio. Defina a variável de ambiente antes de rodar."
            )
        domain = domain.replace("https://", "").replace("http://", "").strip()
        self.base_url = f"https://{domain}/api/v2"
        auth = f"{api_key}:X"
        self.headers = {
            "Authorization": f"Basic {base64.b64encode(auth.encode()).decode()}",
            "Content-Type": "application/json",
        }
        print(f"🔗 Conectado: {self.base_url}")

    def buscar_tickets(
        self,
        status_list,
        prioridades=None,
        limite=None,
        dias_atras=None,
        group_id=None,
    ):
        print("\n🔄 Buscando tickets...")

        filtros = [f"Status: {status_list}"]
        if prioridades:
            filtros.append(f"Prioridades: {prioridades}")
        if limite:
            filtros.append(f"Limite: {limite}")
        if dias_atras:
            filtros.append(f"Últimos {dias_atras} dias")
        if group_id:
            filtros.append(f"Grupo: {group_id}")
        print(f"   {' | '.join(filtros)}")

        data_limite = None
        if dias_atras:
            data_limite = datetime.now() - timedelta(days=dias_atras)

        todos_tickets = []
        page = 1

        while True:
            params = {"page": page, "per_page": 100}
            if group_id:
                params["group_id"] = group_id

            try:
                resp = requests.get(
                    f"{self.base_url}/tickets",
                    headers=self.headers,
                    params=params,
                    timeout=30,
                )

                if resp.status_code == 200:
                    tickets = resp.json()
                    if not tickets:
                        break

                    for t in tickets:
                        if t.get("status") not in status_list:
                            continue
                        if prioridades and t.get("priority") not in prioridades:
                            continue
                        if data_limite:
                            try:
                                created = datetime.fromisoformat(
                                    t["created_at"].replace("Z", "+00:00")
                                ).replace(tzinfo=None)
                                if created < data_limite:
                                    continue
                            except Exception:
                                pass

                        todos_tickets.append(t)
                        if limite and len(todos_tickets) >= limite:
                            return todos_tickets[:limite]

                    print(f"   Página {page}: {len(todos_tickets)} tickets")
                    page += 1

                elif resp.status_code == 429:
                    print("   ⏳ Rate limit...")
                    import time

                    time.sleep(60)
                else:
                    break
            except Exception as exc:
                print(f"   ❌ Erro: {str(exc)[:80]}")
                break

        print(f"✅ Total: {len(todos_tickets)}")
        return todos_tickets

    def para_dataframe(self, tickets):
        if not tickets:
            return pd.DataFrame()
        dados = [
            {
                "ID": t.get("id"),
                "Assunto": t.get("subject", ""),
                "Descricao": t.get("description_text", ""),
                "Status": t.get("status"),
                "Prioridade": t.get("priority"),
                "Criado": t.get("created_at"),
                "Atualizado": t.get("updated_at"),
                "Grupo_ID": t.get("group_id"),
            }
            for t in tickets
        ]
        return pd.DataFrame(dados)


@dataclass
class ScoreInfo:
    pos: float
    neg: float
    score_final: float


class ClassificadorInteligente:
    def __init__(self, threshold_pos=0.40, threshold_neg=0.35, margem=0.0, top_k=10):
        print("\n🔄 Carregando MPNet...")
        self.model = SentenceTransformer(
            "paraphrase-multilingual-mpnet-base-v2", device="cpu"
        )
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.margem = margem
        self.top_k = top_k
        self.indices = {}
        self.negativos = {}
        print("✅ Modelo carregado!")

    def preparar(self, temas_dict):
        print("\n🔄 Criando índices FAISS...")
        for tema, dados in temas_dict.items():
            positivos = dados["positivos"]
            emb_pos = self.model.encode(
                positivos,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )

            dim = emb_pos.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(emb_pos.astype("float32"))
            self.indices[tema] = {"index": index, "count": len(positivos)}

            negativos = dados.get("negativos", [])
            if negativos:
                emb_neg = self.model.encode(
                    negativos,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32,
                )
                self.negativos[tema] = emb_neg

            print(f"   ✅ {tema}: {len(positivos)}+ / {len(negativos)}-")
        print(f"✅ {len(self.indices)} índices prontos!")

    def _score_negativo(self, emb, tema):
        if tema not in self.negativos:
            return 0.0
        scores = np.dot(self.negativos[tema], emb)
        return float(np.max(scores))

    def _score_final(self, score_pos: float, score_neg: float) -> float:
        penalidade = max(0.0, score_neg - self.threshold_neg)
        return score_pos - penalidade

    def classificar(self, texto):
        if pd.isna(texto) or str(texto).strip() == "":
            return "Vazio", 0.0, 0.0, "vazio"

        emb = self.model.encode(
            [str(texto)], normalize_embeddings=True, show_progress_bar=False
        )[0]

        resultados = {}
        for tema, data in self.indices.items():
            scores, _ = data["index"].search(
                emb.reshape(1, -1).astype("float32"), min(self.top_k, data["count"])
            )
            score_pos = float(np.mean(scores[0]))
            score_neg = self._score_negativo(emb, tema)
            score_final = self._score_final(score_pos, score_neg)
            resultados[tema] = ScoreInfo(score_pos, score_neg, score_final)

        ordenados = sorted(resultados.items(), key=lambda x: x[1].score_final, reverse=True)
        melhor_tema, melhor_score = ordenados[0]
        segundo_score = ordenados[1][1].score_final if len(ordenados) > 1 else -1.0

        if melhor_score.score_final < MIN_SCORE_FLOOR:
            return "Inconclusivo", melhor_score.pos, melhor_score.neg, "score_baixo"

        if melhor_score.pos < self.threshold_pos or melhor_score.neg > self.threshold_neg:
            if ALLOW_LOW_CONFIDENCE_ASSIGN:
                return (
                    melhor_tema,
                    melhor_score.pos,
                    melhor_score.neg,
                    "baixa_confianca",
                )
            return "Inconclusivo", melhor_score.pos, melhor_score.neg, "threshold"

        if self.margem > 0 and (melhor_score.score_final - segundo_score) < self.margem:
            if ALLOW_LOW_CONFIDENCE_ASSIGN:
                return melhor_tema, melhor_score.pos, melhor_score.neg, "baixa_margem"
            return "Inconclusivo", melhor_score.pos, melhor_score.neg, "margem_baixa"

        return melhor_tema, melhor_score.pos, melhor_score.neg, "ok"

    def processar_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\n🔄 Classificando {len(df)} tickets...")
        categorias, scores_pos, scores_neg, confiancas = [], [], [], []
        total = len(df)

        for idx, row in df.iterrows():
            texto = f"{row['Assunto']} {row['Descricao']}"
            cat, sp, sn, conf = self.classificar(texto)
            categorias.append(cat)
            scores_pos.append(sp)
            scores_neg.append(sn)
            confiancas.append(conf)

            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                print(f"   {idx + 1}/{total}")

        df["Categoria"] = categorias
        df["Score_Pos"] = scores_pos
        df["Score_Neg"] = scores_neg
        df["Confianca"] = confiancas
        print("✅ Concluído!")
        return df


# ============================================================================
# RELATÓRIOS
# ============================================================================

def gerar_excels(df: pd.DataFrame):
    print("\n🔄 Gerando Excel...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivos = []

    for cat in df["Categoria"].unique():
        df_cat = df[df["Categoria"] == cat].copy()
        df_export = df_cat[
            [
                "ID",
                "Assunto",
                "Descricao",
                "Status",
                "Prioridade",
                "Score_Pos",
                "Score_Neg",
                "Confianca",
                "Criado",
            ]
        ].copy()
        nome = f"Kwai_{cat.replace(' ', '_')}_{timestamp}.xlsx"
        try:
            df_export.to_excel(nome, index=False, engine="openpyxl")
            arquivos.append(nome)
            print(f"   ✅ {nome} ({len(df_cat)})")
        except Exception as exc:
            print(f"   ❌ {nome}: {exc}")

    print(f"✅ {len(arquivos)} arquivos!")
    return arquivos


def relatorio(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("📊 RELATÓRIO")
    print("=" * 70)
    total = len(df)
    contagem = df["Categoria"].value_counts()
    print(f"\n{'Categoria':<35} {'Qtd':>6} {'%':>7} {'Score':>8}")
    print("-" * 70)
    for cat, qtd in contagem.items():
        perc = (qtd / total) * 100
        score = df[df["Categoria"] == cat]["Score_Pos"].mean()
        print(f"{cat:<35} {qtd:>6} {perc:>6.1f}% {score:>7.3f}")
    print("=" * 70)
    print(f"{'TOTAL':<35} {total:>6} {'100%':>7}")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🎯 KWAI CLASSIFICADOR - OTIMIZADO")
    print("=" * 70)

    try:
        api = FreshdeskAPI(FRESHDESK_DOMAIN, FRESHDESK_API_KEY)
        tickets = api.buscar_tickets(
            STATUS_DESEJADOS,
            PRIORIDADES_DESEJADAS,
            LIMITE_TICKETS,
            DIAS_ATRAS,
            GROUP_ID,
        )
        if not tickets:
            print("\n❌ Nenhum ticket!")
            return

        df = api.para_dataframe(tickets)
        print(f"\n✅ DataFrame: {len(df)}")

        clf = ClassificadorInteligente(
            THRESHOLD_POSITIVO, THRESHOLD_NEGATIVO, MARGEM_MINIMA, TOP_K
        )
        clf.preparar(TEMAS)
        df_result = clf.processar_dataframe(df)

        relatorio(df_result)
        arquivos = gerar_excels(df_result)

        print("\n🎉 COMPLETO!")
        print(f"\n📊 {len(arquivos)} arquivos gerados")
        for arq in arquivos:
            print(f"   {arq}")

    except Exception as exc:
        print(f"\n❌ ERRO: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
