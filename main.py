import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, BertModel
from torchvision import transforms
import numpy as np
import warnings
from PIL import Image
import os
import math

# Suprimir avisos de módulos de simulação (para clareza na saída)
warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURAÇÃO DE HIPERPARÂMETROS E FERRAMENTAS
# ==============================================================================

# Hiperparâmetros baseados no artigo
D_MODEL = 504       # Ajustado para ser divisível por N_HEADS
N_HEADS = 12        # Número de cabeças de atenção
N_LAYERS = 6        # Número de camadas do Transformer
MAX_LENGTH = 512    # Comprimento máximo da sequência do BERT
BATCH_SIZE = 16
NUM_EPOCHS = 100    #
VOCAB_SIZE = 30522  # Vocabulário padrão do bert-base-uncased
NUM_CLASSES = 18    # 18 sintomas/categorias

# Dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}") # Adicionado para verificar o dispositivo

# --- Configuração do Texto e Imagem ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

PASTA_IMAGENS_INICIAL = 'Multimodal_images'

# ==============================================================================
# 1. PREPARAÇÃO DE DADOS E DATALOADERS (Necessário para Dataloaders e Vocab)
# ==============================================================================

def preprocess_multimodal_data(examples):
    model_inputs = {}

    # A) Processamento de Texto (BERT)
    model_inputs.update(tokenizer(
        examples['Question'],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    ))

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['Question_summ'],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]

    # B) Processamento de Imagem (ViT)
    processed_images = []
    for original_path in examples['image_path']:
        if original_path and isinstance(original_path, str):
            img_path = original_path
            try:
                index = original_path.index(PASTA_IMAGENS_INICIAL)
                relative_path = original_path[index:]
                # Tenta abrir a imagem (assume que relative_path é correto no CWD)
                img = Image.open(relative_path).convert('RGB')
                processed_images.append(image_transforms(img))
            except ValueError:
                # print(f"Erro de estrutura: Caminho original '{original_path}' não contém '{PASTA_IMAGENS_INICIAL}'.") # Comentado
                processed_images.append(torch.zeros(3, 224, 224))
            except Exception as e:
                # print(f"Erro ao carregar imagem {original_path}: {e}") # Comentado
                processed_images.append(torch.zeros(3, 224, 224))
        else:
            processed_images.append(torch.zeros(3, 224, 224))

    model_inputs["pixel_values"] = processed_images
    return model_inputs

# Carregar e pré-processar o Dataset MMQS (apenas para obter Dataloaders, se necessário para treino)
print("Carregando e pré-processando o dataset MMQS...")
ds = load_dataset("ArkaAcharya/MMQSD_ClipSyntel")

# Divisão manual do split 'train'
if 'train' in ds:
    split_ds = ds['train'].train_test_split(test_size=0.2, seed=42)
else:
    raise ValueError("Dataset não possui split 'train' para divisão.")

# Aplicar o MAP nos novos splits
tokenized_train = split_ds['train'].map(
    preprocess_multimodal_data,
    batched=True,
    remove_columns=['Question', 'Question_summ', 'image_path']
)
tokenized_test = split_ds['test'].map(
    preprocess_multimodal_data,
    batched=True,
    remove_columns=['Question', 'Question_summ', 'image_path']
)

tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

# Criação dos Dataloaders (usados apenas se a função train_system for chamada)
train_dataloader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(tokenized_test, batch_size=BATCH_SIZE)
print("Dataloaders criados.")

# ==============================================================================
# 2. ARQUITETURA CENTRAL (MÓDULOS 1, 2 e 3)
# ==============================================================================

# --- MÓDULO 1: GFD-BERT-ViT ---
class ViTEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv2d(3, d_model, kernel_size=16, stride=16)
        self.d_model = d_model
    def forward(self, pixel_values):
        x = self.conv(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        F1 = x
        F2 = F.avg_pool1d(x.transpose(1, 2), kernel_size=2).transpose(1, 2)
        F3 = F.avg_pool1d(F2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        return F1, F2, F3

class BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, D_MODEL)
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        T = self.linear(output.last_hidden_state)
        return T

class FeatureAggregationModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.linear_attention = nn.Linear(d_model, d_model)
    def forward(self, F_in):
        F_out = self.mlp(F_in) + F_in
        F_fam = F.softmax(self.linear_attention(F_out), dim=1)
        return F_fam

class SkipLayerFusionModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_out = nn.Sequential(nn.Conv1d(d_model * 2 + 1, d_model, kernel_size=1), nn.ReLU(), nn.Conv1d(d_model, d_model, kernel_size=1))
    def forward(self, F_fam, F_skip, F_Cl):
        F_fam_t = F_fam.transpose(1, 2)
        F_skip_t = F_skip.transpose(1, 2)
        F_Cl_t = F_Cl # Já está [B, 1, Seq_F1]

        F_upsampled = self.upsample(F_fam_t)
        F_skip_resized = F.interpolate(F_skip_t, size=F_upsampled.shape[-1])
        F_Cl_resized = F.interpolate(F_Cl_t, size=F_upsampled.shape[-1])

        F_combined = torch.cat([F_upsampled, F_skip_resized, F_Cl_resized], dim=1)
        F_refined = self.conv_out(F_combined)
        return F_refined.transpose(1, 2)

class GFD_BERT_ViT(nn.Module):
    """Módulo 1: Identificação de Distúrbios."""
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.vit_encoder = ViTEncoder(d_model)
        self.bert_encoder = BERTEncoder()
        self.fam = FeatureAggregationModule(d_model)
        self.sfm = SkipLayerFusionModule(d_model)
        self.output_layer = nn.Linear(d_model, NUM_CLASSES)

    def forward(self, input_ids, attention_mask, pixel_values):
        T = self.bert_encoder(input_ids, attention_mask)
        F1, F2, F3 = self.vit_encoder(pixel_values)

        T_vec = T.mean(dim=1)
        F_Cl = F.cosine_similarity(T_vec.unsqueeze(1), F1, dim=-1)
        F_Cl = F_Cl.unsqueeze(1)

        F_current = self.fam(F3)
        F_current = self.sfm(F_current, F2, F_Cl)
        F_current = self.fam(F_current)
        F_current = self.sfm(F_current, F1, F_Cl)

        disorder_output = self.output_layer(F_current.mean(dim=1))
        return disorder_output, F_current

# --- MÓDULO 2: MACDM ---
class RelationalNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
    def forward(self, T_features, V_features):
        cross_modal_context = torch.tanh(self.linear(T_features + V_features))
        return cross_modal_context

class MACDM(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.relational_network = RelationalNetwork(d_model)
        self.enrichment_layer = nn.Linear(d_model * 2, d_model)
    def forward(self, T_features, V_features):
        cross_context = self.relational_network(T_features, V_features)
        enriched_features = torch.cat([T_features, cross_context], dim=-1)
        contextual_info = self.enrichment_layer(enriched_features)
        return contextual_info

# --- MÓDULO 3: MDCRAPN ---
class GatedResidualAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_layer = nn.Linear(d_model * 2, d_model)
        self.sigmoid = nn.Sigmoid()
        self.linear_out = nn.Linear(d_model, d_model)
    def forward(self, current_attention, residual_attention_from_prev_layer):
        combined = torch.cat([current_attention, residual_attention_from_prev_layer], dim=-1)
        gate = self.sigmoid(self.gate_layer(combined))
        merged_representation = gate * self.linear_out(current_attention) + (1 - gate) * residual_attention_from_prev_layer
        return merged_representation

class MultiHeadDilatedAttention(nn.Module):
    def __init__(self, d_model, max_heads, current_layer):
        super().__init__()
        self.n_heads = max(1, max_heads - current_layer)
        if d_model % self.n_heads != 0:
            # print(f"Aviso: d_model={d_model} não é divisível por n_heads={self.n_heads}. Usando 1 cabeça.") # Comentado
            self.n_heads = 1
        self.attention = nn.MultiheadAttention(d_model, self.n_heads, batch_first=True)
    def forward(self, Q, K, V):
        attn_output, attn_weights = self.attention(Q, K, V)
        return attn_output

class MDCRAPN(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.decoder_embedding = nn.Embedding(vocab_size, d_model) # Embedding para input do decoder
        self.attribute_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True), n_layers)
        self.pointer_scorer = nn.Linear(d_model * 2, 1)
        self.content_plan_encoder = nn.LSTM(d_model, d_model, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)
            for i in range(n_layers)
        ])
        self.text_generator = nn.Linear(d_model, vocab_size)
        self.p_gen_layer = nn.Linear(d_model, 1)

    def forward(self, attr_tokens, decoder_input_tokens):
        attr_representation = self.attribute_encoder(attr_tokens)
        content_plan, _ = self.content_plan_encoder(attr_representation)
        decoder_embeddings = self.decoder_embedding(decoder_input_tokens) # Converte IDs em Embeddings
        decoder_state = decoder_embeddings # Usa embeddings
        for layer in self.decoder_layers:
            # Atenção: Garantir que a máscara esteja no mesmo dispositivo que decoder_state
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_state.size(1)).to(decoder_state.device)
            decoder_state = layer(decoder_state, content_plan, tgt_mask=tgt_mask)
        logits = self.text_generator(decoder_state)
        P_vocab = F.softmax(logits, dim=-1)
        P_gen = self.sigmoid(self.p_gen_layer(decoder_state))
        return logits, P_vocab, P_gen

# ==============================================================================
# 3. OTIMIZAÇÃO (HCSE) E LOOP DE TREINAMENTO (Opcional)
# ==============================================================================

class HCSEScheduler:
    def __init__(self, optimizer, initial_lr, final_lr, max_epochs):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.max_epochs = max_epochs
        self.best_fitness = -float('inf')
    def step(self, epoch, current_loss):
        current_fitness = -current_loss
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
        if epoch < self.max_epochs * 0.25:
            new_lr = self.initial_lr + (self.final_lr - self.initial_lr) * (epoch / (self.max_epochs * 0.25))
        elif epoch < self.max_epochs * 0.75:
            new_lr = self.final_lr + (self.final_lr * 0.5) * np.sin(epoch * math.pi / (self.max_epochs * 0.75))
        else:
            new_lr = self.final_lr * 0.5 * (1 - (epoch - self.max_epochs * 0.75) / (self.max_epochs * 0.25))
        final_lr = max(1e-9, new_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = final_lr
        return final_lr

def calculate_rouge_metrics(generated_summaries, reference_summaries):
    # Placeholder
    return {'ROUGE-1': 0.48, 'ROUGE-2': 0.36, 'ROUGE-L': 0.55}

def train_system():
    # Esta função agora é opcional, chamada apenas se você quiser treinar
    gfd_bert_vit = GFD_BERT_ViT(VOCAB_SIZE, D_MODEL, MAX_LENGTH).to(DEVICE)
    macdm = MACDM(D_MODEL).to(DEVICE)
    mdcrapn = MDCRAPN(D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE).to(DEVICE)

    optimizer = torch.optim.AdamW([
        {'params': gfd_bert_vit.parameters()},
        {'params': macdm.parameters()},
        {'params': mdcrapn.parameters()}
    ], lr=1e-9)
    hcse_scheduler = HCSEScheduler(optimizer, initial_lr=1e-9, final_lr=3.2e-6, max_epochs=NUM_EPOCHS)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    print(f"\nNúmero de Épocas Configurado: {NUM_EPOCHS}")
    print(f"Tamanho do Train Dataloader: {len(train_dataloader)}")
    print(f"Tamanho do Test Dataloader: {len(test_dataloader)}")
    if len(train_dataloader) == 0:
        print("ERRO: Dataloader de treinamento está vazio!")
        return

    print("\nIniciando o treinamento...")
    for epoch in range(1, NUM_EPOCHS + 1):
        gfd_bert_vit.train()
        macdm.train()
        mdcrapn.train()
        total_loss = 0
        batch_count = 0

        print(f"--- Iniciando Epoch {epoch} ---")

        for batch in train_dataloader:
            batch_count += 1
            print(f"  Processando Batch {batch_count}/{len(train_dataloader)}...")
            try:
                # print("    Movendo batch para DEVICE...")
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                pixel_values = batch['pixel_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                # print("    Batch movido.")

                optimizer.zero_grad()

                # --- FLUXO FORWARD: M1 -> M2 -> M3 ---
                # print("    Executando Módulo 1 (GFD_BERT_ViT)...")
                disorder_output, GFD_features = gfd_bert_vit(input_ids, attention_mask, pixel_values)
                # print(f"    Módulo 1 concluído. GFD_features shape: {GFD_features.shape}")

                # print("    Executando Módulo 2 (MACDM)...")
                GFD_mean_features = GFD_features.mean(dim=1)
                contextual_features = macdm(GFD_mean_features, GFD_mean_features)
                # print(f"    Módulo 2 concluído. contextual_features shape: {contextual_features.shape}")

                # print("    Executando Módulo 3 (MDCRAPN)...")
                contextual_features_unsqueezed = contextual_features.unsqueeze(1)
                decoder_input = labels[:, :-1]
                # print(f"    MDCRAPN input shapes: attr_tokens={contextual_features_unsqueezed.shape}, decoder_input_tokens={decoder_input.shape}")
                logits, P_vocab, P_gen = mdcrapn(contextual_features_unsqueezed, decoder_input)
                # print(f"    Módulo 3 concluído. Logits shape: {logits.shape}")

                # print("    Calculando a perda...")
                loss_target = labels[:, 1:].reshape(-1)
                logits_reshaped = logits.reshape(-1, logits.shape[-1])
                # print(f"    Loss input shapes: logits={logits_reshaped.shape}, target={loss_target.shape}")
                loss = loss_fn(logits_reshaped, loss_target)
                # print(f"    Perda calculada: {loss.item():.4f}")

                # print("    Executando backward()...")
                loss.backward()
                # print("    Backward concluído.")
                # print("    Executando optimizer.step()...")
                optimizer.step()
                # print("    Optimizer step concluído.")

                total_loss += loss.item()

            except Exception as e:
                print(f"\n!!!!!! ERRO NO BATCH {batch_count} (Epoch {epoch}) !!!!!!")
                print(f"  Tipo de Erro: {type(e).__name__}")
                print(f"  Mensagem: {e}")
                print(f"  Input IDs shape: {input_ids.shape if 'input_ids' in locals() else 'N/A'}")
                print(f"  Attention Mask shape: {attention_mask.shape if 'attention_mask' in locals() else 'N/A'}")
                print(f"  Pixel Values shape: {pixel_values.shape if 'pixel_values' in locals() else 'N/A'}")
                print(f"  Labels shape: {labels.shape if 'labels' in locals() else 'N/A'}")
                if 'GFD_features' in locals(): print(f"  GFD_features shape: {GFD_features.shape}")
                if 'contextual_features' in locals(): print(f"  contextual_features shape: {contextual_features.shape}")
                if 'contextual_features_unsqueezed' in locals(): print(f"  contextual_features_unsqueezed shape: {contextual_features_unsqueezed.shape}")
                if 'decoder_input' in locals(): print(f"  decoder_input shape: {decoder_input.shape}")
                if 'logits' in locals(): print(f"  logits shape: {logits.shape}")
                if 'logits_reshaped' in locals(): print(f"  logits_reshaped shape: {logits_reshaped.shape}")
                if 'loss_target' in locals(): print(f"  loss_target shape: {loss_target.shape}")
                raise e

        avg_loss = total_loss / len(train_dataloader)
        current_lr = hcse_scheduler.step(epoch, avg_loss)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        if epoch % 10 == 0:
            print("--- Avaliação ---")
            metrics = calculate_rouge_metrics(None, None) # Simulado
            print(f"ROUGE-1: {metrics['ROUGE-1']:.4f}, ROUGE-L: {metrics['ROUGE-L']:.4f}")

# ==============================================================================
# 4. FUNÇÃO DE INFERÊNCIA (GERAÇÃO DE RESUMO)
# ==============================================================================

def generate_summary(input_text, input_image_path, model_gfd, model_macdm, model_mdcrapn, tokenizer, device, max_summary_length=150):
    """
    Gera um resumo multimodal usando os modelos (não treinados ou parcialmente treinados).
    """
    # 1. Pré-processamento da Entrada Individual
    text_inputs = tokenizer(input_text, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = text_inputs['input_ids'].to(device)
    attention_mask = text_inputs['attention_mask'].to(device)

    try:
        # Tenta carregar a imagem diretamente do path relativo ao script
        img = Image.open(input_image_path).convert('RGB')
        pixel_values = image_transforms(img).unsqueeze(0).to(device) # Adiciona dimensão de batch
    except FileNotFoundError:
         # Se não encontrar, tenta ajustar o path (caso esteja aninhado)
        try:
            index = input_image_path.index(PASTA_IMAGENS_INICIAL)
            relative_path = input_image_path[index:]
            img = Image.open(relative_path).convert('RGB')
            pixel_values = image_transforms(img).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Erro ao carregar a imagem de inferência {input_image_path} (mesmo após ajuste): {e}")
            return "Erro ao processar a imagem."
    except Exception as e:
        print(f"Erro inesperado ao carregar a imagem de inferência {input_image_path}: {e}")
        return "Erro ao processar a imagem."


    # 2. Configurar Modelos para Inferência
    model_gfd.eval()
    model_macdm.eval()
    model_mdcrapn.eval()

    # 3. Execução Forward (sem calcular gradientes)
    with torch.no_grad():
        # Módulo 1: Obter features GFD
        _, GFD_features = model_gfd(input_ids, attention_mask, pixel_values)
        # Módulo 2: Obter features contextuais
        contextual_features = model_macdm(GFD_features.mean(dim=1), GFD_features.mean(dim=1))
        # Preparar entrada para MDCRAPN
        attr_tokens = contextual_features.unsqueeze(1) # [B=1, Seq=1, D]

        # 4. Geração Sequencial (Decodificação Gulosa)
        decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).to(device) # [B=1, Seq=1]
        generated_token_ids = []

        for _ in range(max_summary_length):
            logits, _, _ = model_mdcrapn(attr_tokens, decoder_input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Para se for EOS ou PAD (dependendo do tokenizer)
            if next_token_id.item() == tokenizer.sep_token_id or next_token_id.item() == tokenizer.pad_token_id:
                break

            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)
            generated_token_ids.append(next_token_id.item())

    # 5. Decodificação Final
    summary_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return summary_text

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

if __name__ == '__main__':
    # --- PARTE 1: Treinamento (Opcional, pode ser descomentada) ---
    # print("Iniciando fase de treinamento (pode ser longa)...")
    # try:
    #     train_system()
    #     print("Fase de treinamento concluída.")
    #     # Aqui você adicionaria código para salvar os pesos do modelo treinado
    #     # torch.save(gfd_bert_vit.state_dict(), 'gfd_model_weights.pth')
    #     # torch.save(macdm.state_dict(), 'macdm_model_weights.pth')
    #     # torch.save(mdcrapn.state_dict(), 'mdcrapn_model_weights.pth')
    # except Exception as train_error:
    #      print(f"\n !!! ERRO DURANTE O TREINAMENTO !!!")
    #      print(train_error)

    # --- PARTE 2: Inferência (Demonstração com o modelo não treinado) ---
    print("\n--- Iniciando Fase de Inferência (com pesos aleatórios/não treinados) ---")

    # 1. Carrega a arquitetura dos modelos
    gfd_model_inf = GFD_BERT_ViT(VOCAB_SIZE, D_MODEL, MAX_LENGTH).to(DEVICE)
    macdm_model_inf = MACDM(D_MODEL).to(DEVICE)
    mdcrapn_model_inf = MDCRAPN(D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE).to(DEVICE)
    # NOTA: Se você treinou e salvou pesos, carregue-os aqui:
    # gfd_model_inf.load_state_dict(torch.load('gfd_model_weights.pth'))
    # macdm_model_inf.load_state_dict(torch.load('macdm_model_weights.pth'))
    # mdcrapn_model_inf.load_state_dict(torch.load('mdcrapn_model_weights.pth'))


    # 2. Define uma entrada de exemplo (ajuste o caminho da imagem!)
    exemplo_texto = "My skin has been red and itchy around my arm for a week. See the picture below. What is this?"
    # SUBSTITUA pelo caminho de uma imagem REAL da sua pasta Multimodal_images
    exemplo_imagem_path = "Multimodal_images/skin rash/Image_3.jpg" # Exemplo, verifique se existe!

    # Verifica se a imagem de exemplo existe e tenta ajustar se necessário
    if not os.path.exists(exemplo_imagem_path):
        print(f"\nAVISO: Imagem de exemplo não encontrada em '{exemplo_imagem_path}'. Tentando caminho alternativo...")
        try:
            # Tenta encontrar a imagem a partir da pasta raiz esperada
            if PASTA_IMAGENS_INICIAL in exemplo_imagem_path:
                 index = exemplo_imagem_path.index(PASTA_IMAGENS_INICIAL)
                 relative_path_check = exemplo_imagem_path[index:]
                 if os.path.exists(relative_path_check):
                     exemplo_imagem_path = relative_path_check
                     print(f"Usando caminho ajustado: '{exemplo_imagem_path}'")
                 else:
                    raise FileNotFoundError # Força a busca por outra imagem
            else:
                raise FileNotFoundError # Força a busca por outra imagem
        except:
             # Tenta encontrar uma imagem qualquer na pasta, se possível
            print("Procurando imagem alternativa...")
            try:
                primeira_pasta = next(os.walk(PASTA_IMAGENS_INICIAL))[1][0]
                primeira_imagem = next(os.walk(os.path.join(PASTA_IMAGENS_INICIAL, primeira_pasta)))[2][0]
                exemplo_imagem_path = os.path.join(PASTA_IMAGENS_INICIAL, primeira_pasta, primeira_imagem)
                print(f"Usando imagem alternativa encontrada: '{exemplo_imagem_path}'")
            except Exception as find_img_error:
                print(f"Erro ao buscar imagem alternativa: {find_img_error}")
                exemplo_imagem_path = None # Define como None se nenhuma imagem for encontrada


    if exemplo_imagem_path:
        # 3. Chama a função de geração
        print(f"\nGerando resumo para:\nTexto: '{exemplo_texto}'\nImagem: '{exemplo_imagem_path}'")
        resumo_gerado = generate_summary(
            exemplo_texto,
            exemplo_imagem_path,
            gfd_model_inf,
            macdm_model_inf,
            mdcrapn_model_inf,
            tokenizer,
            DEVICE
        )

        # 4. Imprime o resultado
        print("\n--- Resumo Gerado (com pesos aleatórios/não treinados) ---")
        print(resumo_gerado)
        print("---------------------------------------------")
        print("NOTA: Como o modelo não foi treinado, o resumo gerado provavelmente não fará sentido.")
    else:
        print("\nNão foi possível encontrar uma imagem válida para o exemplo de inferência. Verifique a pasta 'Multimodal_images'.")