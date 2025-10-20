import streamlit as st
import torch
import torch.nn as nn
import sentencepiece as spm
import json
import math
from pathlib import Path
import copy
import time
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Urdu Transformer Chatbot",
    layout="wide"
)

st.markdown("""
<style>
    .urdu-text {
        direction: rtl;
        text-align: right;
        font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', 'Nafees Web Naskh', Arial, sans-serif;
        font-size: 18px;
        line-height: 2;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
        animation: slideInRight 0.5s ease-out;
    }
    .bot-message {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
        animation: slideInRight 0.5s ease-out;
    }
    .typing-animation {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
        animation: slideInRight 0.5s ease-out;
    }
    .typing-dots {
        display: inline-block;
        direction: ltr;
    }
    .typing-dots span {
        animation: blink 1.4s infinite;
        font-size: 20px;
        margin: 0 2px;
    }
    .typing-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }
    .typing-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }
    @keyframes blink {
        0%, 60%, 100% { opacity: 0; }
        30% { opacity: 1; }
    }
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
    }
    .example-button {
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Model Architecture ====================

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# ==================== Helper Functions ====================

def get_sentinel_ids(sp, required=2):
    SENTINEL_PIECES = ("<extra_id_0>", "<extra_id_1>", "<extra_id_2>")
    ids = []
    for piece in SENTINEL_PIECES:
        pid = sp.piece_to_id(piece)
        if pid != sp.unk_id():
            ids.append(pid)
    if len(ids) < required:
        return []
    return ids

def _collect_special_token_ids(sp):
    specials = {sp.pad_id(), sp.bos_id(), sp.eos_id()}
    unk_id = sp.unk_id()
    if unk_id is not None and unk_id >= 0:
        specials.add(unk_id)
    sentinel_ids = set()
    try:
        sentinel_ids = set(get_sentinel_ids(sp, required=1))
        specials.update(sentinel_ids)
    except:
        sentinel_ids = set()
    return specials, sentinel_ids

def _apply_repetition_penalty(logits, generated_ids, penalty, special_token_ids):
    if penalty == 1.0 or not generated_ids:
        return logits
    safe_specials = special_token_ids or set()
    unique_ids = set(generated_ids) - safe_specials
    for token_id in unique_ids:
        value = logits[0, token_id]
        if value < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits

def _enforce_no_repeat_ngram(logits, generated_ids, ngram_size):
    if ngram_size < 2 or len(generated_ids) < ngram_size - 1:
        return logits
    ngram_dict = {}
    for index in range(len(generated_ids) - ngram_size + 1):
        prefix = tuple(generated_ids[index:index + ngram_size - 1])
        next_token = generated_ids[index + ngram_size - 1]
        ngram_dict.setdefault(prefix, set()).add(next_token)
    prefix = tuple(generated_ids[-(ngram_size - 1):])
    banned = ngram_dict.get(prefix)
    if banned:
        logits[0, list(banned)] = float('-inf')
    return logits

def _top_k_filtering(logits, top_k):
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    top_values, _ = torch.topk(logits, top_k)
    threshold = top_values[..., -1, None]
    logits[logits < threshold] = float('-inf')
    return logits

def _sample_next_token(
    logits,
    generated_ids,
    *,
    temperature,
    special_token_ids=None,
    decode_blocked_ids=None,
):
    filtered = logits.clone()
    if decode_blocked_ids:
        filtered[:, list(decode_blocked_ids)] = float('-inf')

    if not torch.isfinite(filtered).any():
        filtered = logits

    if temperature is not None and temperature > 0 and temperature != 1.0:
        filtered = filtered / temperature

    probs = torch.softmax(filtered, dim=-1)
    if torch.isnan(probs).any():
        probs = torch.softmax(logits, dim=-1)

    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()

# ==================== Generation Functions ====================

def greedy_generate(model, sp, text, max_len=60, special_token_ids=None, decode_blocked_ids=None):
    model.eval()
    device = next(model.parameters()).device
    
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model
    
    with torch.no_grad():
        encoded = sp.encode(text, out_type=int)
        src = torch.tensor([sp.bos_id()] + encoded + [sp.eos_id()], device=device).unsqueeze(0)
        src_mask = (src != sp.pad_id()).unsqueeze(-2)
        memory = actual_model.encode(src, src_mask)

        ys = torch.tensor([[sp.bos_id()]], device=device)
        generated_ids = []
        
        for _ in range(max_len):
            tgt_mask = subsequent_mask(ys.size(1)).to(device)
            out = actual_model.decode(memory, src_mask, ys, tgt_mask)
            next_log_probs = actual_model.generator(out[:, -1])
            
            if decode_blocked_ids:
                next_log_probs[:, list(decode_blocked_ids)] = float('-inf')
            
            next_token = next_log_probs.argmax(dim=-1).item()
            
            if next_token == sp.eos_id():
                break

            ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
            generated_ids.append(next_token)

    cleaned = []
    for idx in generated_ids:
        if idx == sp.eos_id():
            break
        if special_token_ids and idx in special_token_ids:
            continue
        cleaned.append(idx)
    
    return sp.decode(cleaned).strip() if cleaned else ""

def beam_search_generate(model, sp, text, beam_width=3, max_len=60, special_token_ids=None, decode_blocked_ids=None):
    model.eval()
    device = next(model.parameters()).device
    
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model
    
    with torch.no_grad():
        encoded = sp.encode(text, out_type=int)
        src = torch.tensor([sp.bos_id()] + encoded + [sp.eos_id()], device=device).unsqueeze(0)
        src_mask = (src != sp.pad_id()).unsqueeze(-2)
        memory = actual_model.encode(src, src_mask)

        beams = [([sp.bos_id()], 0.0)]
        
        for _ in range(max_len):
            candidates = []
            
            for seq, score in beams:
                if seq[-1] == sp.eos_id():
                    candidates.append((seq, score))
                    continue
                
                ys = torch.tensor([seq], device=device)
                tgt_mask = subsequent_mask(ys.size(1)).to(device)
                out = actual_model.decode(memory, src_mask, ys, tgt_mask)
                next_log_probs = actual_model.generator(out[:, -1])
                
                if decode_blocked_ids:
                    next_log_probs[:, list(decode_blocked_ids)] = float('-inf')
                
                top_probs, top_indices = torch.topk(next_log_probs[0], beam_width)
                
                for prob, idx in zip(top_probs, top_indices):
                    new_seq = seq + [idx.item()]
                    new_score = score + prob.item()
                    candidates.append((new_seq, new_score))
            
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            if all(seq[-1] == sp.eos_id() for seq, _ in beams):
                break
        
        best_seq, _ = beams[0]
        generated_ids = best_seq[1:]
        
    cleaned = []
    for idx in generated_ids:
        if idx == sp.eos_id():
            break
        if special_token_ids and idx in special_token_ids:
            continue
        cleaned.append(idx)
    
    return sp.decode(cleaned).strip() if cleaned else ""

def sampling_generate(model, sp, text, max_len=60, temperature=0.9, 
                     special_token_ids=None, decode_blocked_ids=None):
    model.eval()
    device = next(model.parameters()).device
    
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model
    
    with torch.no_grad():
        encoded = sp.encode(text, out_type=int)
        src = torch.tensor([sp.bos_id()] + encoded + [sp.eos_id()], device=device).unsqueeze(0)
        src_mask = (src != sp.pad_id()).unsqueeze(-2)
        memory = actual_model.encode(src, src_mask)

        ys = torch.tensor([[sp.bos_id()]], device=device)
        generated_ids = []
        
        for _ in range(max_len):
            tgt_mask = subsequent_mask(ys.size(1)).to(device)
            out = actual_model.decode(memory, src_mask, ys, tgt_mask)
            next_log_probs = actual_model.generator(out[:, -1])
            
            next_token = _sample_next_token(
                next_log_probs,
                generated_ids,
                temperature=temperature,
                special_token_ids=special_token_ids,
                decode_blocked_ids=decode_blocked_ids,
            )
            
            if next_token == sp.eos_id():
                break

            ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
            generated_ids.append(next_token)

    cleaned = []
    for idx in generated_ids:
        if idx == sp.eos_id():
            break
        if special_token_ids and idx in special_token_ids:
            continue
        cleaned.append(idx)
    
    return sp.decode(cleaned).strip() if cleaned else ""

def streaming_generate(model, sp, text, max_len=60, temperature=0.9, 
                      special_token_ids=None, decode_blocked_ids=None):
    model.eval()
    device = next(model.parameters()).device
    
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model
    
    with torch.no_grad():
        encoded = sp.encode(text, out_type=int)
        src = torch.tensor([sp.bos_id()] + encoded + [sp.eos_id()], device=device).unsqueeze(0)
        src_mask = (src != sp.pad_id()).unsqueeze(-2)
        memory = actual_model.encode(src, src_mask)

        ys = torch.tensor([[sp.bos_id()]], device=device)
        generated_ids = []
        
        for _ in range(max_len):
            tgt_mask = subsequent_mask(ys.size(1)).to(device)
            out = actual_model.decode(memory, src_mask, ys, tgt_mask)
            next_log_probs = actual_model.generator(out[:, -1])
            
            next_token = _sample_next_token(
                next_log_probs,
                generated_ids,
                temperature=temperature,
                special_token_ids=special_token_ids,
                decode_blocked_ids=decode_blocked_ids,
            )
            
            if next_token == sp.eos_id():
                break

            ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
            generated_ids.append(next_token)
            
            cleaned_ids = []
            for idx in generated_ids:
                if special_token_ids and idx in special_token_ids:
                    continue
                cleaned_ids.append(idx)
            
            yield sp.decode(cleaned_ids).strip()

# ==================== Load Model ====================

@st.cache_resource
def load_model():
    """Load the trained model and tokenizer from Hugging Face Hub."""
    
    hf_repo_id = "hurairamuzammal/transformer_NLP_A02"
    config_filename = "cloud_config.json"

    # Download cloud_config.json from Hugging Face Hub
    config_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=config_filename
    )

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    model_config = config["model_config"]
    # Get filenames from cloud_config.json
    model_filename = config["model_bundle"]
    tokenizer_filename = config["tokenizer_file"]

    # Download tokenizer from Hugging Face Hub
    tokenizer_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=tokenizer_filename
    )
    
    # Download model file from Hugging Face Hub
    model_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=model_filename
    )
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(
        src_vocab=model_config["src_vocab"],
        tgt_vocab=model_config["tgt_vocab"],
        N=model_config["N"],
        d_model=model_config["d_model"],
        d_ff=model_config["d_ff"],
        h=model_config["h"],
        dropout=model_config["dropout"]
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    fixed_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("decoder.layer.", "decoder.layers.")
        fixed_state_dict[new_key] = value
    
    model.load_state_dict(fixed_state_dict)
    
    model.to(device)
    model.eval()
    
    special_token_ids, sentinel_token_ids = _collect_special_token_ids(sp)
    decode_blocked_ids = special_token_ids - sentinel_token_ids
    
    return model, sp, special_token_ids, decode_blocked_ids

# ==================== Streamlit UI ====================

def main():
    st.title("Urdu Transformer Chatbot")
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "current_input" not in st.session_state:
        st.session_state.current_input = ""
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = ""
    if "trigger_generation" not in st.session_state:
        st.session_state.trigger_generation = False
    
    try:
        with st.spinner("Loading model..."):
            model, sp, special_token_ids, decode_blocked_ids = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    st.sidebar.header("Settings")
    
    decoding_strategy = st.sidebar.selectbox(
        "Decoding Strategy",
        ["Greedy", "Beam Search", "Sampling"],
        index=2
    )
    
    max_length = st.sidebar.slider("Max Length", 20, 100, 60)
    
    if decoding_strategy == "Beam Search":
        beam_width = st.sidebar.slider("Beam Width", 2, 5, 3)
    elif decoding_strategy == "Sampling":
        temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.9, 0.1)
    
    if st.sidebar.button("Clear History"):
        st.session_state.conversation_history = []
        st.session_state.current_input = ""
        st.session_state.is_generating = False
        st.session_state.pending_input = ""
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.conversation_history:
        st.markdown("### Conversation History")
        for i, (user_input, bot_response, strategy) in enumerate(st.session_state.conversation_history):
            st.markdown(f"""
            <div class="user-message">
                <strong>You ({strategy}):</strong><br>
                <span class="urdu-text">{user_input}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="bot-message">
                <strong>Bot:</strong><br>
                <span class="urdu-text">{bot_response}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    st.markdown("### Example Prompts")
    
    example_prompts = [
        "ایک بہادر جرنیل دوسرے کی قدر جانتا ہے",
        "انسان پر لازم ہے کہ وہ اللہ تعالی کی وحدانیت کا اقرار کرے",
        "میں اسکول جا رہا ہوں",
        "یہ کتاب بہت اچھی ہے",
        "آج موسم بہت خوشگوار ہے",
        "میں نے کھانا کھایا"
    ]
    
    cols = st.columns(3)
    for idx, example in enumerate(example_prompts):
        with cols[idx % 3]:
            if st.button(f"{example[:25]}..." if len(example) > 25 else f"{example}", 
                        key=f"example_{idx}", 
                        use_container_width=True,
                        disabled=st.session_state.is_generating):
                st.session_state.current_input = example
                st.session_state.trigger_generation = True
    
    st.markdown("---")
    
    st.markdown("### Enter Your Message")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type in Urdu",
            value=st.session_state.current_input,
            placeholder="یہاں اردو میں لکھیں...",
            key="user_input_field",
            label_visibility="collapsed",
            disabled=st.session_state.is_generating
        )
        if user_input != st.session_state.current_input:
            st.session_state.current_input = user_input
    
    with col2:
        generate_button = st.button("Generate", 
                                    use_container_width=True,
                                    disabled=st.session_state.is_generating or not user_input.strip())
    
    if (generate_button and user_input.strip()) or (st.session_state.trigger_generation and st.session_state.current_input.strip()):
        
        if st.session_state.trigger_generation:
            user_input = st.session_state.current_input
            st.session_state.trigger_generation = False

        st.session_state.pending_input = user_input
        st.session_state.current_input = ""
        st.session_state.is_generating = True
        
        st.markdown(f"""
        <div class="user-message">
            <strong>You ({decoding_strategy}):</strong><br>
            <span class="urdu-text">{st.session_state.pending_input}</span>
        </div>
        """, unsafe_allow_html=True)
        
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            response_placeholder.markdown("""
            <div class="typing-animation">
                <strong>Bot:</strong><br>
                <span class="urdu-text">جواب تیار ہو رہا ہے</span>
                <span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>
            </div>
            """, unsafe_allow_html=True)
            
            if decoding_strategy == "Sampling":
                response_generator = streaming_generate(
                    model, sp, st.session_state.pending_input,
                    max_len=max_length,
                    temperature=temperature,
                    special_token_ids=special_token_ids,
                    decode_blocked_ids=decode_blocked_ids
                )
                for decoded_text in response_generator:
                    full_response = decoded_text
                    response_placeholder.markdown(f"""
                    <div class="bot-message">
                        <strong>Bot:</strong><br>
                        <span class="urdu-text">{full_response}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.05)
            else:
                if decoding_strategy == "Greedy":
                    full_response = greedy_generate(
                        model, sp, st.session_state.pending_input, 
                        max_len=max_length,
                        special_token_ids=special_token_ids,
                        decode_blocked_ids=decode_blocked_ids
                    )
                else:
                    full_response = beam_search_generate(
                        model, sp, st.session_state.pending_input, 
                        beam_width=beam_width,
                        max_len=max_length,
                        special_token_ids=special_token_ids,
                        decode_blocked_ids=decode_blocked_ids
                    )
                
                response_placeholder.markdown(f"""
                <div class="bot-message">
                    <strong>Bot:</strong><br>
                    <span class="urdu-text">{full_response}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.session_state.conversation_history.append((st.session_state.pending_input, full_response, decoding_strategy))
            st.session_state.is_generating = False
            st.session_state.pending_input = ""
            
            time.sleep(0.5)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.session_state.is_generating = False
            st.session_state.pending_input = ""
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 14px;'>
        PyTorch Transformer for Urdu
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
