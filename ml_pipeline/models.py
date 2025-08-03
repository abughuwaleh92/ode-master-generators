"""
Neural Network Models for ODE Generation and Analysis

This module contains various ML architectures for learning from ODE datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ODEPatternNet(nn.Module):
    """
    Feed-forward neural network for learning ODE patterns
    
    Tasks:
    - Verification prediction
    - Complexity estimation
    - Generator classification
    - Parameter optimization
    """
    
    def __init__(self, 
                 n_numeric_features: int = 12,
                 n_generators: int = 11,
                 n_functions: int = 34,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.n_numeric_features = n_numeric_features
        self.n_generators = n_generators
        self.n_functions = n_functions
        
        # Embeddings for categorical features
        self.generator_embed = nn.Embedding(n_generators, 32)
        self.function_embed = nn.Embedding(n_functions, 64)
        
        # Calculate input dimension
        input_dim = n_numeric_features + 32 + 64  # numeric + embeddings
        
        # Build hidden layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Task-specific heads
        final_dim = hidden_dims[-1]
        self.verification_head = nn.Linear(final_dim, 1)
        self.complexity_head = nn.Linear(final_dim, 1)
        self.generator_head = nn.Linear(final_dim, n_generators)
        self.parameter_head = nn.Linear(final_dim, 6)  # alpha, beta, M, q, v, a
        
        # Attention mechanism for feature importance
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim),
            nn.Softmax(dim=1)
        )
        
    def forward(self, numeric_features, generator_id, function_id):
        # Get embeddings
        gen_embed = self.generator_embed(generator_id)
        func_embed = self.function_embed(function_id)
        
        # Concatenate all features
        features = torch.cat([numeric_features, gen_embed, func_embed], dim=1)
        
        # Apply attention
        attention_weights = self.feature_attention(features)
        features = features * attention_weights
        
        # Process through network
        hidden = self.feature_net(features)
        
        # Multi-task outputs
        outputs = {
            'verification': torch.sigmoid(self.verification_head(hidden)),
            'complexity': self.complexity_head(hidden),
            'generator_logits': self.generator_head(hidden),
            'parameters': self.parameter_head(hidden),
            'attention_weights': attention_weights
        }
        
        return outputs


class ODETransformer(nn.Module):
    """
    Transformer-based model for ODE sequence modeling
    
    Processes ODE as sequence of tokens for generation and understanding
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_seq_length: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        
        # Output heads
        self.generation_head = nn.Linear(d_model, vocab_size)
        self.classification_head = nn.Linear(d_model, 11)  # 11 generators
        self.verification_head = nn.Linear(d_model, 1)
        
        # Layer normalization
        self.ln = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = self.dropout(token_embeds + position_embeds)
        
        # Create attention mask for padding
        if attention_mask is not None:
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
        
        # Transformer encoding
        hidden_states = self.transformer(
            embeddings.transpose(0, 1),
            src_key_padding_mask=attention_mask
        ).transpose(0, 1)
        
        # Apply layer norm
        hidden_states = self.ln(hidden_states)
        
        # Get outputs
        outputs = {
            'hidden_states': hidden_states,
            'generation_logits': self.generation_head(hidden_states),
            'generator_logits': self.classification_head(hidden_states[:, 0, :]),  # CLS token
            'verification': torch.sigmoid(self.verification_head(hidden_states[:, 0, :]))
        }
        
        return outputs


class ODEVAE(nn.Module):
    """
    Variational Autoencoder for ODE generation
    
    Learns latent representations of ODEs for generation and interpolation
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 latent_dim: int = 64,
                 n_generators: int = 11,
                 n_functions: int = 34):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim + n_generators + n_functions, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Conditional embeddings
        self.generator_embed = nn.Embedding(n_generators, 16)
        self.function_embed = nn.Embedding(n_functions, 16)
        
    def encode(self, x):
        """Encode to latent space"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z, generator_id, function_id):
        """Decode from latent space with conditioning"""
        # Get condition embeddings
        gen_embed = self.generator_embed(generator_id)
        func_embed = self.function_embed(function_id)
        
        # Concatenate with latent code
        decoder_input = torch.cat([z, gen_embed, func_embed], dim=1)
        
        # Decode
        h = self.decoder_input(decoder_input)
        return self.decoder(h)
    
    def forward(self, x, generator_id, function_id):
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decode(z, generator_id, function_id)
        
        return {
            'reconstruction': recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def generate(self, generator_id, function_id, n_samples=1):
        """Generate new ODEs"""
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(n_samples, self.latent_dim)
            
            # Decode
            samples = self.decode(z, generator_id, function_id)
            
        return samples


class ODELanguageModel(nn.Module):
    """
    Language model specifically designed for ODE generation
    
    Based on GPT-2 architecture with ODE-specific modifications
    """
    
    def __init__(self,
                 vocab_size: int,
                 n_positions: int = 512,
                 n_embd: int = 768,
                 n_layer: int = 12,
                 n_head: int = 12):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        
        # Embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)  # token embeddings
        self.wpe = nn.Embedding(n_positions, n_embd)  # position embeddings
        
        # ODE-specific embeddings
        self.generator_embed = nn.Embedding(11, n_embd)
        self.function_embed = nn.Embedding(34, n_embd)
        
        # Transformer blocks
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Output head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.wte.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, generator_id=None, function_id=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.wte(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_embeds = self.wpe(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        
        # Add ODE-specific embeddings if provided
        if generator_id is not None:
            gen_embed = self.generator_embed(generator_id).unsqueeze(1)
            hidden_states = hidden_states + gen_embed
        
        if function_id is not None:
            func_embed = self.function_embed(function_id).unsqueeze(1)
            hidden_states = hidden_states + func_embed
        
        hidden_states = self.drop(hidden_states)
        
        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        return {
            'logits': lm_logits,
            'hidden_states': hidden_states
        }
    
    def generate(self, 
                 input_ids, 
                 generator_id=None, 
                 function_id=None,
                 max_length=100,
                 temperature=1.0,
                 top_k=50,
                 top_p=0.95):
        """Generate ODE text"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self(input_ids, generator_id, function_id)
                logits = outputs['logits'][:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check for EOS token (should be defined in tokenizer)
                if next_token.item() == 2:  # Assuming 2 is EOS
                    break
        
        return input_ids


class TransformerBlock(nn.Module):
    """Transformer block for language model"""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, attention_mask=None):
        # Self-attention
        attn_output, _ = self.attn(
            self.ln_1(x), 
            self.ln_1(x), 
            self.ln_1(x),
            attn_mask=attention_mask
        )
        x = x + attn_output
        
        # Feed-forward
        x = x + self.mlp(self.ln_2(x))
        
        return x


class ODEDiscriminator(nn.Module):
    """
    Discriminator for adversarial training
    
    Distinguishes between real and generated ODEs
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.model(x))


# Model registry
MODEL_REGISTRY = {
    'pattern_net': ODEPatternNet,
    'transformer': ODETransformer,
    'vae': ODEVAE,
    'language_model': ODELanguageModel,
    'discriminator': ODEDiscriminator
}


def get_model(model_name: str, **kwargs):
    """Get model by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MODEL_REGISTRY[model_name](**kwargs)