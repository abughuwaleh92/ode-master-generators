"""
Machine Learning Pipeline for ODE Generation
Train various ML models to learn patterns and generate new ODEs
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config

logger = logging.getLogger(__name__)

class ODEDataset(Dataset):
    """PyTorch dataset for ODE data"""
    
    def __init__(self, features_df: pd.DataFrame, tokenizer=None):
        self.features_df = features_df
        self.tokenizer = tokenizer
        
        # Prepare encoders
        self.generator_encoder = LabelEncoder()
        self.function_encoder = LabelEncoder()
        
        # Fit encoders
        self.generator_encoder.fit(features_df['generator_name'])
        self.function_encoder.fit(features_df['function_name'])
        
    def __len__(self):
        return len(self.features_df)
    
    def __getitem__(self, idx):
        row = self.features_df.iloc[idx]
        
        # Numeric features
        numeric_features = torch.tensor([
            row['complexity_score'],
            row['operation_count'],
            row['atom_count'],
            row['symbol_count'],
            row['pow_deriv_max'],
            row['pow_yprime'],
            row['has_pantograph'],
            row['alpha'],
            row['beta'],
            row['M'],
            row.get('q', 2),
            row.get('v', 3)
        ], dtype=torch.float32)
        
        # Categorical features
        generator_id = self.generator_encoder.transform([row['generator_name']])[0]
        function_id = self.function_encoder.transform([row['function_name']])[0]
        
        # Target
        verified = torch.tensor(row['verified'], dtype=torch.float32)
        
        # ODE text (for language models)
        ode_text = row['ode_symbolic']
        solution_text = row['solution_symbolic']
        
        return {
            'numeric_features': numeric_features,
            'generator_id': generator_id,
            'function_id': function_id,
            'ode_text': ode_text,
            'solution_text': solution_text,
            'verified': verified
        }

class ODEPatternNet(nn.Module):
    """Neural network for learning ODE patterns"""
    
    def __init__(self, 
                 n_numeric_features: int,
                 n_generators: int,
                 n_functions: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        # Embeddings for categorical features
        self.generator_embed = nn.Embedding(n_generators, 32)
        self.function_embed = nn.Embedding(n_functions, 64)
        
        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(n_numeric_features + 32 + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Heads for different tasks
        self.verification_head = nn.Linear(hidden_dim, 1)
        self.complexity_head = nn.Linear(hidden_dim, 1)
        self.generator_head = nn.Linear(hidden_dim, n_generators)
        
    def forward(self, numeric_features, generator_id, function_id):
        # Get embeddings
        gen_embed = self.generator_embed(generator_id)
        func_embed = self.function_embed(function_id)
        
        # Concatenate all features
        features = torch.cat([numeric_features, gen_embed, func_embed], dim=1)
        
        # Process features
        hidden = self.feature_net(features)
        
        # Task outputs
        verification_prob = torch.sigmoid(self.verification_head(hidden))
        complexity_pred = self.complexity_head(hidden)
        generator_logits = self.generator_head(hidden)
        
        return {
            'verification': verification_prob,
            'complexity': complexity_pred,
            'generator': generator_logits
        }

class ODELanguageModel:
    """Fine-tune language model for ODE generation"""
    
    def __init__(self, model_name: str = 'gpt2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        
        # Add special tokens
        special_tokens = {
            'pad_token': '<pad>',
            'additional_special_tokens': [
                '<ode>', '</ode>',
                '<solution>', '</solution>',
                '<generator>', '</generator>',
                '<function>', '</function>'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Generation head
        self.generation_head = nn.Linear(
            self.model.config.hidden_size, 
            len(self.tokenizer)
        ).to(self.device)
        
        self.model.to(self.device)
        
    def prepare_text(self, ode: str, solution: str, generator: str, function: str) -> str:
        """Prepare text for training"""
        return (
            f"<generator>{generator}</generator>"
            f"<function>{function}</function>"
            f"<ode>{ode}</ode>"
            f"<solution>{solution}</solution>"
        )
    
    def train_step(self, batch):
        """Single training step"""
        inputs = self.tokenizer(
            batch['text'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        
        # Generate predictions
        logits = self.generation_head(hidden_states)
        
        # Compute loss (next token prediction)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs['input_ids'][..., 1:].contiguous()
        
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def generate_ode(self, generator: str, function: str, max_length: int = 200):
        """Generate new ODE given generator and function"""
        prompt = f"<generator>{generator}</generator><function>{function}</function><ode>"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.encode('</ode>')[0]
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract ODE
        if '<ode>' in generated_text and '</ode>' in generated_text:
            ode = generated_text.split('<ode>')[1].split('</ode>')[0]
            return ode
        
        return None

class ODEGeneratorTrainer:
    """Main trainer class for ODE generation models"""
    
    def __init__(self, dataset_path: str, features_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.features_path = features_path
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load ODE dataset"""
        logger.info("Loading ODE dataset...")
        
        # Load JSONL dataset
        odes = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    odes.append(json.loads(line))
        
        self.df = pd.DataFrame(odes)
        logger.info(f"Loaded {len(self.df)} ODEs")
        
        # Load features if available
        if self.features_path and Path(self.features_path).exists():
            self.features_df = pd.read_parquet(self.features_path)
        else:
            # Extract features from raw data
            self.features_df = self._extract_features()
    
    def _extract_features(self) -> pd.DataFrame:
        """Extract features from raw ODE data"""
        features = []
        
        for _, row in self.df.iterrows():
            feature_dict = {
                'id': row['id'],
                'generator_name': row['generator_name'],
                'function_name': row['function_name'],
                'generator_type': row['generator_type'],
                'complexity_score': row['complexity_score'],
                'operation_count': row['operation_count'],
                'atom_count': row['atom_count'],
                'symbol_count': row['symbol_count'],
                'verified': row['verified'],
                'has_pantograph': row['has_pantograph'],
                'ode_symbolic': row['ode_symbolic'],
                'solution_symbolic': row['solution_symbolic']
            }
            
            # Add parameters
            for param in ['alpha', 'beta', 'M', 'q', 'v', 'a']:
                if param in row['parameters']:
                    feature_dict[param] = row['parameters'][param]
            
            # Add nonlinearity metrics
            if 'nonlinearity_metrics' in row and row['nonlinearity_metrics']:
                metrics = row['nonlinearity_metrics']
                feature_dict['pow_deriv_max'] = metrics.get('pow_deriv_max', 1)
                feature_dict['pow_yprime'] = metrics.get('pow_yprime', 1)
            else:
                feature_dict['pow_deriv_max'] = 1
                feature_dict['pow_yprime'] = 1
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def train_pattern_network(self, epochs: int = 50, batch_size: int = 32):
        """Train pattern recognition network"""
        logger.info("Training pattern recognition network...")
        
        # Prepare dataset
        dataset = ODEDataset(self.features_df)
        
        # Split data
        train_idx, val_idx = train_test_split(
            range(len(dataset)), 
            test_size=0.2, 
            random_state=42
        )
        
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create model
        model = ODEPatternNet(
            n_numeric_features=12,
            n_generators=len(dataset.generator_encoder.classes_),
            n_functions=len(dataset.function_encoder.classes_)
        )
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(
                    batch['numeric_features'],
                    batch['generator_id'],
                    batch['function_id']
                )
                
                # Compute losses
                verification_loss = nn.BCELoss()(
                    outputs['verification'].squeeze(),
                    batch['verified']
                )
                
                complexity_loss = nn.MSELoss()(
                    outputs['complexity'].squeeze(),
                    batch['numeric_features'][:, 0]  # complexity_score
                )
                
                generator_loss = nn.CrossEntropyLoss()(
                    outputs['generator'],
                    batch['generator_id']
                )
                
                # Combined loss
                loss = verification_loss + 0.1 * complexity_loss + 0.1 * generator_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            correct_verifications = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        batch['numeric_features'],
                        batch['generator_id'],
                        batch['function_id']
                    )
                    
                    # Verification accuracy
                    preds = (outputs['verification'].squeeze() > 0.5).float()
                    correct_verifications += (preds == batch['verified']).sum().item()
                    total += len(batch['verified'])
            
            val_accuracy = correct_verifications / total
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss/len(train_loader):.4f} - "
                f"Val Accuracy: {val_accuracy:.4f}"
            )
        
        # Save model
        torch.save(model.state_dict(), 'ode_pattern_model.pth')
        logger.info("Model saved to ode_pattern_model.pth")
        
        return model
    
    def train_language_model(self, epochs: int = 10, batch_size: int = 8):
        """Train language model for ODE generation"""
        logger.info("Training language model for ODE generation...")
        
        # Prepare language model
        lm = ODELanguageModel()
        
        # Prepare training data
        texts = []
        for _, row in self.features_df.iterrows():
            text = lm.prepare_text(
                ode=row['ode_symbolic'],
                solution=row['solution_symbolic'],
                generator=row['generator_name'],
                function=row['function_name']
            )
            texts.append(text)
        
        # Training loop (simplified)
        optimizer = torch.optim.AdamW(
            list(lm.model.parameters()) + list(lm.generation_head.parameters()),
            lr=5e-5
        )
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Create batch
                batch = {'text': batch_texts}
                
                # Training step
                optimizer.zero_grad()
                loss = lm.train_step(batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(texts) / batch_size)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Save model
        torch.save({
            'model_state': lm.model.state_dict(),
            'generation_head': lm.generation_head.state_dict()
        }, 'ode_language_model.pth')
        
        logger.info("Language model saved")
        
        return lm
    
    def analyze_patterns(self):
        """Analyze patterns in the dataset"""
        logger.info("Analyzing ODE patterns...")
        
        # Verification patterns
        print("\nVerification Success by Generator:")
        verification_by_gen = self.features_df.groupby('generator_name')['verified'].agg(['mean', 'count'])
        print(verification_by_gen)
        
        # Complexity patterns
        print("\nComplexity Distribution:")
        complexity_stats = self.features_df['complexity_score'].describe()
        print(complexity_stats)
        
        # Function-Generator combinations
        print("\nMost successful Function-Generator combinations:")
        success_combos = self.features_df[self.features_df['verified']].groupby(
            ['generator_name', 'function_name']
        ).size().sort_values(ascending=False).head(10)
        print(success_combos)
        
        # Parameter influence
        print("\nParameter influence on verification:")
        param_cols = ['alpha', 'beta', 'M']
        for param in param_cols:
            if param in self.features_df.columns:
                verified_mean = self.features_df[self.features_df['verified']][param].mean()
                failed_mean = self.features_df[~self.features_df['verified']][param].mean()
                print(f"{param}: Verified={verified_mean:.3f}, Failed={failed_mean:.3f}")

class ODEGeneratorEvaluator:
    """Evaluate generated ODEs"""
    
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        
    def load_model(self, path: str):
        """Load trained model"""
        # Implementation depends on model type
        pass
    
    def evaluate_ode(self, ode: str) -> Dict[str, float]:
        """Evaluate a generated ODE"""
        # Parse ODE
        # Extract features
        # Run through model
        # Return scores
        pass
    
    def generate_novel_odes(self, n: int = 10) -> List[str]:
        """Generate novel ODEs"""
        novel_odes = []
        
        # Use trained model to generate
        # Verify they're actually novel
        # Return list
        
        return novel_odes

# Training script
def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ODE generation models')
    parser.add_argument('dataset', help='Path to ODE dataset (JSONL)')
    parser.add_argument('--features', help='Path to features file (Parquet)')
    parser.add_argument('--model', choices=['pattern', 'language', 'both'], default='both')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--analyze', action='store_true', help='Analyze patterns only')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ODEGeneratorTrainer(args.dataset, args.features)
    
    if args.analyze:
        trainer.analyze_patterns()
    else:
        if args.model in ['pattern', 'both']:
            trainer.train_pattern_network(epochs=args.epochs)
        
        if args.model in ['language', 'both']:
            trainer.train_language_model(epochs=args.epochs)
        
        logger.info("Training complete!")

if __name__ == "__main__":
    main()