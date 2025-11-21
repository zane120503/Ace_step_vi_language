from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from datetime import datetime
import argparse
import torch
import json
import matplotlib
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from acestep.text2music_dataset import Text2MusicDataset
from loguru import logger
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    Wav2Vec2FeatureExtractor,
    UMT5EncoderModel,
    AutoTokenizer,
)
import torchaudio
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from acestep.apg_guidance import apg_forward, MomentumBuffer
from tqdm import tqdm
import random
import os
import warnings
import time
import gc
from acestep.pipeline_ace_step import ACEStepPipeline


# Suppress CUDA capability warnings for newer GPUs (e.g., RTX 5060 Ti with sm_120)
# PyTorch may warn about unsupported CUDA capabilities, but GPU can still work
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

# Note: PyTorch/CUDA on Windows with dedicated GPU cannot use "shared GPU memory"
# Shared memory shown in Task Manager is just a display metric, not usable by PyTorch
# We can only use dedicated VRAM (6GB on RTX 3050)
# To work around this, we keep models on CPU and only move to GPU temporarily during forward pass
if torch.cuda.is_available():
    # Set memory fraction to use maximum dedicated VRAM
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of dedicated VRAM to leave some buffer
    logger.info("GPU memory management: Using dedicated VRAM only (shared memory not available for PyTorch)")


class Pipeline(LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        num_workers: int = 4,
        train: bool = True,
        T: int = 1000,
        weight_decay: float = 1e-2,
        every_plot_step: int = 2000,
        shift: float = 3.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        timestep_densities_type: str = "logit_normal",
        ssl_coeff: float = 1.0,
        checkpoint_dir=None,
        max_steps: int = 200000,
        warmup_steps: int = 10,
        dataset_path: str = "./data/your_dataset_path",
        lora_config_path: str = None,
        adapter_name: str = "lora_adapter",
    ):
        super().__init__()

        self.save_hyperparameters()
        self.is_train = train
        self.T = T
        
        # Track training time for ETA calculation
        self.training_start_time = None
        self.step_times = []

        # Initialize scheduler
        self.scheduler = self.get_scheduler()

        # step 1: load model
        logger.info("Initializing ACEStepPipeline...")
        # Clear GPU cache before loading to avoid OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection to free any lingering GPU memory
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared before loading checkpoint")
        
        acestep_pipeline = ACEStepPipeline(checkpoint_dir)
        logger.info("Loading ACE-Step checkpoint...")
        # Temporarily set device to CPU to load checkpoint without OOM
        original_device = acestep_pipeline.device
        acestep_pipeline.device = torch.device("cpu")
        try:
            acestep_pipeline.load_checkpoint(acestep_pipeline.checkpoint_dir)
        finally:
            # Restore original device (but models will stay on CPU as per our design)
            acestep_pipeline.device = original_device
        logger.info("ACE-Step checkpoint loaded successfully")

        # Clear GPU cache before converting models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
            # Log GPU information for debugging with newer GPUs
            gpu_name = torch.cuda.get_device_name(0)
            gpu_capability = torch.cuda.get_device_capability(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name}, Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}, Memory: {gpu_memory:.2f} GB")
            # Warn if CUDA capability is very new (sm_120+) but continue
            if gpu_capability[0] >= 12:
                logger.warning(f"GPU has compute capability sm_{gpu_capability[0]}{gpu_capability[1]} which is very new. "
                             "PyTorch may show warnings, but GPU should still work.")

        logger.info("Preparing transformer for LoRA...")
        # Move to CPU first, then convert to float to avoid OOM
        try:
            model_device = next(acestep_pipeline.ace_step_transformer.parameters()).device
            if model_device.type == 'cuda':
                logger.info("Moving transformer to CPU...")
                acestep_pipeline.ace_step_transformer = acestep_pipeline.ace_step_transformer.cpu()
                torch.cuda.empty_cache()
        except:
            # If can't determine device, just move to CPU to be safe
            logger.info("Moving transformer to CPU (device check failed)...")
            acestep_pipeline.ace_step_transformer = acestep_pipeline.ace_step_transformer.cpu()
            torch.cuda.empty_cache()
        transformers = acestep_pipeline.ace_step_transformer.float()
        transformers.enable_gradient_checkpointing()

        assert lora_config_path is not None, "Please provide a LoRA config path"
        if lora_config_path is not None:
            logger.info(f"Loading LoRA config from {lora_config_path}...")
            try:
                from peft import LoraConfig
            except ImportError:
                raise ImportError("Please install peft library to use LoRA training")
            with open(lora_config_path, encoding="utf-8") as f:
                import json
                lora_config = json.load(f)
            lora_config = LoraConfig(**lora_config)
            transformers.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)
            self.adapter_name = adapter_name
            logger.info("LoRA adapter added successfully")

        self.transformers = transformers

        # Keep DCAE on CPU to avoid OOM, but move to GPU temporarily when needed
        # DCAE encode/decode needs extra VRAM for activations, so keeping on CPU is safer
        try:
            dcae_device = next(acestep_pipeline.music_dcae.parameters()).device
            if dcae_device.type == 'cuda':
                logger.info("Moving DCAE to CPU to avoid OOM...")
                acestep_pipeline.music_dcae = acestep_pipeline.music_dcae.cpu()
                torch.cuda.empty_cache()
        except:
            logger.info("Moving DCAE to CPU (device check failed)...")
            acestep_pipeline.music_dcae = acestep_pipeline.music_dcae.cpu()
            torch.cuda.empty_cache()
        self.dcae = acestep_pipeline.music_dcae.float()
        self.dcae.requires_grad_(False)

        # Use UMT5 (same as original model) for compatibility
        # UMT5 supports Vietnamese but needs proper preprocessing
        logger.info("Loading UMT5 tokenizer and encoder (compatible with trained model)...")
        # Use os module (imported at top level, line 34)
        # Check if checkpoint_dir exists and build path
        if checkpoint_dir:
            text_encoder_checkpoint_path = os.path.join(checkpoint_dir, "umt5-base")
        else:
            text_encoder_checkpoint_path = None
        if text_encoder_checkpoint_path and os.path.exists(text_encoder_checkpoint_path):
            # Use checkpoint from local directory
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_encoder_checkpoint_path
            )
            self.text_encoder_model = UMT5EncoderModel.from_pretrained(
                text_encoder_checkpoint_path, torch_dtype=torch.float32
            ).eval()
        else:
            # Fallback: try to use from acestep_pipeline if available
            logger.warning("UMT5 checkpoint not found, trying to use from pipeline...")
            # We'll load it from the pipeline's checkpoint path
            # For now, use a compatible UMT5 model
            try:
                # Try to get from acestep_pipeline if it's already loaded
                if hasattr(acestep_pipeline, 'text_encoder_model') and hasattr(acestep_pipeline, 'text_tokenizer'):
                    self.text_encoder_model = acestep_pipeline.text_encoder_model.float()
                    self.text_tokenizer = acestep_pipeline.text_tokenizer
                    logger.info("Using UMT5 from acestep_pipeline")
                else:
                    # Last resort: use google/umt5-base (may need to download)
                    logger.warning("Loading UMT5 from HuggingFace (google/umt5-base)...")
                    self.text_tokenizer = AutoTokenizer.from_pretrained("google/umt5-base")
                    self.text_encoder_model = UMT5EncoderModel.from_pretrained(
                        "google/umt5-base", torch_dtype=torch.float32
                    ).eval()
            except Exception as e:
                logger.error(f"Failed to load UMT5: {e}")
                raise
        
        self.text_encoder_model.eval()
        self.text_encoder_model.requires_grad_(False)
        
        # Keep text encoder on GPU to reduce RAM usage (112M ≈ 224MB VRAM)
        # This is small enough to keep on GPU permanently
        if torch.cuda.is_available():
            logger.info("Keeping text encoder on GPU to reduce RAM usage...")
            self.text_encoder_model = self.text_encoder_model.cuda()
            torch.cuda.empty_cache()

        if self.is_train:
            self.transformers.train()

            # download first
            logger.info("Loading MERT model...")
            try:
                self.mert_model = AutoModel.from_pretrained(
                    "m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=checkpoint_dir
                ).eval()
                logger.info("MERT model loaded successfully")
            except:
                import json
                # os is already imported at top level (line 34), no need to import again

                mert_config_path = os.path.join(
                    os.path.expanduser("~"),
                    ".cache",
                    "huggingface",
                    "hub",
                    "models--m-a-p--MERT-v1-330M",
                    "blobs",
                    "14f770758c7fe5c5e8ead4fe0f8e5fa727eb6942"
                )

                with open(mert_config_path) as f:
                    mert_config = json.load(f)
                mert_config["conv_pos_batch_norm"] = False
                with open(mert_config_path, mode="w") as f:
                    json.dump(mert_config, f)
                self.mert_model = AutoModel.from_pretrained(
                    "m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=checkpoint_dir
                ).eval()
            self.mert_model.requires_grad_(False)
            self.resampler_mert = torchaudio.transforms.Resample(
                orig_freq=48000, new_freq=24000, dtype=torch.float32
            )
            self.processor_mert = Wav2Vec2FeatureExtractor.from_pretrained(
                "m-a-p/MERT-v1-330M", trust_remote_code=True
            )
            logger.info("MERT processor loaded successfully")

            logger.info("Loading mHuBERT model...")
            self.hubert_model = AutoModel.from_pretrained("utter-project/mHuBERT-147").eval()
            logger.info("mHuBERT model loaded successfully")
            self.hubert_model.requires_grad_(False)
            self.resampler_mhubert = torchaudio.transforms.Resample(
                orig_freq=48000, new_freq=16000, dtype=torch.float32
            )
            self.processor_mhubert = Wav2Vec2FeatureExtractor.from_pretrained(
                "utter-project/mHuBERT-147",
                cache_dir=checkpoint_dir,
            )
            logger.info("mHuBERT processor loaded successfully")
            logger.info("All SSL models loaded. Initialization complete.")

            self.ssl_coeff = ssl_coeff
    
    def to(self, *args, **kwargs):
        """Override to() to prevent Lightning from moving entire model to GPU.
        We'll handle device placement manually in training_step to avoid OOM."""
        # If Lightning tries to move to CUDA, we'll keep models on CPU
        # and only move them temporarily during forward pass
        device = None
        if args:
            device = args[0] if isinstance(args[0], (torch.device, str)) else None
        elif 'device' in kwargs:
            device = kwargs['device']
        
        if device and (isinstance(device, str) and 'cuda' in device.lower() or 
                      (isinstance(device, torch.device) and device.type == 'cuda')):
            # Don't move entire model to GPU - we'll handle it manually
            logger.info("Skipping automatic GPU move to prevent OOM. Models will stay on CPU.")
            return self
        else:
            # For CPU moves, allow default behavior
            return super().to(*args, **kwargs)

    def infer_mert_ssl(self, target_wavs, wav_lengths):
        # Input is N x 2 x T (48kHz), convert to N x T (24kHz), mono
        # Ensure input is on CPU with float32 dtype for resampler
        target_wavs_mono = target_wavs.mean(dim=1).cpu().float()
        mert_input_wavs_mono_24k = self.resampler_mert(target_wavs_mono)
        bsz = target_wavs.shape[0]
        actual_lengths_24k = wav_lengths // 2  # 48kHz -> 24kHz

        # Normalize the actual audio part
        means = torch.stack(
            [
                mert_input_wavs_mono_24k[i, : actual_lengths_24k[i]].mean()
                for i in range(bsz)
            ]
        )
        vars = torch.stack(
            [
                mert_input_wavs_mono_24k[i, : actual_lengths_24k[i]].var()
                for i in range(bsz)
            ]
        )
        mert_input_wavs_mono_24k = (
            mert_input_wavs_mono_24k - means.view(-1, 1)
        ) / torch.sqrt(vars.view(-1, 1) + 1e-7)

        # MERT SSL constraint
        # Define the length of each chunk (5 seconds of samples)
        chunk_size = 24000 * 5  # 5 seconds, 24000 samples per second
        total_length = mert_input_wavs_mono_24k.shape[1]

        num_chunks_per_audio = (actual_lengths_24k + chunk_size - 1) // chunk_size

        # Process chunks
        all_chunks = []
        chunk_actual_lengths = []
        for i in range(bsz):
            audio = mert_input_wavs_mono_24k[i]
            actual_length = actual_lengths_24k[i]
            for start in range(0, actual_length, chunk_size):
                end = min(start + chunk_size, actual_length)
                chunk = audio[start:end]
                if len(chunk) < chunk_size:
                    chunk = F.pad(
                        chunk, (0, chunk_size - len(chunk))
                    )  # Pad insufficient parts with zeros
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end - start)

        # Stack all chunks to (total_chunks, chunk_size)
        all_chunks = torch.stack(all_chunks, dim=0)

        # Batch inference
        # Move chunks to same device as model
        model_device = next(self.mert_model.parameters()).device
        all_chunks = all_chunks.to(model_device)
        with torch.no_grad():
            # Output shape: (total_chunks, seq_len, hidden_size)
            mert_ssl_hidden_states = self.mert_model(all_chunks).last_hidden_state
        # Move back to CPU if needed
        if model_device.type == 'cuda':
            mert_ssl_hidden_states = mert_ssl_hidden_states.cpu()

        # Calculate the number of features for each chunk
        chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]

        # Trim the hidden states of each chunk
        chunk_hidden_states = [
            mert_ssl_hidden_states[i, : chunk_num_features[i], :]
            for i in range(len(all_chunks))
        ]

        # Organize hidden states by audio
        mert_ssl_hidden_states_list = []
        chunk_idx = 0
        for i in range(bsz):
            audio_chunks = chunk_hidden_states[
                chunk_idx : chunk_idx + num_chunks_per_audio[i]
            ]
            audio_hidden = torch.cat(
                audio_chunks, dim=0
            )  # Concatenate chunks of the same audio
            mert_ssl_hidden_states_list.append(audio_hidden)
            chunk_idx += num_chunks_per_audio[i]

        return mert_ssl_hidden_states_list

    def infer_mhubert_ssl(self, target_wavs, wav_lengths):
        # Step 1: Preprocess audio
        # Input: N x 2 x T (48kHz, stereo) -> N x T (16kHz, mono)
        # Ensure input is on CPU with float32 dtype for resampler
        target_wavs_mono = target_wavs.mean(dim=1).cpu().float()
        mhubert_input_wavs_mono_16k = self.resampler_mhubert(target_wavs_mono)
        bsz = target_wavs.shape[0]
        actual_lengths_16k = wav_lengths // 3  # Convert lengths from 48kHz to 16kHz

        # Step 2: Zero-mean unit-variance normalization (only on actual audio)
        means = torch.stack(
            [
                mhubert_input_wavs_mono_16k[i, : actual_lengths_16k[i]].mean()
                for i in range(bsz)
            ]
        )
        vars = torch.stack(
            [
                mhubert_input_wavs_mono_16k[i, : actual_lengths_16k[i]].var()
                for i in range(bsz)
            ]
        )
        mhubert_input_wavs_mono_16k = (
            mhubert_input_wavs_mono_16k - means.view(-1, 1)
        ) / torch.sqrt(vars.view(-1, 1) + 1e-7)

        # Step 3: Define chunk size for MHubert (30 seconds at 16kHz)
        chunk_size = 16000 * 30  # 30 seconds = 480,000 samples

        # Step 4: Split audio into chunks
        num_chunks_per_audio = (
            actual_lengths_16k + chunk_size - 1
        ) // chunk_size  # Ceiling division
        all_chunks = []
        chunk_actual_lengths = []

        for i in range(bsz):
            audio = mhubert_input_wavs_mono_16k[i]
            actual_length = actual_lengths_16k[i]
            for start in range(0, actual_length, chunk_size):
                end = min(start + chunk_size, actual_length)
                chunk = audio[start:end]
                if len(chunk) < chunk_size:
                    chunk = F.pad(chunk, (0, chunk_size - len(chunk)))  # Pad with zeros
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end - start)

        # Step 5: Stack all chunks for batch inference
        all_chunks = torch.stack(all_chunks, dim=0)  # Shape: (total_chunks, chunk_size)

        # Step 6: Batch inference with MHubert model
        # Move chunks to same device as model
        model_device = next(self.hubert_model.parameters()).device
        all_chunks = all_chunks.to(model_device)
        with torch.no_grad():
            mhubert_ssl_hidden_states = self.hubert_model(all_chunks).last_hidden_state
            # Shape: (total_chunks, seq_len, hidden_size)
        # Move back to CPU if needed
        if model_device.type == 'cuda':
            mhubert_ssl_hidden_states = mhubert_ssl_hidden_states.cpu()

        # Step 7: Compute number of features per chunk (assuming model stride of 320)
        chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]

        # Step 8: Trim hidden states to remove padding effects
        chunk_hidden_states = [
            mhubert_ssl_hidden_states[i, : chunk_num_features[i], :]
            for i in range(len(all_chunks))
        ]

        # Step 9: Reorganize hidden states by original audio
        mhubert_ssl_hidden_states_list = []
        chunk_idx = 0
        for i in range(bsz):
            audio_chunks = chunk_hidden_states[
                chunk_idx : chunk_idx + num_chunks_per_audio[i]
            ]
            audio_hidden = torch.cat(
                audio_chunks, dim=0
            )  # Concatenate chunks for this audio
            mhubert_ssl_hidden_states_list.append(audio_hidden)
            chunk_idx += num_chunks_per_audio[i]
        return mhubert_ssl_hidden_states_list

    def get_text_embeddings(self, texts, device, text_max_length=256):
        # Preprocess Vietnamese text: comprehensive normalization for UMT5
        # UMT5 (Universal Multilingual T5) supports Vietnamese natively
        # Key points:
        # 1. Model was trained with UMT5 embeddings → must use UMT5 (not ViT5)
        # 2. UMT5 tokenizer can handle Vietnamese diacritics (ă, â, ê, ô, ơ, ư, đ)
        # 3. When you train with Vietnamese data, model learns Vietnamese pronunciation
        # 4. Proper normalization ensures UMT5 tokenizes Vietnamese correctly
        import unicodedata
        import re
        processed_texts = []
        for text in texts:
            # Step 1: Normalize Unicode to NFC (canonical composition)
            # Critical for Vietnamese: ensures diacritics are properly combined
            # Example: "ă" should be single character, not "a" + combining mark
            text = unicodedata.normalize('NFC', text)
            
            # Step 2: Preserve Vietnamese diacritics - UMT5 needs them for correct tokenization
            # Vietnamese: ă, â, ê, ô, ơ, ư, đ, and their uppercase variants
            # UMT5 tokenizer will split Vietnamese words correctly if diacritics are preserved
            
            # Step 3: Normalize whitespace (keep single spaces between words)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Step 4: Remove zero-width characters that might confuse tokenizer
            text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)  # Zero-width spaces
            
            # Step 5: Keep original case - UMT5 is case-sensitive
            # Model was trained with mixed case, so preserve case for semantic meaning
            # Vietnamese proper nouns (names, places) should keep their case
            
            processed_texts.append(text)
        
        inputs = self.text_tokenizer(
            processed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_length,
        )
        # Text encoder is kept on GPU, so move inputs to GPU
        encoder_device = next(self.text_encoder_model.parameters()).device
        inputs = {key: value.to(encoder_device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        # Move results to target device (usually GPU for training)
        last_hidden_states = last_hidden_states.to(device)
        attention_mask = attention_mask.to(device)
        return last_hidden_states, attention_mask

    def preprocess(self, batch, train=True):
        target_wavs = batch["target_wavs"].to(torch.float32)
        wav_lengths = batch["wav_lengths"]

        dtype = target_wavs.dtype
        bs = target_wavs.shape[0]
        device = target_wavs.device

        # SSL constraints
        mert_ssl_hidden_states = None
        mhubert_ssl_hidden_states = None
        if train:
            # Force SSL inference on CPU to avoid GPU OOM
            target_wavs_cpu = target_wavs.cpu()
            wav_lengths_cpu = wav_lengths.cpu()

            with torch.no_grad():
                mert_ssl_hidden_states = self.infer_mert_ssl(target_wavs_cpu, wav_lengths_cpu)
                mhubert_ssl_hidden_states = self.infer_mhubert_ssl(
                    target_wavs_cpu, wav_lengths_cpu
                )

        # 1: text embedding
        texts = batch["prompts"]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
            texts, device
        )
        encoder_text_hidden_states = encoder_text_hidden_states.to(dtype)

        # DCAE encode: Keep on CPU to avoid OOM
        # DCAE encode needs ~1.2GB VRAM for activations, which is too much when combined with text encoder
        # So we keep DCAE on CPU (slightly slower but avoids OOM)
        dcae_device = next(self.dcae.parameters()).device
        # Ensure DCAE is on CPU
        if dcae_device.type == 'cuda':
            self.dcae = self.dcae.cpu()
            torch.cuda.empty_cache()
        # Move input to CPU if needed
        if target_wavs.device.type == 'cuda':
            target_wavs = target_wavs.cpu()
        target_latents, _ = self.dcae.encode(target_wavs, wav_lengths)
        # Move latents back to target device
        target_latents = target_latents.to(device)
        attention_mask = torch.ones(
            bs, target_latents.shape[-1], device=device, dtype=dtype
        )

        speaker_embds = batch["speaker_embs"].to(dtype)
        keys = batch["keys"]
        lyric_token_ids = batch["lyric_token_ids"]
        lyric_mask = batch["lyric_masks"]

        # cfg
        if train:
            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bs,), device=device) < 0.15),
                torch.zeros(size=(bs,), device=device),
                torch.ones(size=(bs,), device=device),
            ).long()
            # N x T x 768
            encoder_text_hidden_states = torch.where(
                full_cfg_condition_mask.unsqueeze(1).unsqueeze(1).bool(),
                encoder_text_hidden_states,
                torch.zeros_like(encoder_text_hidden_states),
            )

            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bs,), device=device) < 0.50),
                torch.zeros(size=(bs,), device=device),
                torch.ones(size=(bs,), device=device),
            ).long()
            # N x 512
            speaker_embds = torch.where(
                full_cfg_condition_mask.unsqueeze(1).bool(),
                speaker_embds,
                torch.zeros_like(speaker_embds),
            )

            # Lyrics
            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bs,), device=device) < 0.15),
                torch.zeros(size=(bs,), device=device),
                torch.ones(size=(bs,), device=device),
            ).long()
            lyric_token_ids = torch.where(
                full_cfg_condition_mask.unsqueeze(1).bool(),
                lyric_token_ids,
                torch.zeros_like(lyric_token_ids),
            )
            lyric_mask = torch.where(
                full_cfg_condition_mask.unsqueeze(1).bool(),
                lyric_mask,
                torch.zeros_like(lyric_mask),
            )

        return (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
        )

    def get_scheduler(self):
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.T,
            shift=self.hparams.shift,
        )

    def configure_optimizers(self):
        trainable_params = [
            p for name, p in self.transformers.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            params=[
                {"params": trainable_params},
            ],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.8, 0.9),
        )
        max_steps = self.hparams.max_steps
        warmup_steps = self.hparams.warmup_steps  # New hyperparameter for warmup steps

        # Create a scheduler that first warms up linearly, then decays linearly
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from 0 to learning_rate
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Linear decay from learning_rate to 0
                progress = float(current_step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                return max(0.0, 1.0 - progress)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=-1
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def train_dataloader(self):
        self.train_dataset = Text2MusicDataset(
            train=True,
            train_dataset_path=self.hparams.dataset_path,
        )
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.num_workers > 0,  # Only pin memory if using workers
            collate_fn=self.train_dataset.collate_fn,
        )

    def get_sd3_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_timestep(self, bsz, device):
        if self.hparams.timestep_densities_type == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            # In practice, we sample the random variable u from a normal distribution u ∼ N (u; m, s)
            # and map it through the standard logistic function
            u = torch.normal(
                mean=self.hparams.logit_mean,
                std=self.hparams.logit_std,
                size=(bsz,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(
                indices, 0, self.scheduler.config.num_train_timesteps - 1
            )
            timesteps = self.scheduler.timesteps[indices].to(device)

        return timesteps

    def run_step(self, batch, batch_idx):
        # Initialize training start time on first step
        if self.training_start_time is None:
            self.training_start_time = time.time()
            logger.info(f"🚀 Training started. Target: {self.hparams.max_steps} steps")

        step_start_time = time.time()
        self.plot_step(batch, batch_idx)
        (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
        ) = self.preprocess(batch)

        target_image = target_latents
        device = target_image.device
        dtype = target_image.dtype
        # Step 1: Generate random noise, initialize settings
        noise = torch.randn_like(target_image, device=device)
        bsz = target_image.shape[0]
        timesteps = self.get_timestep(bsz, device)

        # Add noise according to flow matching.
        sigmas = self.get_sd3_sigmas(
            timesteps=timesteps, device=device, n_dim=target_image.ndim, dtype=dtype
        )
        noisy_image = sigmas * noise + (1.0 - sigmas) * target_image

        # This is the flow-matching target for vanilla SD3.
        target = target_image

        # SSL constraints for CLAP and vocal_latent_channel2
        all_ssl_hiden_states = []
        if mert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mert_ssl_hidden_states)
        if mhubert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mhubert_ssl_hidden_states)

        # N x H -> N x c x W x H
        x = noisy_image
        
        # Move transformer to GPU temporarily for forward pass
        # Only move if not already on GPU (to save memory)
        transformer_device = next(self.transformers.parameters()).device
        if transformer_device.type != 'cuda':
            logger.debug("Moving transformer to GPU for forward pass...")
            self.transformers = self.transformers.to(device)
        
        # Step 5: Predict noise
        transformer_output = self.transformers(
            hidden_states=x,
            attention_mask=attention_mask,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=timesteps.to(device).to(dtype),
            ssl_hidden_states=all_ssl_hiden_states,
        )
        
        # Move transformer back to CPU to save memory (optional, can keep on GPU if enough memory)
        # Uncomment if OOM occurs:
        # self.transformers = self.transformers.cpu()
        # torch.cuda.empty_cache()
        model_pred = transformer_output.sample
        proj_losses = transformer_output.proj_losses

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_image

        # Compute loss. Only calculate loss where chunk_mask is 1 and there is no padding
        # N x T x 64
        # N x T -> N x c x W x T
        mask = (
            attention_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, target_image.shape[1], target_image.shape[2], -1)
        )
        mask = mask.to(device=model_pred.device, dtype=model_pred.dtype)

        selected_model_pred = (model_pred * mask).reshape(bsz, -1).contiguous()
        selected_target = (target * mask).reshape(bsz, -1).contiguous()

        loss = F.mse_loss(selected_model_pred, selected_target, reduction="none")
        loss = loss.mean(1)
        loss = loss * mask.reshape(bsz, -1).mean(1)
        loss = loss.mean()

        prefix = "train"

        self.log(
            f"{prefix}/denoising_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        total_proj_loss = 0.0
        for k, v in proj_losses:
            self.log(
                f"{prefix}/{k}_loss", v, on_step=True, on_epoch=False, prog_bar=True
            )
            total_proj_loss += v

        if len(proj_losses) > 0:
            total_proj_loss = total_proj_loss / len(proj_losses)

        loss = loss + total_proj_loss * self.ssl_coeff
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # Log learning rate if scheduler exists
        if self.lr_schedulers() is not None:
            learning_rate = self.lr_schedulers().get_last_lr()[0]
            self.log(
                f"{prefix}/learning_rate",
                learning_rate,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        # with torch.autograd.detect_anomaly():
        #     self.manual_backward(loss)
        
        # Calculate and log training progress
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        self.step_times.append(step_duration)
        
        # Keep only last 50 step times for average calculation
        if len(self.step_times) > 50:
            self.step_times = self.step_times[-50:]
        
        current_step = self.global_step
        max_steps = self.hparams.max_steps
        
        # Log progress every 10 steps (or first step) to avoid spam
        if current_step % 10 == 0 or current_step == 1:
            # Calculate statistics
            avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else step_duration
            elapsed_time = step_end_time - self.training_start_time
            remaining_steps = max(0, max_steps - current_step)
            estimated_remaining_time = avg_step_time * remaining_steps
            
            # Format time helper
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}s"
                elif seconds < 3600:
                    return f"{seconds/60:.1f}m"
                else:
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    return f"{hours}h {minutes}m"
            
            progress_pct = (current_step / max_steps * 100) if max_steps > 0 else 0
            
            logger.info(
                f"📊 Step {current_step}/{max_steps} ({progress_pct:.1f}%) | "
                f"Avg: {format_time(avg_step_time)}/step | "
                f"Elapsed: {format_time(elapsed_time)} | "
                f"ETA: {format_time(estimated_remaining_time)}"
            )
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, batch_idx)

    def on_save_checkpoint(self, checkpoint):
        state = {}
        log_dir = self.logger.log_dir
        epoch = self.current_epoch
        step = self.global_step
        checkpoint_name = f"epoch={epoch}-step={step}_lora"
        checkpoint_dir = os.path.join(log_dir, "checkpoints", checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.transformers.save_lora_adapter(checkpoint_dir, adapter_name=self.adapter_name)
        return state

    @torch.no_grad()
    def diffusion_process(
        self,
        duration,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        omega_scale=10.0,
    ):

        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        device = encoder_text_hidden_states.device
        dtype = encoder_text_hidden_states.dtype
        bsz = encoder_text_hidden_states.shape[0]

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )

        frame_length = int(duration * 44100 / 512 / 8)
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps=infer_steps, device=device, timesteps=None
        )

        target_latents = randn_tensor(
            shape=(bsz, 8, 16, frame_length),
            generator=random_generators,
            device=device,
            dtype=dtype,
        )
        attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)
        if do_classifier_free_guidance:
            attention_mask = torch.cat([attention_mask] * 2, dim=0)
            encoder_text_hidden_states = torch.cat(
                [
                    encoder_text_hidden_states,
                    torch.zeros_like(encoder_text_hidden_states),
                ],
                0,
            )
            text_attention_mask = torch.cat([text_attention_mask] * 2, dim=0)

            speaker_embds = torch.cat(
                [speaker_embds, torch.zeros_like(speaker_embds)], 0
            )

            lyric_token_ids = torch.cat(
                [lyric_token_ids, torch.zeros_like(lyric_token_ids)], 0
            )
            lyric_mask = torch.cat([lyric_mask, torch.zeros_like(lyric_mask)], 0)

        momentum_buffer = MomentumBuffer()

        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latents = target_latents
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformers(
                hidden_states=latent_model_input,
                attention_mask=attention_mask,
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embeds=speaker_embds,
                lyric_token_idx=lyric_token_ids,
                lyric_mask=lyric_mask,
                timestep=timestep,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_with_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = apg_forward(
                    pred_cond=noise_pred_with_cond,
                    pred_uncond=noise_pred_uncond,
                    guidance_scale=guidance_scale,
                    momentum_buffer=momentum_buffer,
                )

            target_latents = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=target_latents,
                return_dict=False,
                omega=omega_scale,
            )[0]

        return target_latents

    def predict_step(self, batch):
        (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
        ) = self.preprocess(batch, train=False)

        infer_steps = 60
        guidance_scale = 15.0
        omega_scale = 10.0
        seed_num = 1234
        random.seed(seed_num)
        bsz = target_latents.shape[0]
        random_generators = [torch.Generator(device=self.device) for _ in range(bsz)]
        seeds = []
        for i in range(bsz):
            seed = random.randint(0, 2**32 - 1)
            random_generators[i].manual_seed(seed)
            seeds.append(seed)
        duration = 240  # Fixed duration (24 * 10)
        pred_latents = self.diffusion_process(
            duration=duration,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embds,
            lyric_token_ids=lyric_token_ids,
            lyric_mask=lyric_mask,
            random_generators=random_generators,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
        )

        audio_lengths = batch["wav_lengths"]
        # DCAE decode: Keep on CPU to avoid OOM
        # DCAE decode needs significant VRAM, so we keep it on CPU
        dcae_device = next(self.dcae.parameters()).device
        # Ensure DCAE is on CPU
        if dcae_device.type == 'cuda':
            self.dcae = self.dcae.cpu()
            torch.cuda.empty_cache()
        # Move latents to CPU if needed
        if pred_latents.device.type == 'cuda':
            pred_latents = pred_latents.cpu()
        sr, pred_wavs = self.dcae.decode(
            pred_latents, audio_lengths=audio_lengths, sr=48000
        )
        return {
            "target_wavs": batch["target_wavs"],
            "pred_wavs": pred_wavs,
            "keys": keys,
            "prompts": batch["prompts"],
            "candidate_lyric_chunks": batch["candidate_lyric_chunks"],
            "sr": sr,
            "seeds": seeds,
        }

    def construct_lyrics(self, candidate_lyric_chunk):
        lyrics = []
        for chunk in candidate_lyric_chunk:
            lyrics.append(chunk["lyric"])

        lyrics = "\n".join(lyrics)
        return lyrics

    def plot_step(self, batch, batch_idx):
        global_step = self.global_step
        local_rank = getattr(self, "local_rank", 0)

        dist_rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            dist_rank = torch.distributed.get_rank()

        current_device = 0
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()

        if (
            global_step % self.hparams.every_plot_step != 0
            or local_rank != 0
            or dist_rank != 0
            or current_device != 0
        ):
            return
        results = self.predict_step(batch)

        target_wavs = results["target_wavs"]
        pred_wavs = results["pred_wavs"]
        keys = results["keys"]
        prompts = results["prompts"]
        candidate_lyric_chunks = results["candidate_lyric_chunks"]
        sr = results["sr"]
        seeds = results["seeds"]
        i = 0
        for key, target_wav, pred_wav, prompt, candidate_lyric_chunk, seed in zip(
            keys, target_wavs, pred_wavs, prompts, candidate_lyric_chunks, seeds
        ):
            key = key
            prompt = prompt
            lyric = self.construct_lyrics(candidate_lyric_chunk)
            key_prompt_lyric = f"# KEY\n\n{key}\n\n\n# PROMPT\n\n{prompt}\n\n\n# LYRIC\n\n{lyric}\n\n# SEED\n\n{seed}\n\n"
            log_dir = self.logger.log_dir
            save_dir = f"{log_dir}/eval_results/step_{self.global_step}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torchaudio.save(
                f"{save_dir}/target_wav_{key}_{i}.wav", target_wav.float().cpu(), sr
            )
            torchaudio.save(
                f"{save_dir}/pred_wav_{key}_{i}.wav", pred_wav.float().cpu(), sr
            )
            with open(
                f"{save_dir}/key_prompt_lyric_{key}_{i}.txt", "w", encoding="utf-8"
            ) as f:
                f.write(key_prompt_lyric)
            i += 1


def main(args):
    model = Pipeline(
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        shift=args.shift,
        max_steps=args.max_steps,
        every_plot_step=args.every_plot_step,
        dataset_path=args.dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        adapter_name=args.exp_name,
        lora_config_path=args.lora_config_path
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=None,
        every_n_train_steps=args.every_n_train_steps,
        save_top_k=-1,
    )
    # add datetime str to version
    logger_callback = TensorBoardLogger(
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + args.exp_name,
        save_dir=args.logger_dir,
    )
    # Use appropriate strategy for Windows (DDP not supported on Windows)
    # For single GPU, use "auto" or None. For multi-GPU on Windows, use "ddp_spawn"
    if args.devices == 1:
        strategy = "auto"  # Lightning will choose appropriate strategy
    else:
        strategy = "ddp_spawn"  # Multi-GPU on Windows
    
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy=strategy,
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        log_every_n_steps=1,
        logger=logger_callback,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs,
        val_check_interval=args.val_check_interval,
    )

    trainer.fit(
        model,
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--num_nodes", type=int, default=1)
    args.add_argument("--shift", type=float, default=3.0)
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--num_workers", type=int, default=8)
    args.add_argument("--epochs", type=int, default=-1)
    args.add_argument("--max_steps", type=int, default=2000000)
    args.add_argument("--every_n_train_steps", type=int, default=2000)
    args.add_argument("--dataset_path", type=str, default="./zh_lora_dataset")
    args.add_argument("--exp_name", type=str, default="chinese_rap_lora")
    args.add_argument("--precision", type=str, default="32")
    args.add_argument("--accumulate_grad_batches", type=int, default=1)
    args.add_argument("--devices", type=int, default=1)
    args.add_argument("--logger_dir", type=str, default="./exps/logs/")
    args.add_argument("--ckpt_path", type=str, default=None)
    args.add_argument("--checkpoint_dir", type=str, default=None)
    args.add_argument("--gradient_clip_val", type=float, default=0.5)
    args.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    args.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=1)
    args.add_argument("--every_plot_step", type=int, default=2000)
    args.add_argument("--val_check_interval", type=int, default=None)
    args.add_argument("--lora_config_path", type=str, default="config/zh_rap_lora_config.json")
    args = args.parse_args()
    main(args)
