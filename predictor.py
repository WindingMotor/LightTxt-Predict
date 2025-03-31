import os
import time
import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer

class Predictor:
    """Ultra-lightweight LLM-based predictive text system with performance optimizations"""
    
    # NOTE: Use newer Qwen/Qwen2.5-0.5B for higher confidence but higher latency
    # NOTE: For even lower latency, use distilgpt2 for worse results but lower latency

    # NOTE: Test microsoft/Phi-4-mini-instruct (might be better on faster machines) vs qwen

    # Default to Qwen/Qwen2-0.5B for lower latency
    def __init__(self, model_name="Qwen/Qwen2-0.5B", cache_dir="model_cache", quantize=True, 
                precompute_common=True, use_half_precision=True):
        """Initialize the lightweight LLM model and tokenizer with optimizations"""
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configure max context length for Qwen2 models
        self.max_context_length = 32768 if "qwen2" in model_name.lower() else 1024
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load the tokenizer with padding configuration to fix the warning
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            padding_side='left',  # Consistent padding for faster processing
            use_fast=True,  # Use the fast tokenizer implementation
            trust_remote_code=True  # Required for Qwen models
        )
        
        # Ensure padding token is set correctly
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the model with optimizations
        print(f"Loading model {model_name} on {self.device}...")
        start_time = time.time()

        flash_attn_available = False
        try:
            import flash_attn
            flash_attn_available = True
            print("Flash attention available!")
        except ImportError:
            pass
        
        # Select appropriate attention implementation
        attn_implementation = None
        if self.device == "cuda" and flash_attn_available:
            attn_implementation = "flash_attention_2"
        
        # Add optimization flags
        model_kwargs = {
            "cache_dir": cache_dir,
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16 if use_half_precision and self.device == "cuda" else None,
            "trust_remote_code": True,
            "device_map": "auto"  # This will handle device placement automatically
        }
        
        # Only add attn_implementation if it's set
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        
        # Apply quantization to reduce memory usage if requested
        if quantize and self.device == "cpu":
            print("Applying 8-bit quantization to reduce memory usage...")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        # Move model to device
        self.model.to(self.device)
        
        # Enable evaluation mode
        self.model.eval()
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        print(f"Model size: {self._get_model_size_mb():.1f} MB")
        
        # Improved multi-level cache for predictions
        # Level 1: Exact match cache
        self.prediction_cache = {}
        # Level 2: Prefix cache - store known good contexts
        self.context_cache = {}
        
        # Max cache sizes to prevent memory issues
        self.max_cache_size = 200
        self.max_context_cache_size = 50
        
        # Precompute common starting predictions
        if precompute_common:
            self._precompute_common_predictions()
            
        # Start a background thread to periodically clear old cache entries
        self._start_cache_maintenance()
        
        # Pre-allocate buffers for inference to reduce memory allocations
        self.inference_buffer = None
        if self.device == "cuda":
            # Pre-warm the GPU to avoid cold start latency
            self._warm_up_model()
    
    def _get_model_size_mb(self):
        """Get approximate model size in MB"""
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return model_size / (1024 * 1024)
    
    def _precompute_common_predictions(self):
        """Precompute predictions for common starting phrases"""
        common_phrases = [
            "", "I", "The", "A", "What", "How", "When", "My name is", 
            "I am", "Can you", "Please", "Thank you", "Hello", "Hi"
        ]
        
        print("Precomputing predictions for common phrases...")
        for phrase in common_phrases:
            # Compute in the background to not block initialization
            threading.Thread(
                target=lambda p: self.predict_next(p, preload_only=True),
                args=(phrase,)
            ).start()
    
    def _warm_up_model(self):
        """Warm up the model to avoid cold-start latency"""
        print("Warming up model...")
        dummy_input = self.tokenizer("Hello world", return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**dummy_input)
    
    def _start_cache_maintenance(self):
        """Start a background thread to periodically clean up the cache"""
        def cache_maintenance():
            while True:
                time.sleep(60)  # Run every minute
                try:
                    self._clean_cache()
                except Exception as e:
                    print(f"Error in cache maintenance: {e}")
        
        # Start the maintenance thread
        thread = threading.Thread(target=cache_maintenance, daemon=True)
        thread.start()
    
    def _clean_cache(self):
        """Clean up old cache entries to prevent memory issues"""
        if len(self.prediction_cache) > self.max_cache_size:
            # Remove oldest entries (assuming keys added chronologically)
            keys = list(self.prediction_cache.keys())
            for key in keys[:len(keys) // 2]:  # Remove half of the entries
                del self.prediction_cache[key]
        
        if len(self.context_cache) > self.max_context_cache_size:
            keys = list(self.context_cache.keys())
            for key in keys[:len(keys) // 2]:
                del self.context_cache[key]
    
    def _get_cached_context(self, text):
        """Find the longest cached context that matches the beginning of the given text"""
        matching_contexts = []
        for context in self.context_cache:
            if text.lower().startswith(context.lower()):
                matching_contexts.append(context)
        
        if not matching_contexts:
            return None
        
        # Return the longest matching context
        return max(matching_contexts, key=len)
    
    def predict_next(self, text, top_k=3, max_length=None, preload_only=False):
        """Predict the most likely next words given the input text
        
        Args:
            text: Input text to predict from
            top_k: Number of suggestions to return
            max_length: Maximum number of words to consider from input
            preload_only: If True, only compute and cache the result without returning it
        """
        # Handle empty input with fixed predictions to avoid unnecessary computation
        if not text.strip():
            default_predictions = [("I", 0.25), ("The", 0.2), ("Hello", 0.15)]
            return (default_predictions, 0) if not preload_only else None
        
        # Check exact cache match first
        cache_key = f"{text.lower()}|{top_k}"
        if cache_key in self.prediction_cache:
            cached_result = self.prediction_cache[cache_key]
            return cached_result if not preload_only else None
        
        # Set max length to ensure we don't process too much text
        if max_length is None:
            # Only process the last few words for efficiency
            max_length = min(10, len(text.split()))
        
        start_time = time.time()
        
        try:
            # Get the context (last few words)
            tokens = text.strip().split()
            context = " ".join(tokens[-max_length:])
            
            # For Qwen2 models, we can leverage more context if available
            if len(tokens) > max_length and hasattr(self, 'max_context_length') and self.max_context_length > 1024:
                # Take advantage of longer context while keeping efficiency
                extended_length = min(len(tokens), 50)  # Use up to 50 tokens for extended context
                if extended_length > max_length:
                    extended_context = " ".join(tokens[-extended_length:])
                    # Check if the extended context is reasonable in size
                    tokenized = self.tokenizer(extended_context, return_tensors="pt")
                    if tokenized["input_ids"].shape[1] <= 512:  # Still keep it reasonable
                        context = extended_context
            
            # Use the tokenizer with explicit attention mask to avoid warnings
            inputs = self.tokenizer(context, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate predictions more efficiently
            with torch.no_grad():
                # Use inference mode for additional speedup
                with torch.inference_mode():
                    # Get model's probability distribution over next tokens
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0, -1, :]
                    
                    # Convert to probabilities efficiently
                    probs = torch.nn.functional.softmax(logits, dim=0)
                    
                    # Get top predicted token IDs efficiently
                    # Request more tokens than needed to filter out special tokens
                    multiplier = 2 if top_k <= 5 else 1.5  # Adaptive multiplier
                    candidates_k = min(int(top_k * multiplier), len(probs) - 1)
                    top_tokens = torch.topk(probs, candidates_k)
                    
                    # Process predictions efficiently
                    predictions = []
                    for i, token_id in enumerate(top_tokens.indices):
                        # Skip special tokens and single character tokens (except common ones)
                        token = self.tokenizer.decode(token_id).strip()
                        if (token and 
                            not token.startswith("<") and 
                            not token.endswith(">") and 
                            (len(token) > 1 or token in "aIiAoO")):
                            predictions.append((token, float(top_tokens.values[i])))
                        
                        # Stop when we have enough predictions
                        if len(predictions) >= top_k:
                            break
                    
                    # Only generate more tokens if we don't have enough predictions
                    # and if the model seems to give good quality responses
                    if len(predictions) < top_k and len(context) > 1:
                        # Generate efficiently with batched approach
                        # FIX: Use only max_new_tokens parameter to avoid the warning about max_length
                        gen_outputs = self.model.generate(
                            inputs["input_ids"],
                            max_new_tokens=1,  # Just one more token
                            do_sample=True,
                            top_k=30,
                            top_p=0.95,
                            num_return_sequences=min(top_k - len(predictions), 2),
                            pad_token_id=self.tokenizer.pad_token_id,
                            attention_mask=inputs["attention_mask"]
                        )
                        
                        for output in gen_outputs:
                            # Extract only the newly generated token
                            new_token = self.tokenizer.decode(
                                output[inputs["input_ids"].shape[1]:].squeeze()
                            ).strip()
                            
                            if new_token and not new_token.startswith("<") and not new_token.endswith(">"):
                                # Add with a lower probability to differentiate from top predictions
                                if (new_token, 0.5) not in predictions:
                                    predictions.append((new_token, 0.5))
            
            # Ensure we have the requested number of predictions
            predictions = predictions[:top_k]
            
            # Normalize probabilities for consistency
            if predictions:
                total = sum(prob for _, prob in predictions)
                if total > 0:
                    predictions = [(word, prob/total) for word, prob in predictions]
            
            # Add default predictions if we don't have enough
            while len(predictions) < top_k:
                default_words = ["the", "and", "to", "a", "of", "is", "in", "for", "that"]
                for word in default_words:
                    if not any(word == p[0].lower() for p in predictions):
                        predictions.append((word, 0.1))
                        break
                
                # Break if we've gone through all default words
                if len(predictions) < top_k:
                    break
            
            prediction_time = time.time() - start_time
            
            # Cache the result
            result = (predictions, prediction_time)
            self.prediction_cache[cache_key] = result
            
            # Also cache a simplified form of the context
            simple_context = " ".join(context.strip().lower().split()[:5])
            if simple_context and len(simple_context) > 2:
                self.context_cache[simple_context] = True
            
            return result if not preload_only else None
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            # Provide fallback predictions
            return ([("the", 0.3), ("a", 0.2), ("is", 0.1)], 0) if not preload_only else None
    
    def batch_predict(self, texts, top_k=3):
        """Predict for multiple inputs at once for improved efficiency"""
        results = []
        
        # Check cache first for all inputs
        cached_results = []
        texts_to_process = []
        
        for text in texts:
            cache_key = f"{text.lower()}|{top_k}"
            if cache_key in self.prediction_cache:
                cached_results.append((True, self.prediction_cache[cache_key]))
                texts_to_process.append(None)
            else:
                cached_results.append((False, None))
                texts_to_process.append(text)
        
        # Process all non-cached texts in a batch
        if any(text is not None for text in texts_to_process):
            valid_texts = [t for t in texts_to_process if t is not None]
            
            # Tokenize all inputs at once
            batch_inputs = self.tokenizer(valid_texts, return_tensors="pt", padding=True)
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            
            with torch.no_grad():
                with torch.inference_mode():
                    # Get model outputs for the batch
                    batch_outputs = self.model(**batch_inputs)
                    
                    # Process each output individually
                    batch_results = []
                    for i, logits in enumerate(batch_outputs.logits):
                        # Get probabilities for the last token
                        next_token_logits = logits[-1, :]
                        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0)
                        
                        # Get top tokens
                        top_tokens = torch.topk(next_token_probs, min(top_k * 2, len(next_token_probs)))
                        
                        # Convert to words
                        predictions = []
                        for j, token_id in enumerate(top_tokens.indices):
                            token = self.tokenizer.decode(token_id).strip()
                            if token and not token.startswith("<") and not token.endswith(">") and len(token) > 1:
                                predictions.append((token, float(top_tokens.values[j])))
                            
                            if len(predictions) >= top_k:
                                break
                        
                        # Normalize probabilities
                        if predictions:
                            total = sum(prob for _, prob in predictions)
                            predictions = [(word, prob/total) for word, prob in predictions]
                        
                        batch_results.append(predictions)
            
            # Map batch results back to the original texts
            idx = 0
            for i, (is_cached, _) in enumerate(cached_results):
                if not is_cached:
                    result = batch_results[idx]
                    idx += 1
                    
                    # Cache the new result
                    cache_key = f"{texts[i].lower()}|{top_k}"
                    self.prediction_cache[cache_key] = (result, 0)  # No timing info for batch
                    
                    cached_results[i] = (True, (result, 0))
        
        # Return all results
        return [result for _, result in cached_results]