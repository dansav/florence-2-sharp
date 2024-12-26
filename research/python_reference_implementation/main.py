# Basic imports
import os
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from transformers import BartTokenizer
import matplotlib.pyplot as plt


class Florence2Reference:
    def __init__(self):
        """
        Initialize Florence2 reference implementation
        model_dir: Directory containing the ONNX model files
        """
        # Initialize ONNX Runtime sessions
        self.vision_session = ort.InferenceSession(f"./data/models/vision_encoder.onnx")
        self.embed_session = ort.InferenceSession(f"./data/models/embed_tokens.onnx")
        self.encoder_session = ort.InferenceSession(f"./data/models/encoder_model.onnx")
        self.decoder_session = ort.InferenceSession(f"./data/models/decoder_model.onnx")

        # Initialize tokenizer and special tokens (from config.json)
        self.tokenizer = BartTokenizer.from_pretrained("./data/tokenizer")
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.decoder_start_token_id = 2  # From config

        # Image preprocessing constants - ensure numpy float32
        self.image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.image_size = (768, 768)  # height, width

        # Add task prompt mappings
        self.task_prompts_without_inputs = {
            "<CAPTION>": "What does the image describe?",
            "<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
            "<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
            "<OD>": "Locate the objects with category name in the image.",
            "<OCR>": "What is the text in the image?",
            "<OCR_WITH_REGION>": "What is the text in the image, with regions?",
            "<DENSE_REGION_CAPTION>": "Locate the objects in the image, with their descriptions.",
            "<REGION_PROPOSAL>": "Locate the region proposals in the image.",
        }

    def verify_image_preprocessing(self, image_input):
        """
        Verify the preprocessed image values
        """
        print("\nImage preprocessing verification:")
        print(f"Min pixel value: {np.min(image_input)}")
        print(f"Max pixel value: {np.max(image_input)}")
        print(f"Mean pixel value: {np.mean(image_input)}")
        print(f"Std pixel value: {np.std(image_input)}")

    def verify_vision_features(self, vision_features):
        """
        Verify the vision encoder output features
        """
        print("\nVision features verification:")
        print(f"Min feature value: {np.min(vision_features)}")
        print(f"Max feature value: {np.max(vision_features)}")
        print(f"Mean feature value: {np.mean(vision_features)}")
        print(f"Std feature value: {np.std(vision_features)}")

        # Check for dead features (all zeros)
        zero_features = np.all(vision_features == 0, axis=(0, 1))
        print(f"Number of dead features: {np.sum(zero_features)}")

    def preprocess_image(self, image):
        """
        Preprocess image for the vision encoder
        """
        # Resize image
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        # Save original size for debugging
        print(f"Original image size: {image.size}")

        # Display original image
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(image)
        # plt.title("Original Image")

        image = image.resize(self.image_size, Image.LANCZOS)

        # Convert to numpy array and normalize
        # Explicitly specify float32 dtype
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Store copy for visualization
        # pre_norm_image = image_array.copy()

        # Handle broadcasting for mean and std
        mean = self.image_mean.reshape(-1, 1, 1)
        std = self.image_std.reshape(-1, 1, 1)

        # Normalize image
        image_array = np.transpose(image_array, (2, 0, 1))  # HWC to CHW
        image_array = (image_array - mean) / std

        # Visualize normalized image
        # Convert back to HWC and denormalize for visualization
        # viz_image = image_array * std + mean
        # viz_image = np.transpose(viz_image, (1, 2, 0))  # CHW to HWC
        # viz_image = np.clip(viz_image, 0, 1)  # Clip values to valid range

        # plt.subplot(1, 2, 2)
        # plt.imshow(viz_image)
        # plt.title("Preprocessed Image")

        # plt.tight_layout()
        # plt.show()

        # Print some statistics about the image values
        # print("\nImage statistics:")
        # print(f"Pre-normalization - Min: {pre_norm_image.min():.3f}, Max: {pre_norm_image.max():.3f}")
        # print(f"Post-normalization - Min: {image_array.min():.3f}, Max: {image_array.max():.3f}")

        # Add batch dimension and ensure float32
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

        # Verify preprocessing
        # self.verify_image_preprocessing(image_array)

        return image_array

    def get_attention_mask(self, input_ids):
        """
        Create attention mask for input tokens
        """
        attention_mask = (input_ids != self.pad_token_id).astype(np.int64)
        return attention_mask

    def create_decoder_inputs(self, decoder_input_ids):
        """
        Create inputs for the decoder model
        """
        # Create attention mask for decoder inputs
        decoder_attention_mask = self.get_attention_mask(decoder_input_ids)

        # Get embeddings for decoder input tokens
        decoder_inputs = {
            "input_ids": decoder_input_ids.astype(np.int64)  # Ensure correct dtype
        }
        decoder_embeds = self.embed_session.run(["inputs_embeds"], decoder_inputs)[0]

        return decoder_embeds, decoder_attention_mask

    def process_text_prompt(self, prompt):
        """
        Process text prompt through tokenizer
        """

        # Translate known task tokens to actual prompts
        prompt = self.get_task_prompt(prompt)

        # Tokenize input text
        tokenized = self.tokenizer(
            prompt,
            # padding='max_length',
            max_length=128,
            truncation=True,
            return_tensors="np",
        )

        # Ensure correct dtypes
        input_ids = tokenized["input_ids"].astype(np.int64)
        attention_mask = tokenized["attention_mask"].astype(np.int64)

        print(f"\nText prompt: {prompt}")
        print(f"Input token ids shape: {input_ids.shape}")
        print(f"Input tokens: {self.tokenizer.convert_ids_to_tokens(input_ids[0])}")

        return input_ids, attention_mask

    def generate_initial_decoder_ids(self, batch_size=1):
        """
        Generate initial decoder input ids
        """
        # Start with decoder_start_token_id from config
        decoder_input_ids = np.array(
            [[self.decoder_start_token_id]] * batch_size, dtype=np.int64
        )
        return decoder_input_ids

    def get_next_token(self, logits):
        """
        Get the next token from logits using greedy selection
        """
        # Get the last prediction (shape: [1, vocab_size])
        next_token_logits = logits[:, -1, :]

        # Get the token with highest probability
        next_token = np.argmax(next_token_logits, axis=-1)
        return next_token

    def get_task_prompt(self, task_token):
        """
        Convert task token to actual prompt
        """
        return self.task_prompts_without_inputs.get(task_token, task_token)

    def run_inference(self, image, prompt):
        """
        Run the complete inference pipeline
        """
        # 1. Process image through vision encoder
        image_input = self.preprocess_image(image)
        vision_outputs = self.vision_session.run(
            ["image_features"], {"pixel_values": image_input}
        )
        image_features = vision_outputs[0]

        # Verify vision features
        # self.verify_vision_features(image_features)

        # 2. Process text prompt
        input_ids, attention_mask = self.process_text_prompt(prompt)

        # 3. Get text embeddings
        text_embed_outputs = self.embed_session.run(
            ["inputs_embeds"], {"input_ids": input_ids}
        )
        inputs_embeds = text_embed_outputs[0]
        print(f"Text input embeddings shape: {inputs_embeds.shape}")

        # 4. Run through encoder with combined features
        # Concatenate image features and text embeddings
        combined_features = np.concatenate([image_features, inputs_embeds], axis=1)

        # Create combined attention mask
        image_attention_mask = np.ones((1, image_features.shape[1]), dtype=np.int64)
        combined_attention_mask = np.concatenate(
            [image_attention_mask, attention_mask], axis=1
        )
        print(f"Combined attention mask shape: {combined_attention_mask.shape}")

        # 4. Run through encoder
        encoder_outputs = self.encoder_session.run(
            ["last_hidden_state"],
            {
                "attention_mask": combined_attention_mask,
                "inputs_embeds": combined_features,
            },
        )
        encoder_hidden_states = encoder_outputs[0]
        print(f"Encoder hidden states shape: {encoder_hidden_states.shape}")

        # 5. Prepare decoder inputs
        decoder_input_ids = self.generate_initial_decoder_ids()
        decoder_embeds, _ = self.create_decoder_inputs(
            decoder_input_ids
        )  # We don't use the mask
        print(f"Decoder embeddings shape: {decoder_embeds.shape}")

        # 6. Run decoder for first step
        decoder_outputs = self.decoder_session.run(
            ["logits"],
            {
                "encoder_attention_mask": combined_attention_mask,
                "encoder_hidden_states": encoder_hidden_states,
                "inputs_embeds": decoder_embeds,
            },
        )

        # Get logits from decoder output
        logits = decoder_outputs[0]
        print(f"Output logits shape: {logits.shape}")

        return {
            "image_features": image_features,
            "text_embeds": inputs_embeds,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": combined_attention_mask,
            "logits": logits,
        }

    def generate_text(self, initial_outputs, max_length=50):
        """
        Generate text using the outputs from an initial inference pass
        """
        # Initialize generation
        generated_token_ids = [self.decoder_start_token_id]

        # Get encoder outputs from initial inference
        encoder_hidden_states = initial_outputs["encoder_hidden_states"]
        # encoder_attention_mask = np.ones((1, encoder_hidden_states.shape[1]), dtype=np.int64)  # Match encoder sequence length
        # Create encoder attention mask that matches encoder sequence length
        encoder_attention_mask = initial_outputs["encoder_attention_mask"]

        # Print shapes for debugging
        print(f"\nGeneration shapes:")
        print(f"Encoder hidden states: {encoder_hidden_states.shape}")
        print(f"Encoder attention mask: {encoder_attention_mask.shape}")

        for _ in range(max_length):
            # Prepare decoder inputs
            decoder_input_ids = np.array([generated_token_ids], dtype=np.int64)
            decoder_embeds, _ = self.create_decoder_inputs(
                decoder_input_ids
            )  # We don't use the mask

            # Run decoder with correct inputs
            decoder_outputs = self.decoder_session.run(
                ["logits"],
                {
                    "encoder_attention_mask": encoder_attention_mask,
                    "encoder_hidden_states": encoder_hidden_states,
                    "inputs_embeds": decoder_embeds,
                },
            )
            logits = decoder_outputs[0]

            # Get next token (fix deprecation warning)
            next_token = self.get_next_token(logits)
            next_token_scalar = next_token.item()  # Convert to scalar

            # Debug print with scalar token
            print(f"Generated token: {self.tokenizer.decode([next_token_scalar])}")

            # If we hit the EOS token, stop generating
            if next_token_scalar == self.eos_token_id:
                break

            # Add the token to our generated sequence
            generated_token_ids.append(next_token_scalar)

        # Decode the sequence
        decoded_text = self.tokenizer.decode(
            generated_token_ids, skip_special_tokens=True
        )

        generation_info = {
            "token_ids": generated_token_ids,
            "decoded_text": decoded_text,
        }

        return generation_info


# Example usage:
if __name__ == "__main__":
    # Initialize the model
    model = Florence2Reference()

    # Load test image and define prompt
    image_path = "./data/test_data/car.jpg"
    print(f"Image path exists: {os.path.exists(image_path)}")
    print(f"Full image path: {os.path.abspath(image_path)}")

    # Test with different prompts from documentation
    prompts = [
        # "<CAPTION>",  # Basic caption
        # "<DETAILED_CAPTION>",  # Detailed caption
        # "<MORE_DETAILED_CAPTION>",  # Very detailed caption
        # "<OD>",  # Object detection
        # "<OCR>",  # OCR
        # "What can you see in this image?",  # Generic question
        # "What does the image describe?",  # Another generic question

        "Some example with region <loc_999><loc_999>",

        #task_prompt = '<REGION_TO_CATEGORY>' results = run_example(task_prompt, text_input="<loc_52><loc_332><loc_932><loc_774>")
    ]

    for prompt in prompts:
        print(f"\n\nTesting with prompt: {prompt}")
        outputs = model.run_inference(image_path, prompt)

        print("\nInitial outputs shapes:")
        for key, value in outputs.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {value} ({type(value)})")

        text_result = model.generate_text(outputs)
        print("\nGenerated text:", text_result["decoded_text"])
        print("Token IDs:", text_result["token_ids"])

        print("------------------------------------------------------")
