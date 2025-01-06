# Florence-2 Implementation Guide

## 1. Component Overview

### Models and Their Functions

- **Vision Encoder**: DaViT architecture for image feature extraction
- **Token Embedder**: Converts text tokens to embeddings
- **Language Encoder**: BART encoder for text processing
- **Language Decoder**: BART decoder that combines vision and text features for generation

### Data Flow Architecture

1. Image path: Image → Preprocessing → Vision Features → Projection
2. Text path: Prompt → Tokenization → Embeddings → Encoding
3. Combined: Vision Features + Encoded Text → Decoder → Output Tokens → Post-processing

## 2. Image Processing Requirements

### Input Image Preprocessing

- Target size: 768x768 pixels (resize input, stretch or pad)
- Channel order: RGB
- Format conversion: HWC (height, width, channels) → CHW (channels, height, width)
- Normalization:
  - Scale pixel values to range [0,1] (divide by 255)
  - Mean values: [0.485, 0.456, 0.406]
  - Standard deviation: [0.229, 0.224, 0.225]
- Final tensor shape: [batch_size, 3, height, width]
- Data type: float32

## 3. Text Processing Requirements

### Tokenization

- Vocabulary size: 51,289 tokens
- Special tokens:
  - PAD: 1
  - BOS (beginning of sequence): 0
  - EOS (end of sequence): 2
- Task prompts (examples):
  - Caption: "<CAPTION>"
  - Object Detection: "<OD>"
  - OCR: "<OCR>"
  - Detailed Caption: "<DETAILED_CAPTION>"

### Attention Mask Creation

- Shape matches input sequence length
- Values: 1 for valid tokens, 0 for padding
- Used in both encoder and decoder stages

## 4. Model Input/Output Specifications

### Vision Encoder

- Input:
  - Name: "pixel_values"
  - Shape: [batch_size, 3, height, width]
  - Type: float32
- Output:
  - Name: "image_features"
  - Shape: [batch_size, sequence_length, 768]
  - Type: float32

### Token Embedder

- Input:
  - Name: "input_ids"
  - Shape: [batch_size, sequence_length]
  - Type: int64
- Output:
  - Name: "inputs_embeds"
  - Shape: [batch_size, sequence_length, 768]
  - Type: float32

### Language Encoder

- Inputs:
  - Name: "attention_mask"
    - Shape: [batch_size, sequence_length]
    - Type: int64
  - Name: "inputs_embeds"
    - Shape: [batch_size, sequence_length, 768]
    - Type: float32
- Output:
  - Name: "last_hidden_state"
  - Shape: [batch_size, sequence_length, 768]
  - Type: float32

### Language Decoder

- Inputs:
  - Name: "encoder_attention_mask"
    - Shape: [batch_size, sequence_length]
    - Type: int64
  - Name: "encoder_hidden_states"
    - Shape: [batch_size, sequence_length, 768]
    - Type: float32
  - Name: "inputs_embeds"
    - Shape: [batch_size, decoder_sequence_length, 768]
    - Type: float32
- Output:
  - Name: "logits"
  - Shape: [batch_size, sequence_length, vocab_size]
  - Type: float32

## 5. Post-Processing Requirements

### Task-Specific Output Processing

- Caption tasks: Convert logits to text tokens, then to string
- Object Detection: Parse coordinates and labels from output text
- OCR: Extract text and locations
- Coordinates format: Normalized 0-1000 range for boxes and polygons

### Main Post-Processing Types

- pure_text: Direct text output
- description_with_bboxes: Text with bounding boxes
- ocr: Text with quadrilateral regions
- phrase_grounding: Phrases with corresponding regions
- polygons: Mask coordinates for segmentation

## 6. Implementation Considerations

### Performance Optimization Points

- Batch processing capabilities
- Memory management for large images
- Tensor reuse across generation steps
- Attention mask caching

### Error Handling Requirements

- Image format validation
- Token sequence length validation
- Coordinate normalization checks
- Task prompt validation

### Memory Management

- Tensor pooling for repeated operations
- Proper disposal of large tensors
- Batch size considerations

## 7. Validation Points

### Input Validation

- Image dimensions and format
- Text prompt format
- Sequence length limits
- Coordinate ranges

### Output Validation

- Token sequence validity
- Coordinate bounds
- Text decoding integrity
- Task-specific output format compliance

## NOTE

This document was generated and edited using [Claude.ai](https://claude.ai/), and [ChatGPT](https://chatgpt.com/).