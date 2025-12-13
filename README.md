# Hand-Writing-Reading-Model

A deep learning model for recognizing handwritten words from images using EMNIST dataset. The model can segment individual characters from a word image and predict each character using a CNN trained on EMNIST letters.

## Features

- Train a CNN model on EMNIST letters dataset
- Predict single characters from images
- Predict complete words by segmenting and recognizing each character
- Automatic orientation detection for better accuracy
- Smart preprocessing that preserves character strokes while maintaining EMNIST format

## Installation

1. Clone the repository:
```bash
git clone https://github.com/busenursoker/Hand-Writing-Reading-Model.git
cd Hand-Writing-Reading-Model
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

Download the EMNIST dataset from [Kaggle](https://www.kaggle.com/datasets/crawford/emnist) and place the following files in the `data/` directory:

- `emnist-letters-train.csv`
- `emnist-letters-test.csv`
- `emnist-letters-mapping.txt`

Or use other EMNIST sets (balanced, byclass, bymerge, digits, mnist) by specifying with `--set` flag.

## Usage

### Training

Train the model on EMNIST letters dataset:

```bash
python3 model.py train --set letters --epochs 10
```

Options:
- `--set`: Dataset to use (letters, balanced, byclass, digits, mnist, bymerge). Default: `letters`
- `--epochs`: Number of training epochs. Default: `10`
- `--batch`: Batch size. Default: `128`
- `--lr`: Learning rate. Default: `0.001`
- `--data-dir`: Directory containing EMNIST CSV files. Default: `data`
- `--limit`: Limit number of training samples (for testing)
- `--limit-test`: Limit number of test samples

The trained model will be saved to `artifacts/model.keras` and labels to `artifacts/labels.json`.

### Prediction

#### Predict a single word:

```bash
python3 model.py predict-word --image test.jpg
```

Options:
- `--image`: Path to the input image (required)
- `--infer-orientation`: Orientation fix to apply (`none`, `transpose`, `transpose_fliplr`, `auto`). Default: `none`
- `--dump-chars`: Save processed character images to `artifacts/processed_chars/`
- `--no-debug-boxes`: Don't save debug image with character bounding boxes

#### Predict a single character:

```bash
python3 model.py predict-char --image character.jpg --topk 3
```

Options:
- `--image`: Path to the input image (required)
- `--infer-orientation`: Orientation fix to apply. Default: `auto`
- `--topk`: Number of top predictions to show. Default: `3`

## How It Works

1. **Training**: The model is trained on EMNIST dataset with proper orientation correction (EMNIST images are rotated 90 degrees, so we transpose them during training).

2. **Preprocessing**: 
   - Images are denoised and contrast-enhanced using CLAHE
   - Characters are binarized using Otsu thresholding
   - Aspect ratio is preserved during resizing
   - Final images match EMNIST format: black background with white ink

3. **Word Segmentation**: 
   - Uses morphological operations to separate touching characters
   - Finds bounding boxes for each character
   - Processes each character individually

4. **Orientation Detection**:
   - Automatically detects the best orientation using confidence scores and aspect ratio heuristics
   - Defaults to `none` for real photos (which are already upright)

## Project Structure

```
Hand-Writing-Reading-Model/
├── model.py              # Main training and prediction script
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── data/                 # EMNIST dataset files (not included)
│   ├── emnist-letters-train.csv
│   ├── emnist-letters-test.csv
│   └── emnist-letters-mapping.txt
└── artifacts/            # Generated files (created during training/prediction)
    ├── model.keras       # Trained model
    ├── labels.json       # Character labels mapping
    ├── debug_word_boxes.png
    └── processed_chars/  # Processed character images (if --dump-chars used)
```

## Model Architecture

The model uses a CNN architecture:
- Conv2D (32 filters, 5x5) + MaxPool2D
- Conv2D (48 filters, 5x5) + MaxPool2D
- Flatten
- Dense (256) + ReLU
- Dense (84) + ReLU
- Dense (num_classes) + Softmax

## Notes

- The model works best with non-cursive handwriting
- Images should have good contrast between ink and background
- For best results, use images with clear, separated characters
- The model expects upright text (not rotated)

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
# handwriting
# handwriting
