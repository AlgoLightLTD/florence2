﻿# Florence-2 Image Processing Project

---

## Project Overview

This project uses the Florence-2 model, a large-scale multimodal vision-language model from Microsoft, to perform various image processing tasks such as object detection, dense region captioning, region proposal, caption-to-phrase grounding, referring expression segmentation, and open vocabulary detection. The project allows users to perform these tasks on a given image and visualize the results in a montage.

### Key Features:

- **Object Detection**: Detects objects in the image and draws bounding boxes around them.
- **Dense Region Captioning**: Provides dense captions for regions within the image.
- **Region Proposal**: Suggests potential regions of interest within the image.
- **Caption-to-Phrase Grounding**: Grounds specific captions to regions in the image.
- **Referring Expression Segmentation**: Segments regions in the image based on a referring expression.
- **Open Vocabulary Detection**: Detects objects described by a user-provided phrase.
- **Montage Visualization**: Displays the results of all tasks in a single montage.

## Installation and Setup

### 1. **Environment Setup**

Ensure you have Python 3.8 or later installed on your system. It is recommended to use a virtual environment to avoid package conflicts.

```bash
python -m venv florence_env
source florence_env/bin/activate  # On Windows use `florence_env\Scripts\activate`
```

### 2. **Install Required Packages**

Install the necessary Python packages using `pip`. The project relies on `transformers`, `torch`, `Pillow`, `matplotlib`, and `requests`.

```bash
pip install torch transformers pillow matplotlib requests
```

### 3. **Install Florence-2 and Workaround for Flash Attention**

Florence-2 model requires a workaround to avoid unnecessary flash attention dependencies. The script includes a fix for this during model loading.

### 4. **Clone or Download the Project**

If you haven't already, clone or download the project files to your local machine.

```bash
git clone https://github.com/your-username/florence-image-processing.git
cd florence-image-processing
```

### 5. **Download and Setup the Florence-2 Model**

The script automatically downloads and saves the Florence-2 model when first run. The model and processor will be saved in the `ckpt/microsoft_Florence-2-large` directory.

## Usage

### 1. **Running the Script**

To run the main script and see the results, simply execute:

```bash
python florence2.py
```

The script will perform all the specified image processing tasks on a sample image and display the results in a montage.

### 2. **Understanding the Script**

- **Model Download and Load**: The model is downloaded and loaded using a custom workaround to handle dependencies related to flash attention.
- **Image Processing**: The `process_image` function performs various tasks on the input image. It handles multiple actions like object detection, segmentation, and captioning.
- **Visualization**: The `display_montage` function displays the results in a montage for easy visualization.

### 3. **Customization**

- **Input Image**: You can change the input image by modifying the `image` variable in the script.
- **Actions**: The actions performed on the image can be customized by specifying a different set of actions in the `process_image` function call.
- **Text Input**: For tasks requiring text input (e.g., open vocabulary detection), modify the `text_input` argument in the `process_image` function.

## Project Structure

```plaintext
florence-image-processing/
├── florence2.py               # Main script containing the processing logic
├── README.md                  # This readme file
└── ckpt/                      # Directory for saving the downloaded model
    └── microsoft_Florence-2-large/
        ├── config.json
        ├── pytorch_model.bin
        └── processor_config.json
```

## Example

The example provided in the script uses a sample image of a car from the Hugging Face dataset. The results include bounding boxes for detected objects, segmented regions based on referring expressions, and caption grounding.

## Troubleshooting

### 1. **Dependencies Not Found**

Ensure all required packages are installed. Use `pip install -r requirements.txt` if you have a `requirements.txt` file.

### 2. **Flash Attention Warning**

You may see a warning about flash attention not being compiled. This is expected due to the workaround applied. The model should still work correctly.

### 3. **Errors During Model Loading**

If you encounter issues during model loading, ensure that the config file is correctly modified for DaViT as outlined in the script.

## Contributing

Contributions to improve this project are welcome! Please fork the repository, make your changes, and submit a pull request.


---

