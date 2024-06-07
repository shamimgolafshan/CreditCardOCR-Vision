# CreditCardOCR-Vision
Here's a suggested README file based on the details from your document about the machine vision project, presented in English:

```markdown
# Machine Vision Project: Credit Card Number Recognition

This project is developed as part of the Machine Vision course in the Artificial Intelligence group at the Computer Engineering faculty. It utilizes Optical Character Recognition (OCR) techniques to accurately recognize and extract credit card numbers from images.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Running the Code](#running-the-code)
- [Technologies Used](#technologies-used)
- [Results and Discussion](#results-and-discussion)
- [Further Work](#further-work)
- [Credits](#credits)

## Project Overview
The project focuses on recognizing digits from credit card images. It employs Geometric Transformations and OCR techniques to handle images that are rotated or have different sizes. The project aims to extract credit card numbers by recognizing individual digits using reference OCR images.

## Getting Started
### Prerequisites
- Python 3.x
- Libraries: OpenCV, NumPy, SciPy
- Anaconda (recommended for managing dependencies)

### Installation
1. Clone the repository or download the project files.
2. Install the required Python libraries:
   ```bash
   pip install opencv-python numpy scipy
   ```

## Running the Code
To execute the OCR process:
1. Open the Anaconda Prompt and navigate to the project directory.
2. Run the following command:
   ```bash
   python ocr_class_based.py --image images/credit_card_01.png --reference ocr_a_reference.png
   ```
   This script will process the input image and utilize the specified reference image to identify and extract the credit card numbers.

## Technologies Used
- **Python**: Primary programming language for the project.
- **OpenCV**: Used for image processing operations.
- **NumPy and SciPy**: Used for numerical operations and optimizations.

## Results and Discussion
The project effectively recognizes digits from various credit card images under different conditions. It includes handling images with different orientations and sizes without altering their aspect ratio. Specific parameter tuning, such as the area range and threshold in the `detectMSERFeatures` function, plays a crucial role in the accuracy of the digit recognition:
   ```python
   [mserRegions, mserConnComp] = detectMSERFeatures(I, 'RegionAreaRange',[100, 1000],'ThresholdDelta',9.4);
   ```

## Further Work
Further improvements could include refining the parameter tuning for recognizing digits in more complex backgrounds and under varied lighting conditions. Exploring additional image preprocessing techniques could also enhance the robustness of the OCR process.

## Credits
Project developed by:
- Shamim Golafshan

For more detailed technical insights and discussions, refer to the comments and documentation within the codebase.
```

This README provides a structured overview of your project, instructions on how to get started, and details about the technologies used. It also sets the stage for potential further developments.
