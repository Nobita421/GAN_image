# Running GAN Image Generator in Google Colab

This guide explains how to run the project in Google Colab using the free GPU tier.

## 1. Setup

1.  **Clone the repository:**
    ```bash
    !git clone https://github.com/Nobita421/GAN_image.git
    %cd GAN_image
    ```

2.  **Install dependencies:**
    ```bash
    !pip install -r requirements.txt
    ```

3.  **Enable GPU:**
    *   Go to **Runtime** > **Change runtime type**
    *   Select **T4 GPU** (or any available GPU)
    *   Click **Save**

## 2. Data Preparation

Since you cannot easily upload a large dataset folder to Colab directly, you have two options:

### Option A: Use Dummy Data (For Testing)
Generate random noise images to verify the pipeline works.
```bash
!python setup_dummy_data.py --count 500
```

### Option B: Upload Real Data (Zip File)
1.  Zip your images on your computer (e.g., `images.zip`).
2.  Upload the zip file to the Colab file browser (left sidebar).
3.  Unzip it:
    ```bash
    !unzip images.zip -d data/celeba_preprocessed
    ```

## 3. Training

Run the training script.
```bash
!python train.py
```
*   **Note:** If you see "OUT_OF_RANGE" messages, this is normal TensorFlow behavior at the end of each epoch.
*   Check the `samples/` folder in the file browser to see generated images improving over time.

## 4. Inference & Evaluation

Generate new images using the trained model:
```bash
!python inference.py
```
The generated images will be saved in `samples/`.

Calculate FID score (requires real data):
```bash
!python evaluation.py
```

## 5. Download Results

To download the trained model or samples:
```bash
!zip -r checkpoints.zip checkpoints/
!zip -r samples.zip samples/
```
Then download the zip files from the file browser.
