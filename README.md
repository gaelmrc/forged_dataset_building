
# Forged Dataset Building
This project builds a **forged dataset** from authentic images.  
The pipeline works as follows:

1. Randomly select a **square region** from an authentic image.
2. Generate a **caption** of that region with [GIT (microsoft/git-large-r-coco)](https://huggingface.co/microsoft/git-large-r-coco).
3. Send the caption to a **text-to-image model** ([Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)) to generate a forged patch.
4. Apply **background removal** to keep only the object.
5. Paste the forged patch back into the authentic image and produce:
   - the forged image  
   - the forged square  
   - the authentic square  
   - a mask of the forged region  
   - metadata (JSON + pickle)
  
  <p align="center">
  <img src="images/procedure.Img2Text2Img_process.jpg" alt="" width="85%">
</p>
<p align="center"><em> Complete procedure to forge an image.</em></p>


---
- **Captioning**: [GIT (microsoft/git-large-r-coco)](https://huggingface.co/microsoft/git-large-r-coco)  
- **Text-to-Image**: [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)  

---

## Requirements

You need a **Hugging Face API key** to run this project.

1. Create a [Hugging Face account](https://huggingface.co/join).  
2. Generate an **Access Token** (Profile > Settings > Access Tokens).  
3. Export it as an environment variable:

Linux / macOS:
```bash
export HF_TOKEN=hf_********************************

