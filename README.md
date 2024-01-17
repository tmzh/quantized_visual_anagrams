# Requirements
Python 3.9

# Installation

```cmd
git clone https://github.com/dangeng/visual_anagrams.git
conda env create -f visual_anagrams/environment.yml
conda activate visual_anagrams
pip install -e ./visual_anagrams
pip install -r requirements.txt
```

# Run
Generate images
```cmd
python main.py  --prompts "old man" "girl" --style "an oil painting of" --views identity rotate_180 --num_samples 10 --num_inference_steps 40 --guidance_scale 10.0 --name oil.painting.girl.oldman
```

Generate animation

```cmd
python animate.py --im_path results/campfire.oldman/0001/sample_256.png --metadata_path results/campfire.oldman/metadata.pkl
```