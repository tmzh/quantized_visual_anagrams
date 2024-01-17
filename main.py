import argparse
from pathlib import Path
from itertools import chain

import torch
import gc

from diffusers import DiffusionPipeline
from transformers import T5EncoderModel

from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.utils import save_illusion, save_metadata
from visual_anagrams.views import get_views
from visual_anagrams import *

device = 'cuda'


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def im_to_np(im):
    im = (im / 2 + 0.5).clamp(0, 1)
    im = im.detach().cpu().permute(1, 2, 0).numpy()
    im = (im * 255).round().astype("uint8")
    return im


def get_embeds(embeds_dir, prompts):
    for p in prompts:
        if not Path.exists(Path(f'{embeds_dir}/{p}.pt')):
            generate_and_save_embeds(embeds_dir, p)

    embeds = [torch.load(f'{embeds_dir}/{p}.pt') for p in prompts]
    positive_embeds, negative_embeds = zip(*embeds)
    positive_embeds = torch.cat(positive_embeds)
    negative_embeds = torch.cat(negative_embeds)

    return positive_embeds, negative_embeds


def generate_and_save_embeds(embeds_dir, prompt):
    text_encoder = T5EncoderModel.from_pretrained(
        "DeepFloyd/IF-I-M-v1.0",
        subfolder="text_encoder",
        device_map="auto",
        variant="fp16",
        torch_dtype=torch.float16,
    )

    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-M-v1.0",
        text_encoder=text_encoder,  # pass the previously instantiated 8bit text encoder
        unet=None
    )
    embed = pipe.encode_prompt(prompt)
    torch.save(embed, f'{embeds_dir}/{prompt}.pt')

    del text_encoder
    del pipe
    flush()


def get_stages():
    # Stage 1 pipeline
    stage_1 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-M-v1.0",
        text_encoder=None,
        variant="fp16",
        torch_dtype=torch.float16,
    )


    # Stage 2 pipeline
    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-M-v1.0",
        text_encoder=None,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    # stage_2.enable_model_cpu_offload()
    return stage_1.to(device), stage_2.to(device)


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default='results', help='Location to samples and metadata')
    parser.add_argument("--embeds_dir", type=str, default='embeds', help='Location to retrieve embeds from')
    parser.add_argument("--prompts", required=True, type=str, nargs='+',
                        help='Prompts to use, corresponding to each view.')
    parser.add_argument("--views", required=True, type=str, 
                        help='Name of views to use. See `get_views` in `views.py`.')
    parser.add_argument("--style", default='', type=str, help='Optional string to prepend prompt with')
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--reduction", type=str, default='mean')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--noise_level", type=int, default=50, help='Noise level for stage 2')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--save_metadata", action='store_true',
                        help='If true, save metadata about the views. May use lots of disk space, particular for '
                             'permutation views.')

    args = parser.parse_args()

    stop_words = stopwords = ['an', 'a', 'of', 'the' , 'in', 'on']
    name = '.'.join(w for w in chain(*[p.split(' ') for p in [args.views, args.style] + args.prompts]) if w not in stopwords)

    # Create missing dirs
    save_dir = Path(args.save_dir) / name
    save_dir.mkdir(exist_ok=True, parents=True)

    embeds_dir = Path(args.embeds_dir) 
    embeds_dir.mkdir(exist_ok=True, parents=True)

    # Get views
    views = get_views(['identity', args.views])

    # Save metadata
    save_metadata(views, args, save_dir)

    # Get prompt embeddings
    prompts = [f'{args.style} {p}'.strip() for p in args.prompts]
    prompt_embeds, negative_prompt_embeds = get_embeds(args.embeds_dir, prompts)

    stage_1, stage_2 = get_stages()

    # Sample illusions
    for i in range(args.num_samples):
        # Admin stuff
        # generator = torch.manual_seed(args.seed + i)
        generator = torch.Generator()
        seed = generator.seed()
        sample_dir = save_dir / f'{seed:016}'
        sample_dir.mkdir(exist_ok=True, parents=True)

        # Sample 64x64 image
        image = sample_stage_1(stage_1,
                               prompt_embeds,
                               negative_prompt_embeds,
                               views,
                               num_inference_steps=args.num_inference_steps,
                               guidance_scale=args.guidance_scale,
                               reduction=args.reduction,
                               generator=generator)
        save_illusion(image, views, sample_dir)

        # Sample 256x256 image, by upsampling 64x64 image
        image = sample_stage_2(stage_2,
                               image,
                               prompt_embeds,
                               negative_prompt_embeds,
                               views,
                               num_inference_steps=args.num_inference_steps,
                               guidance_scale=args.guidance_scale,
                               reduction=args.reduction,
                               noise_level=args.noise_level,
                               generator=generator)
        save_illusion(image, views, sample_dir)
