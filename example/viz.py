import argparse
import ast
import json
import os

import numpy as np
import torch
import torchview
from PIL import Image

from config_networks import CustomNN

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_gif(image_paths, output_gif_path, duration=200):
    images = [Image.open(image_path) for image_path in image_paths]
    # no size shenanigans needed, just save as gif
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,  # 0 means infinite loop
    )


def generate_random_input(batch_size, input_shape, seed=torch.tensor(0)):
    """
    generates random input of a particular shape
    ensures gradients flow to seed
    """
    if type(input_shape) is dict:
        return {k: generate_random_input(batch_size, s, seed=seed) for k, s in input_shape.items()}
    elif type(input_shape[0]) is int:
        shape = tuple([s if s >= 0 else int(torch.randint(20, 30, (1,))) for s in (batch_size,) + tuple(input_shape)])
        return torch.normal(0, 1, shape) + seed
    else:
        return tuple(generate_random_input(batch_size, s, seed=seed) for s in input_shape)


p = argparse.ArgumentParser()
p.add_argument(
    "--config_file",
    nargs="+",
    type=str,
    default=[os.path.join(DIR, "net_configs", filename) for filename in os.listdir(os.path.join(DIR, "net_configs"))],
    help="config files to display (can be json or plaintext)",
)
p.add_argument("--output_dir", type=str, default=os.path.join(DIR, "images"), help="directory to output files")
p.add_argument("--recursion_depth", type=int, default=1000, help="depth to expand nested torch modules")
p.add_argument("--save_gif", action="store_true", help="whether to save gif of scrolling through network (useful for networks that are really long")
p.add_argument("--duration", type=int, default=200, help="duration (ms) of each frame in the gif")
p.add_argument("--scroll", type=int, default=10, help="amount (pixels) to scroll for each frame in the gif")
args = p.parse_args()

for filename in args.config_file:
    model_name, _ = os.path.basename(filename).split(".")

    with open(filename) as f:
        if filename.endswith(".json"):
            structrue = json.load(f)
        else:
            structrue = ast.literal_eval(f.read())
    model = CustomNN(structrue)
    print(f"Visualizing {model_name} with output shape {model.output_shape}")
    x = generate_random_input(1, structrue["input_shape"])
    try:
        model_graph = torchview.draw_graph(
            model,
            input_data=x,
            expand_nested=True,
            save_graph=True,
            directory=args.output_dir,
            filename=f"visualize_{model_name}",
            depth=args.recursion_depth,
            device="cpu",
        )
    except RuntimeError:
        model_graph = torchview.draw_graph(
            model,
            input_data=(x,),
            expand_nested=True,
            save_graph=True,
            directory=args.output_dir,
            filename=f"visualize_{model_name}",
            depth=args.recursion_depth,
            device="cpu",
        )

    if args.save_gif:
        output_gif = os.path.join(args.output_dir, f"visualize_{model_name}.gif")

        img_obj = Image.open(os.path.join(args.output_dir, f"visualize_{model_name}.png"))
        img = np.asarray(img_obj)
        if img.shape[1] > img.shape[0]:
            scroll_dim = 1
        else:
            scroll_dim = 0

        def save_frame(frame_bounds, save_file):
            if scroll_dim == 0:
                temp_img = img[frame_bounds[0] : frame_bounds[1]]
            else:
                temp_img = img[:, frame_bounds[0] : frame_bounds[1]]
            Image.fromarray(temp_img, mode=img_obj.mode).save(save_file)

        img_files = []
        frame_bounds = np.array([0, img.shape[1 - scroll_dim]])
        i = 0
        while frame_bounds is not None:
            fn = os.path.join(args.output_dir, f"temp_img_{i}.png")
            save_frame(frame_bounds, save_file=fn)
            img_files.append(fn)
            if frame_bounds[1] >= img.shape[scroll_dim]:
                frame_bounds = None
                continue
            if i > 0:
                # linger at start
                frame_bounds += args.scroll
            if frame_bounds[1] > img.shape[scroll_dim]:
                frame_bounds = np.array([img.shape[scroll_dim] - img.shape[1 - scroll_dim], img.shape[scroll_dim]])
            i += 1
        i += 1
        # linger at end
        fn = os.path.join(args.output_dir, f"temp_img_{i}.png")
        save_frame(np.array([img.shape[scroll_dim] - img.shape[1 - scroll_dim], img.shape[scroll_dim]]), save_file=fn)
        img_files.append(fn)

        create_gif(image_paths=img_files, output_gif_path=output_gif, duration=args.duration)
        for fn in img_files:
            os.remove(fn)
