import os
import json
import torch

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def save_module(model, save_dir, module_name, additional_name='current'):

    check_path(save_dir)
    model_save_dir = os.path.join(save_dir, module_name)
    check_path(model_save_dir)

    if "optim" not in module_name:
        model_save_path = os.path.join(model_save_dir, additional_name)
        check_path(model_save_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_save_path)

    else:
        model_save_path = os.path.join(model_save_dir, additional_name)
        torch.save(
            {
             "epoch": model["epoch"], "style_optim": model["style_optim"].state_dict(),
             "mi_optim": model["mi_optim"].state_dict(),
             "content_optim": model["content_optim"].state_dict(),
             "px_optim": model["px_optim"].state_dict(),
             "style_scheduler": model["style_scheduler"].state_dict(),
             "content_scheduler": model["content_scheduler"].state_dict()
             },
            model_save_path
        )
    print('saved {} at {}'.format(module_name, model_save_path))


def load_module(save_path):
    print(save_path)
    return torch.load(save_path)

def read_files(file_path):
    with open(file_path, "r") as input_file:
        texts_dict = json.load(input_file)
    return texts_dict

