import os
import torch

from pytorch_lightning.utilities.rank_zero import rank_zero_only


@rank_zero_only
def load_checkpoint(conf, seg_module, exit_on_fail=False):
    print("\n" + "#" * 65)
    path = conf['paths']['ckpt_model_path']
    num_classes = len(conf["labels_configs"]["AERIAL_LABEL-COSIA"]["value_name"])

    if not path or not os.path.isfile(path):
        print("Invalid checkpoint path.")
        if exit_on_fail:
            raise SystemExit()
        return

    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)  # Support .ckpt, .pt, .pth

    model_dict = seg_module.state_dict()
    ckpt_classes = next((v.shape[0] for k, v in state_dict.items() if 'classifier.weight' in k or 'criterion.weight' in k), None)

    if ckpt_classes == num_classes:
        seg_module.load_state_dict(state_dict, strict=False)
        print("-----> Loaded weights: matched number of classes. <-----")
    else:
        print(f"------> !! Mismatch: checkpoint has {ckpt_classes}, config has {num_classes}. Adapting... <-----")
        for k in list(state_dict):
            if k in model_dict and state_dict[k].shape != model_dict[k].shape:
                print(f"→ Reinitializing: {k}")
                if 'criterion' in k:
                    state_dict[k] = torch.tensor([conf["classes"][i][0] for i in conf["classes"]])
                else:
                    state_dict[k] = torch.zeros_like(model_dict[k])
        seg_module.load_state_dict(state_dict, strict=False)

    print("#" * 65 + "\n")
