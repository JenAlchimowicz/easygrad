import re
from typing import List, Dict

import numpy as np


# Turns encoder.layer.0.attention.self.query.weight into encoder.layer[0].attention.self.query.weight
def transform_param_name(param_name: str) -> str:
    pattern = re.compile(r"(\.\d+)")
    transformed_param_name = re.sub(pattern, lambda m: f"[{m.group(1)[1:]}]", param_name)
    return transformed_param_name


# Both models need to be initialised
def transfer_huggingface_weights(easy_model, hf_model, to_transpose: List[str] = None):
    to_transpose = to_transpose or []
    for name, param in list(hf_model.named_parameters()):
        easy_param = eval(f"easy_model.{transform_param_name(name)}")
        torch_param = param.detach().numpy()

        if any(name.endswith(x) for x in to_transpose):
            torch_param = torch_param.T

        # Adjust for this: easy: (1,5), torch: (5,)
        elif len(easy_param.shape) == 2 and len(torch_param.shape) == 1:
            torch_param = torch_param[None, :]

        if torch_param.shape != easy_param.shape:
            raise ValueError(f"Wrong parameter shapes. Parameter: {name}. Easy shape: \
                             {easy_param.shape}, torch shape: {torch_param.shape}")

        # Transfer weights
        easy_param.data = torch_param


def compare_easy_to_hf_outputs(easy_output: np.ndarray, hf_output: np.ndarray) -> Dict[str, float]:
    # mean_percentage_error can be very high because if hf_output is close to 0
    # even small error is scaled to some insane numbers. Better look at median.
    return {
        "mean_error": np.mean(np.abs(easy_output - hf_output)),
        "median_error": np.median(np.abs(easy_output - hf_output)),
        "mean_percentage_error": np.mean(np.abs(easy_output - hf_output) / np.abs(hf_output)),
        "median_percentage_error": np.median(np.abs(easy_output - hf_output) / np.abs(hf_output)),
    }
