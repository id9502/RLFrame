import os
import math
import torch
import subprocess
import xml.etree.ElementTree

#
# Author : Alwyn Mathew, modified by Mx.Jing
#
# Purpose : GPU status
#

# this is need since we get information from nvidia-smi, which will always uses PCI_BUS_ID order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

__all__ = ["decide_device"]


def _get_gpu_info():

    def extract(elem, tag, drop_s):
        text = elem.find(tag).text
        if drop_s in text:
            text = text.replace(drop_s, "")
        return float(text)

    # https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
    def sm_number(major, minor, mpc):
        if major == 2:  # Fermi
            return mpc * 48 if minor == 1 else mpc * 32
        elif major == 3:  # Kepler
            return mpc * 192
        elif major == 5:  # Maxwell
            return mpc * 128
        elif major == 6:  # Pascal
            return mpc * 128 if minor == 1 else (mpc * 64 if minor == 0 else -1)
        elif major == 7:  # Volta/Turing
            return mpc * 64 if minor == 0 or minor == 5 else -1
        else:
            return -1

    cmd_out = subprocess.check_output(['nvidia-smi', '-q', '-x'])
    gpus = xml.etree.ElementTree.fromstring(cmd_out).findall("gpu")
    return [{
                "gpu_util": extract(g.find("utilization"), "gpu_util", "%"),
                "mem_free": extract(g.find("fb_memory_usage"), "free", "MiB"),
                "sm_clock": extract(g.find("max_clocks"), "sm_clock", "MHz"),
                "sm_number": sm_number(torch.cuda.get_device_properties(i).major,
                                       torch.cuda.get_device_properties(i).minor,
                                       torch.cuda.get_device_properties(i).multi_processor_count),
                "major": torch.cuda.get_device_properties(i).major
            } for i, g in enumerate(gpus)]


def decide_device(device_list=()) -> torch.device:
    if torch.cuda.is_available():
        available_gpu = [dev for dev in device_list if dev < torch.cuda.device_count()]
        if len(available_gpu) == 0:
            return torch.device("cpu")
        gpu_properties = _get_gpu_info()
        max_score = -1
        max_score_id = 0
        for gpu_id in available_gpu:
            score = math.log(gpu_properties[gpu_id]["mem_free"]) +\
                    math.log(gpu_properties[gpu_id]["sm_number"] * gpu_properties[gpu_id]["major"]) / 10. *\
                    math.sqrt(1. - gpu_properties[gpu_id]["gpu_util"] / 100.)
            if score > max_score:
                max_score = score
                max_score_id = gpu_id
        return torch.device(max_score_id)
    else:
        return torch.device("cpu")
