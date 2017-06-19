import os

is_local = "LOCAL" in os.environ

if is_local:
    config = {
        "output_dir": ".",
        "data_root_dir": "/home/faruk/Desktop/experiment_root",
        "inception-top": {
            "batch_size": 2,
            "epoch_n": 1,
            "data_n": 10,
            "repeat-n": 1,
            "steps_per_epoch": 1,
        },
        "deep-conv-autoencoder": {
            "batch_size": 5,
            "epoch_n": 1,
            "data_n": 5000,
        },
    }
else:
    config = {
        "output_dir": "/output",
        "data_root_dir": "/input",
        "inception-top": {
            "batch_size": 300,
            "epoch_n": 75,
            "data_n": 5000,
            "repeat-n": 1,
            "steps_per_epoch": 250,
        },
        "deep-conv-autoencoder": {
            "batch_size": 300,
            "epoch_n": 10,
            "data_n": 5000,
            "repeat-n": 1,
            "steps_per_epoch": 250,
        },
    }