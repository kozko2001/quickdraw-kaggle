{
    "agent": "MnistAgent",
    "exp_name": "mnast-cont-sgd-32",
    "batch_size": 9000,
    "image_size": 32,
    "images_per_class": 2000,
    "data_root": "/home/kozko/tmp/kaggle/quickdraw/input/quickdraw-dataset",
    "learning_rate": 1e-2,
    "optim": "SGD",
    "w_decay": 1e-4,
    "momentum": 0.9,
    "nesterov": true,
    "cuda": true,
    "seed": 42,
    "gpu_device": 0,
    "max_epoch": 60,
    "log_interval": 10,
    "pct_data": 1,
    "num_classes": 340,
    "model": {
        "cls": "MnasNet",
        "input_size": 32
    }
}
