CONFIG = {
    "project_name": "supervised-contrastive-reid",
    "n_views": 4,
    "num_inpainted": 2,
    "backbone": "resnet101",
    "num_classes": 751,
    "num_epochs_phase1": 20,
    "num_epochs_phase2": 30,
    "learning_rate": 1e-7,
    "device": "cuda:0",
    "gpus": 1,
    # 'total_views': 6
}
