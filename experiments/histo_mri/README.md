# Training with the Histo-MRI correspondences

The training configuration YAML file is `config.yaml`, and the data augmentation configuration YAML file is `data-augmentation.yaml`. By default **no data augmentation** is applied.

Train using

```zsh
uv run -m experiments.histo_mri.main --dataset_dir path/to/HistoMRI
```

and for the version with data augmentation

```zsh
uv run -m experiments.histo_mri.main --dataset_dir path/to/HistoMRI --use_data_augmentation
```

Training occurs on the available CUDA device, if not loudly falls back to the CPU.

By default:

- preprocessed inputs go into the `experiments/histo_mri/cache` directory
- model checkpoints and qualitative visuals (if `save_images_to_disk: true` in `config.yaml`, which it is) are saved into the `experiments/histo_mri/checkpoints` directory

These can be modified via command-line arguments.


