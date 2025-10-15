from .. import PROJECT_ROOT
from ..core import set_deterministic_seed
from ..core.type_definitions import (
    Scheduler,
    PositiveFloat,
    PositiveInteger,
    StrictlyPositiveFloat,
    StrictlyPositiveInteger,
    assert_annotated_type,
)
from ..loss import Reduction
from ..loss.composite import (
    CompositeLoss,
    CompositeLossConfig,
    CompositeLossKeys,
)
from ..models.registration import (
    RegistrationNetwork,
    RegistrationNetworkConfig,
    RegistrationNetworkOutput,
)
from .checkpointer import Checkpointer, CheckpointerConfig
from .dataloader import (
    MultiModalPersistentDataLoader as MMPDataLoader,
    MultiModalPersistentDataLoaderOutput as MMPDLOutput,
)
from .early_stopping import EarlyStopping, EarlyStoppingMode
from .icebreaker import UnfreezeController
from .logging import TableIdentifiers as LoggingTableIDs
from .logging.base_logger import BaseLogger
from .scheduler import Scheduler
from .utils import (
    MetricAverager,
    save_image_grid,
    split_weight_decay,
    steps_to_epochs,
    validation_image_filename,
)
from .qualitative import (
    GateEdgesMode,
    canny_tolerant_overlay,
    checkerboard_overlay,
    clinical_convention,
)

from dataclasses import asdict, dataclass, fields
from monai.metrics.meandice import DiceMetric
from pathlib import Path
from typing import Annotated, Any, Callable, get_args, get_origin

import operator
import torch
import tqdm

SchedulerHook = tuple[RegistrationNetwork, CompositeLoss]


@dataclass
class TrainingConfig:
    seed: int
    # Loss Function ===========================================================
    loss_config: CompositeLossConfig
    # Training Scheme =========================================================
    early_stopping_min_delta_rel: PositiveFloat
    early_stopping_min_delta_abs: PositiveFloat
    # > Phase A: Linear Probe =================================================
    lp_early_stopping_min_epochs: PositiveInteger
    lp_max_epochs: StrictlyPositiveInteger
    lp_patience_steps: StrictlyPositiveInteger
    lp_learning_rate: StrictlyPositiveFloat
    # > Phase B: Fine-Tuning ==================================================
    ft_early_stopping_min_epochs: PositiveInteger
    ft_max_epochs: StrictlyPositiveInteger
    ft_patience_steps: StrictlyPositiveInteger
    ft_learning_rate: StrictlyPositiveFloat
    ft_unfreeze_plan: list[str]
    ft_unfreeze_learning_rate_multipliers: list[StrictlyPositiveFloat]
    ft_unfreeze_min_delta_rel: PositiveFloat
    ft_unfreeze_min_delta_abs: PositiveFloat
    ft_unfreeze_patience_steps: StrictlyPositiveInteger
    # Defaults ================================================================
    # > Training Scheme =======================================================
    monitored_validation_metric: CompositeLossKeys = CompositeLossKeys.NCC
    early_stopping_mode: EarlyStoppingMode = EarlyStoppingMode.MIN
    # >> Phase A: Linear Probe ================================================
    lp_weight_decay: PositiveFloat = 0.01
    # >> Phase B: Fine-Tuning =================================================
    ft_weight_decay: PositiveFloat = 0.01
    # > Optimisation ==========================================================
    gradient_accumulation_steps: StrictlyPositiveInteger = 1
    gradient_max_norm: StrictlyPositiveFloat = 1.0
    # > Checkpointing =========================================================
    checkpointer_root: Path | None = None
    checkpointer_use_compression: bool = False
    checkpointer_cleanup_on_exit: bool = True
    checkpointer_is_better: Callable[[Any, Any], bool] = operator.lt
    lp_checkpointer_top_k: StrictlyPositiveInteger = 1
    ft_checkpointer_top_k: StrictlyPositiveInteger = 3
    # > Logging ===============================================================
    model_log: str = "gradients"
    model_log_freq_batch: StrictlyPositiveInteger = 50
    save_images_to_disk: bool = True

    def assert_types(self):
        # Assert annotated types.
        for field in fields(self):
            if (origin := get_origin(field.type)) is Annotated:
                assert_annotated_type(
                    getattr(self, field.name),
                    field.type,
                    ValueError(field.name),
                )
            elif (
                origin is list
                and len(args := get_args(field.type)) == 1
                and get_origin(inner_type := args[0]) is Annotated
            ):
                for j, inner_value in enumerate(getattr(self, field.name)):
                    assert_annotated_type(
                        inner_value,
                        inner_type,
                        ValueError(f"{field.name}[{j}]"),
                    )
        # Check unfreezing plans.
        if (plans := len(self.ft_unfreeze_plan)) != (
            mults := len(self.ft_unfreeze_learning_rate_multipliers)
        ):
            raise ValueError(
                f"received {plans} unfreezing plans and {mults} learning rate multipliers"
            )
        elif not plans:
            raise ValueError(f"no unfreezing plans specified!")


def train_for_one_epoch(
    model: RegistrationNetwork,
    dataloader: MMPDataLoader,
    optimiser: torch.optim.Optimizer,
    loss_function: CompositeLoss,
    metric_averager: MetricAverager,
    gradient_accumulation_steps: StrictlyPositiveInteger,
    gradient_max_norm: float,
) -> dict[str, float]:
    """Trains the registration network for a single epoch on the supplied
    DataLoader and returns the mean loss."""
    # Ensure weights are updatable.
    model.train()
    # Zero optimiser gradients.
    optimiser.zero_grad()
    # Reset epoch averager.
    metric_averager.reset()
    # Unload dataloader.
    dl_output: MMPDLOutput
    for step, dl_output in tqdm.tqdm(
        enumerate(dataloader), desc="Training", total=len(dataloader)
    ):
        mri_batch = dl_output.mri
        mri_mask_batch = dl_output.mri_mask
        histology_list = dl_output.histology
        histology_mask_list = dl_output.histology_mask
        # Get model prediction, inject noise.
        # TODO: inject noise
        parameters = model.predict_rigid_transform_parameters(
            mri_batch, histology_list
        )
        thetas = model.stn.build_theta(parameters)
        pred = RegistrationNetworkOutput(
            warped_histology_batch=model.apply_rigid_transform(
                thetas, tensors=histology_list, grid_sampling_mode="bilinear"
            ),
            warped_histology_mask_batch=model.apply_rigid_transform(
                thetas,
                tensors=histology_mask_list,
                grid_sampling_mode="bilinear",
            ),
            parameters=parameters,
            thetas=thetas,
        )
        # Compute and record loss.
        loss: torch.Tensor = loss_function(
            y_true=mri_batch, pred=pred, mask=mri_mask_batch
        )
        metric_averager.update(
            **{f"train/{k}": v for k, v in loss_function.latest.items()}
        )
        # Backpropagate.
        (loss / gradient_accumulation_steps).backward()
        # Clip gradients, advance optimiser, and zero gradients
        # on gradient accumulation boundary or final batch.
        boundary = (step + 1) % gradient_accumulation_steps == 0
        last = step + 1 == len(dataloader)
        if boundary or last:
            # Clip gradients 👷🥽🦺.
            if gradient_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_max_norm
                )
            # Advance optimiser.
            optimiser.step()
            # Zero gradients.
            optimiser.zero_grad()
    # Return mean metrics.
    return metric_averager.mean()


def evaluate(
    model: RegistrationNetwork,
    dataloader: MMPDataLoader,
    loss_function: CompositeLoss,
    metric_averager: MetricAverager,
    dice_metric: DiceMetric,
    epoch: int,
    save_to_disk: tuple[str, Path, Path, Path] | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluates the model's performance on the supplied DataLoader and
    returns the mean metrics and qualitative visualisations."""
    # Ensure weights don't update.
    model.eval()
    # Reset metric averager.
    metric_averager.reset()
    # Prepare table rows.
    table_rows: list[dict[str, Any]] = []
    # Run validation.
    with torch.no_grad():
        dl_output: MMPDLOutput
        for batch, dl_output in tqdm.tqdm(
            enumerate(dataloader), desc="Validation", total=len(dataloader)
        ):
            # Unload dataloader.
            ids = dl_output.correspondence_id
            mri_batch = dl_output.mri
            mri_mask_batch = dl_output.mri_mask
            histology_list = dl_output.histology
            histology_mask_list = dl_output.histology_mask
            histology_rgb_list = dl_output.histology_rgb
            assert histology_rgb_list is not None
            # Get prediction.
            pred: RegistrationNetworkOutput = model(
                mri_batch, histology_list, histology_mask_list
            )
            # Compute and record loss.
            loss_function(y_true=mri_batch, pred=pred, mask=mri_mask_batch)
            dice: torch.Tensor = dice_metric(
                mri_mask_batch, pred.warped_histology_mask_batch
            )  # pyright: ignore[reportAssignmentType]
            metric_averager.update(
                **(
                    {f"val/{k}": v for k, v in loss_function.latest.items()}
                    | {"val/dice": dice.mean().item()}
                )
            )
            # Qualitative results.
            # > Warp RGB images and their masks.
            warped_histology_rgb_batch = model.apply_rigid_transform(
                pred.thetas,
                histology_rgb_list,
                grid_sampling_mode="bilinear",
            ).type(torch.uint8)
            warped_histology_mask_batch = model.apply_rigid_transform(
                pred.thetas,
                dl_output.histology_mask,
                grid_sampling_mode="nearest",
            ).type(torch.bool)
            # > Checkerboard overlay.
            checkerboard_batch = checkerboard_overlay(
                mri_batch=mri_batch,
                mri_mask_batch=mri_mask_batch,
                warped_histology_rgb_batch=warped_histology_rgb_batch,
                warped_histology_mask_batch=warped_histology_mask_batch,
                patch_size=16,
                mri_kernel_size=5,
                mri_border_colour=(1, 0, 0),
                histology_kernel_size=3,
                histology_border_colour=(0, 1, 0),
                display_in_clinical_convention=True,
            )
            # > Band-gated Canny overlay.
            canny_band_batch = canny_tolerant_overlay(
                mri_batch,
                mri_mask_batch,
                pred.warped_histology_batch,
                warped_histology_mask_batch,
                sigma_px=loss_function.boundary_heatmap.sigma,
                mode=GateEdgesMode.BAND,
                display_in_clinical_convention=True,
            )
            metric_averager.update(
                **{
                    "val/canny_band_mri_coverage": canny_band_batch.mri_coverage.mean().item(),
                    "val/canny_band_histology_coverage": canny_band_batch.histology_coverage.mean().item(),
                    "val/canny_band_symmetric_coverage": canny_band_batch.symmetric_coverage.mean().item(),
                }
            )
            # > Mask-gated Canny overlay.
            canny_mask_batch = canny_tolerant_overlay(
                mri_batch,
                mri_mask_batch,
                pred.warped_histology_batch,
                warped_histology_mask_batch,
                sigma_px=loss_function.boundary_heatmap.sigma,
                mode=GateEdgesMode.MASK,
                display_in_clinical_convention=True,
            )
            metric_averager.update(
                **{
                    "val/canny_mask_mri_coverage": canny_mask_batch.mri_coverage.mean().item(),
                    "val/canny_mask_histology_coverage": canny_mask_batch.histology_coverage.mean().item(),
                    "val/canny_mask_symmetric_coverage": canny_mask_batch.symmetric_coverage.mean().item(),
                }
            )
            # > Create table rows.
            cc_mri_batch = clinical_convention(mri_batch)
            cc_mri_mask_batch = clinical_convention(mri_mask_batch)
            cc_pred_warped_histology_batch = clinical_convention(
                pred.warped_histology_batch
            )
            cc_pred_warped_histology_mask_batch = clinical_convention(
                pred.warped_histology_mask_batch
            )
            for j, id_ in enumerate(ids):
                table_rows.append(
                    {
                        LoggingTableIDs.ID.value: id_,
                        LoggingTableIDs.MRI.value: cc_mri_batch[j],
                        LoggingTableIDs.MRI_MASK.value: cc_mri_mask_batch[j],
                        LoggingTableIDs.HISTOLOGY.value: cc_pred_warped_histology_batch[
                            j
                        ],
                        LoggingTableIDs.HISTOLOGY_MASK.value: cc_pred_warped_histology_mask_batch[
                            j
                        ],
                        LoggingTableIDs.CHECKERBOARD.value: checkerboard_batch[
                            j
                        ],
                        LoggingTableIDs.CANNY_BAND.value: canny_band_batch.overlay[
                            j
                        ],
                        LoggingTableIDs.CANNY_MASK.value: canny_mask_batch.overlay[
                            j
                        ],
                    }
                )
            # > Save to disk?
            if save_to_disk is None:
                continue
            suffix, checkerboard_dir, canny_band_dir, canny_mask_dir = (
                save_to_disk
            )

            # >>>> MONKEY PATCHING IN
            if epoch == 1:
                root = checkerboard_dir.parent  # "wlog"
                save_image_grid(
                    cc_mri_batch,
                    root
                    / validation_image_filename(
                        name="mri", epoch=epoch, batch=batch, suffix=suffix
                    ),
                )
                save_image_grid(
                    cc_mri_mask_batch.float(),
                    root
                    / validation_image_filename(
                        name="mri_mask", epoch=epoch, batch=batch, suffix=suffix
                    ),
                )
            hist_dir = checkerboard_dir.parent / "hist"
            hist_dir.mkdir(exist_ok=True, parents=True)
            save_image_grid(
                clinical_convention(warped_histology_rgb_batch.float() / 255),
                hist_dir
                / validation_image_filename(
                    name="warped_hist", epoch=epoch, batch=batch, suffix=suffix
                ),
            )
            save_image_grid(
                clinical_convention(warped_histology_mask_batch.float()),
                hist_dir
                / validation_image_filename(
                    name="warped_hist_mask",
                    epoch=epoch,
                    batch=batch,
                    suffix=suffix,
                ),
            )

            # >>>> MONKEY PATCHING OUT

            # >> Checkerboard
            save_image_grid(
                checkerboard_batch,
                checkerboard_dir
                / validation_image_filename(
                    name="checkerboard", epoch=epoch, batch=batch, suffix=suffix
                ),
            )
            # >> Canny band overlay
            save_image_grid(
                canny_band_batch.overlay,
                canny_band_dir
                / validation_image_filename(
                    name="canny_band", epoch=epoch, batch=batch, suffix=suffix
                ),
            )
            # >> Canny mask overlay
            save_image_grid(
                canny_mask_batch.overlay,
                canny_mask_dir
                / validation_image_filename(
                    name="canny_mask", epoch=epoch, batch=batch, suffix=suffix
                ),
            )
    # Return mean loss value.
    return metric_averager.mean(), table_rows


def train_model(
    training_config: TrainingConfig,
    model_config: RegistrationNetworkConfig,
    logger: BaseLogger,
    training_dataloader: MMPDataLoader,
    validation_dataloader: MMPDataLoader,
    lp_scheduler_factories: (
        list[Callable[[SchedulerHook], Scheduler]] | None
    ) = None,
    ft_scheduler_factories: (
        list[Callable[[SchedulerHook], Scheduler]] | None
    ) = None,
    verbose: bool = False,
):
    """Run the two-phase training pipeline on a model."""
    # ========== Type Checking ==========================================================
    training_config.assert_types()
    assert validation_dataloader.visualisation
    if lp_scheduler_factories is None:
        lp_scheduler_factories = []
    if ft_scheduler_factories is None:
        ft_scheduler_factories = []
    # ========== Training Prep ==========================================================
    # Serialise training config.
    training_config_dict = asdict(training_config)
    # Start logger.
    logger.start()
    # Create checkpointing directory if not specified.
    if training_config.checkpointer_root is None:
        root = PROJECT_ROOT / "checkpoints" / logger.run_id()
        root.mkdir(exist_ok=True, parents=True)
        training_config.checkpointer_root = root
    # Create checkpointers.
    monitored_metric = f"val/{training_config.monitored_validation_metric}"
    lp_checkpointer = Checkpointer(
        config=CheckpointerConfig(
            root=training_config.checkpointer_root,
            top_k=training_config.lp_checkpointer_top_k,
            is_better=training_config.checkpointer_is_better,
            metric_name=monitored_metric,
            checkpoints_subdir="lp-checkpoints",
        )
    )
    ft_checkpointer = Checkpointer(
        config=CheckpointerConfig(
            root=training_config.checkpointer_root,
            top_k=training_config.ft_checkpointer_top_k,
            is_better=training_config.checkpointer_is_better,
            metric_name=monitored_metric,
            checkpoints_subdir="ft-checkpoints",
        )
    )
    # Optionally create qualitative directories.
    save_to_disk: tuple[Path, Path, Path] | bool
    if training_config.save_images_to_disk:
        checkerboard_dir = training_config.checkpointer_root / "checkerboard"
        checkerboard_dir.mkdir(exist_ok=True, parents=True)
        canny_band_dir = training_config.checkpointer_root / "canny_band"
        canny_band_dir.mkdir(exist_ok=True, parents=True)
        canny_mask_dir = training_config.checkpointer_root / "canny_mask"
        canny_mask_dir.mkdir(exist_ok=True, parents=True)
        save_to_disk = (checkerboard_dir, canny_band_dir, canny_mask_dir)
    else:
        save_to_disk = False
    # Create early stoppers and gradual unfreezing scheduler.
    lp_epoch_early_stopper = EarlyStopping(
        mode=training_config.early_stopping_mode,
        patience=steps_to_epochs(
            steps=training_config.lp_patience_steps,  # LP Patience
            len_dataloader=len(training_dataloader),
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        ),
        min_delta_rel=training_config.early_stopping_min_delta_rel,
        min_delta_abs=training_config.early_stopping_min_delta_abs,
    )
    ft_epoch_early_stopper = EarlyStopping(
        mode=training_config.early_stopping_mode,
        patience=steps_to_epochs(
            steps=training_config.ft_patience_steps,  # FT Patience
            len_dataloader=len(training_dataloader),
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        ),
        min_delta_rel=training_config.early_stopping_min_delta_rel,
        min_delta_abs=training_config.early_stopping_min_delta_abs,
    )
    unfreeze_scheduler = EarlyStopping(
        mode=training_config.early_stopping_mode,
        patience=steps_to_epochs(
            steps=training_config.ft_unfreeze_patience_steps,
            len_dataloader=len(training_dataloader),
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        ),
        min_delta_rel=training_config.ft_unfreeze_min_delta_rel,
        min_delta_abs=training_config.ft_unfreeze_min_delta_abs,
    )
    # Create loss functions, metric averagers, and validation Dice metric.
    training_metric_averager = MetricAverager()
    validation_metric_averager = MetricAverager()
    loss_function = CompositeLoss(config=training_config.loss_config)
    dice_metric = DiceMetric(reduction=Reduction.MEAN)
    # Create model.
    set_deterministic_seed(training_config.seed)
    model = RegistrationNetwork(config=model_config)
    # ⚠️ Freeze encoders and `BatchNorm2d` layers.
    thawer = UnfreezeController(
        model=model, plan=training_config.ft_unfreeze_plan
    )
    # Watch model.
    logger.watch(
        model,
        log=training_config.model_log,
        log_freq_batch=training_config.model_log_freq_batch,
    )
    # Create phase A schedulers.
    lp_scheduler_hook: SchedulerHook = model, loss_function
    lp_schedulers = [
        factory(lp_scheduler_hook) for factory in lp_scheduler_factories
    ]
    # ========== TRAINING ===============================================================
    ERROR = None  # lol
    try:
        # ========== [Phase A] ==========================================================
        set_deterministic_seed(training_config.seed)
        # > Build optimiser parameter groups.
        lp_params = filter(
            lambda tup: "_encoder.conv1" in tup[0]
            or "regression_head" in tup[0],
            model.named_parameters(),
        )
        if training_config.lp_weight_decay:
            decay, no_decay = split_weight_decay(lp_params)
            lp_param_groups = [
                {
                    "name": "lp-wd",
                    "params": decay,
                    "lr": training_config.lp_learning_rate,
                    "weight_decay": training_config.lp_weight_decay,
                },
                {
                    "name": "lp",
                    "params": no_decay,
                    "lr": training_config.lp_learning_rate,
                    "weight_decay": 0.0,
                },
            ]
        else:
            lp_param_groups = [
                {
                    "name": "lp",
                    "params": lp_params,
                    "lr": training_config.lp_learning_rate,
                    "weight_decay": 0.0,
                }
            ]
        optimiser = torch.optim.AdamW(lp_param_groups)
        # > Train.
        for epoch in range(1, training_config.lp_max_epochs + 1):
            # Train for one epoch.
            training_metrics = train_for_one_epoch(
                model=model,
                dataloader=training_dataloader,
                optimiser=optimiser,
                loss_function=loss_function,
                metric_averager=training_metric_averager,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                gradient_max_norm=training_config.gradient_max_norm,
            )
            # Validate.
            validation_metrics, validation_table_rows = evaluate(
                model=model,
                dataloader=validation_dataloader,
                loss_function=loss_function,
                metric_averager=validation_metric_averager,
                dice_metric=dice_metric,
                epoch=epoch,
                save_to_disk=("lp", *save_to_disk) if save_to_disk else None,
            )
            # Log.
            if verbose:
                print(
                    f"[LP] Epoch: {epoch} / {training_config.lp_max_epochs} | Training Loss: {training_metrics['train/' + CompositeLossKeys.LOSS]:.4f}"
                    f" | Validation Loss/Dice: {validation_metrics['val/' + CompositeLossKeys.LOSS]:.4f}/{validation_metrics['val/dice']:.4f}"
                )
            logger.log_validation_table(
                table_name="val/qualitative",
                rows=validation_table_rows,
                epoch=epoch,
            )
            metrics = (
                {"phase": "linear_probe"}
                | training_metrics
                | validation_metrics
            )
            logger.log_epoch_metrics(metrics=metrics, epoch=epoch)
            # Obtain relevant metric.
            relevant_metric: float = metrics[
                monitored_metric
            ]  # pyright: ignore[reportAssignmentType]
            # Checkpointing?
            lp_checkpointer.consider(
                epoch=epoch,
                model=model,
                metric=relevant_metric,
                training_config=training_config_dict,
                optimiser=None,
            )
            # Advance schedulers.
            for scheduler in lp_schedulers:
                scheduler.step()
            # Stopping early?
            if (
                epoch >= training_config.lp_early_stopping_min_epochs
                and lp_epoch_early_stopper.step(relevant_metric)
            ):
                break
        LAST_LP_EPOCH = epoch  # pyright: ignore[reportPossiblyUnboundVariable]

        # ========== Intermission =======================================================
        # Load in best model from Phase A.
        best_lp_checkpoints = lp_checkpointer.topk_descending()
        lp_checkpointer.load(
            checkpoint_path=best_lp_checkpoints[0].filepath,
            model=model,
            optimiser=None,
        )
        # Create phase B noise scheduler.
        ft_scheduler_hook = model, loss_function
        ft_schedulers = [
            factory(ft_scheduler_hook) for factory in ft_scheduler_factories
        ]

        # ========== [Phase B] ==========================================================
        set_deterministic_seed(training_config.seed)
        if thawer.next() is None:
            raise RuntimeError("Failed to perform initial unfreezing!")
        # > Build optimiser parameter groups.
        ft_params = filter(
            lambda tup: "_encoder.conv1" in tup[0]
            or "regression_head" in tup[0],
            model.named_parameters(),
        )
        if training_config.ft_weight_decay:
            decay, no_decay = split_weight_decay(ft_params)
            ft_param_groups = [
                {
                    "name": "ft-wd",
                    "params": decay,
                    "lr": training_config.ft_learning_rate,
                    "weight_decay": training_config.ft_weight_decay,
                },
                {
                    "name": "ft",
                    "params": no_decay,
                    "lr": training_config.ft_learning_rate,
                    "weight_decay": 0.0,
                },
            ]
        else:
            ft_param_groups = [
                {
                    "name": "ft",
                    "params": ft_params,
                    "lr": training_config.ft_learning_rate,
                    "weight_decay": 0.0,
                }
            ]
        for layer, multiplier in zip(
            training_config.ft_unfreeze_plan,
            training_config.ft_unfreeze_learning_rate_multipliers,
        ):
            params = filter(
                lambda tup: f"_encoder.{layer}" in tup[0],
                model.named_parameters(),
            )
            if training_config.ft_weight_decay:
                decay, no_decay = split_weight_decay(params)
                ft_param_groups.extend(
                    (
                        {
                            "name": f"ft-{layer}-wd",
                            "params": decay,
                            "lr": training_config.ft_learning_rate * multiplier,
                            "weight_decay": training_config.ft_weight_decay,
                        },
                        {
                            "name": f"ft-{layer}",
                            "params": no_decay,
                            "lr": training_config.ft_learning_rate * multiplier,
                            "weight_decay": 0.0,
                        },
                    )
                )
            else:
                ft_param_groups.append(
                    {
                        "name": f"ft-{layer}",
                        "params": params,
                        "lr": training_config.ft_learning_rate * multiplier,
                        "weight_decay": 0.0,
                    }
                )
        optimiser = torch.optim.AdamW(ft_param_groups)
        # > Train.
        for epoch in range(1, training_config.ft_max_epochs + 1):
            # Train for one epoch.
            training_metrics = train_for_one_epoch(
                model=model,
                dataloader=training_dataloader,
                optimiser=optimiser,
                loss_function=loss_function,
                metric_averager=training_metric_averager,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                gradient_max_norm=training_config.gradient_max_norm,
            )
            # Validate.
            validation_metrics, validation_table_rows = evaluate(
                model=model,
                dataloader=validation_dataloader,
                loss_function=loss_function,
                metric_averager=validation_metric_averager,
                dice_metric=dice_metric,
                epoch=LAST_LP_EPOCH + epoch,
                save_to_disk=("ft", *save_to_disk) if save_to_disk else None,
            )
            # Log.
            if verbose:
                print(
                    f"[FT] Epoch: {epoch} / {training_config.ft_max_epochs} | Training Loss: {training_metrics['train/' + CompositeLossKeys.LOSS]:.4f}"
                    f" | Validation Loss/Dice: {validation_metrics['val/' + CompositeLossKeys.LOSS]:.4f}/{validation_metrics['val/dice']:.4f}"
                )
            logger.log_validation_table(
                table_name="val/qualitative",
                rows=validation_table_rows,
                epoch=epoch,
            )
            metrics = (
                {"phase": "fine_tune", "epoch_ft": epoch}
                | training_metrics
                | validation_metrics
            )
            metrics.update(thawer.status())
            logger.log_epoch_metrics(
                metrics=metrics, epoch=LAST_LP_EPOCH + epoch
            )
            # Obtain relevant metric.
            relevant_metric: float = metrics[monitored_metric]
            # Checkpointing?
            ft_checkpointer.consider(
                epoch=epoch,
                model=model,
                metric=relevant_metric,
                training_config=training_config_dict,
                optimiser=None,
            )
            # Advance schedulers.
            for scheduler in ft_schedulers:
                scheduler.step()
            # Stopping early?
            if (
                epoch >= training_config.ft_early_stopping_min_epochs
                and thawer.melted()
                and ft_epoch_early_stopper.step(relevant_metric)
            ):
                break
            # Unfreeze next block if on local plateau.
            if unfreeze_scheduler.step(relevant_metric):
                if thawer.next() is not None:
                    # Block was unfrozen, reset scheduler AND
                    # early stopper.
                    unfreeze_scheduler.reset()
                    ft_epoch_early_stopper.reset()

        # ========== Cleanup 🧹 ==========================================================
        # > Optionally compress checkpoints.
        if training_config.checkpointer_use_compression:
            lp_checkpointer.compress()
            ft_checkpointer.compress()
        # > Log checkpoints.
        artefact_name = f"registration-network"
        for k, ckpt in enumerate(best_lp_checkpoints):
            aliases = [f"epoch{ckpt.epoch:04d}", logger.run_id(), "lp"]
            if not k:
                aliases.append("best-lp")
            logger.log_checkpoint(
                path=ckpt.filepath, name=artefact_name, aliases=aliases
            )
        for k, ckpt in enumerate(ft_checkpointer.topk_descending()):
            aliases = [f"epoch{ckpt.epoch:04d}", logger.run_id(), "ft"]
            if not k:
                aliases.append("best-overall")
            logger.log_checkpoint(
                path=ckpt.filepath, name=artefact_name, aliases=aliases
            )
    except Exception as e:
        ERROR = e
        raise
    finally:
        # Final bit of housekeeping.
        logger.finish()
        if ERROR is None and training_config.checkpointer_cleanup_on_exit:
            lp_checkpointer.cleanup()
            ft_checkpointer.cleanup()
    # ========== Done! ==================================================================


def main():
    # Create dataset.
    from .. import PROJECT_ROOT
    from ..datasets.sample_histo_mri.sample_histo_mri import SampleHistoMri
    from ..models.segmentor import Segmentor
    from ..regularisation.rigid_transform import (
        RigidTransformRegularisationLossConfig,
    )
    from ..transforms.preprocessor import Preprocessor, PreprocessorConfig
    from .dataloader import MultiModalPersistentDataLoader as MMPDataLoader
    from .dataset import (
        MultiModalDataset as Dataset,
        MultiModalDatasetView as DatasetView,
    )
    from .data_augmentation import DataAugmentation
    from .logging.wandb_logger import WandBConfig, WandBLogger

    DEVICE = torch.device("cpu")

    cd = SampleHistoMri()
    seg = Segmentor(device=DEVICE)
    preproc = Preprocessor(PreprocessorConfig())
    aug = DataAugmentation.from_yaml(
        filepath=PROJECT_ROOT / "configs" / "baseline-data-augmentation.yaml",
        device=DEVICE,
    )
    dataset = Dataset(
        correspondence_discoverer=cd,
        segmentor=seg,
        preprocessor=preproc,
        device=DEVICE,
    )
    # Split dataset.
    SEED = 0xDEADBEEF
    train = DatasetView(
        dataset, indices=range(len(dataset)), transform=None  # aug.transform()
    )
    val = DatasetView(dataset, indices=range(len(dataset)))

    # Create data loaders.
    BATCH_SIZE = 1
    train_dl = MMPDataLoader(train, visualisation=False, batch_size=BATCH_SIZE)
    val_dl = MMPDataLoader(val, visualisation=True, batch_size=len(val))

    # Create model config.
    model_config = RegistrationNetworkConfig(
        seed=SEED,
        height=preproc.config.mri_patch_size,
        width=preproc.config.mri_patch_size,
        model_name="resnet18",
        regression_head_bottleneck_layer_size=256,
        regression_head_shrinkage_range=(0.05, 0.25),
        device=DEVICE,
    )

    # Create phase A schedulers.
    from .scheduler import CosineAnnealing, PulseWindowScheduler, StepValue

    def lp_sobel_scheduler_factory(hook: SchedulerHook) -> Scheduler:
        _, loss = hook

        def update_fn(x: PositiveFloat):
            loss.config.sobel_weight = x

        values = {10: 0.2}
        return StepValue(
            update_fn=update_fn,
            initial_weight=0.0,
            total_epochs=training_config.lp_early_stopping_min_epochs,
            value_at_epochs={e - 1: v for e, v in values.items()},
        )

    def lp_noise_scheduler_factory(
        hook: SchedulerHook,
    ) -> Scheduler:
        m, _ = hook

        def update_fn(x: PositiveFloat):
            m.regression_head.config.noise_weight = x

        return PulseWindowScheduler(
            update_fn=update_fn,
            initial_weight=0.0,  # want nothing outside of windows
            total_epochs=training_config.lp_early_stopping_min_epochs,
            pulse_windows=[
                (
                    epoch,
                    epoch + 4,
                    CosineAnnealing,
                    {
                        "initial_weight": 1.0,
                    },
                )
                for epoch in (22, 34, 46, 58, 70)
            ],
        )

    lp_scheduler_factories = [
        # lp_noise_scheduler_factory,
        # lp_sobel_scheduler_factory,
    ]

    # Create phase B schedulers.
    ft_scheduler_factories = []

    # Create loss function.
    loss_config = CompositeLossConfig(
        ncc_weight=1,
        sobel_weight=0,
        boundary_heatmap_weight=0,
        transformation_parameters_weight=1,
        # ====
        transformation_regularisation_config=RigidTransformRegularisationLossConfig(
            translation_weight=1,
            scale_weight=1,
            scale_log_prior_confidence_interval=0.95,
            regression_head_shrinkage_range=model_config.regression_head_shrinkage_range,
        ),
    )

    # Create training configuration.
    training_config = TrainingConfig(
        seed=SEED,
        gradient_accumulation_steps=1,
        gradient_max_norm=1.0,
        # Loss Function ===========================================================
        loss_config=loss_config,
        # Checkpointer ============================================================
        checkpointer_use_compression=False,
        # Logger ==================================================================
        model_log="gradients",
        model_log_freq_batch=1,
        # Training ===
        monitored_validation_metric=CompositeLossKeys.NCC,
        early_stopping_mode=EarlyStoppingMode.MIN,
        early_stopping_min_delta_rel=1e-3,
        early_stopping_min_delta_abs=1e-4,
        # > Phase A: Linear Probe =================================================
        lp_checkpointer_top_k=1,
        lp_early_stopping_min_epochs=200,
        lp_max_epochs=300,
        lp_patience_steps=10,
        lp_learning_rate=0.001,
        lp_weight_decay=0,
        # > Phase B: Fine-Tuning ==================================================
        ft_checkpointer_top_k=3,
        ft_early_stopping_min_epochs=1,
        ft_max_epochs=1,
        ft_patience_steps=2,
        ft_learning_rate=0.001,
        ft_weight_decay=0.01,
        ft_unfreeze_plan=["layer1", "layer2", "layer3", "layer4"],
        ft_unfreeze_learning_rate_multipliers=[0.75, 0.5, 0.25, 0.1],
        ft_unfreeze_patience_steps=10,
        ft_unfreeze_min_delta_rel=5e-4,
        ft_unfreeze_min_delta_abs=5e-5,
    )

    # Create logger.
    wandb_config = WandBConfig(
        batch_size=BATCH_SIZE,
        training_config=training_config,
        dataset_id=cd.dataset_id,
        dataset_split=(1, 1, 1),
        data_augmentation=aug,
        dir=PROJECT_ROOT / "wandb",
        mode="disabled",
        group="testing",
    )
    logger = WandBLogger(
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        **wandb_config.wandb_init_kwargs(),
    )

    torch.autograd.set_detect_anomaly(True)
    # Train 🦅
    train_model(
        training_config=training_config,
        model_config=model_config,
        logger=logger,
        training_dataloader=train_dl,
        validation_dataloader=val_dl,
        lp_scheduler_factories=lp_scheduler_factories,
        ft_scheduler_factories=ft_scheduler_factories,
        verbose=True,
    )


if __name__ == "__main__":
    main()
