import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock, patch
import math

from probity.collection.activation_store import ActivationStore
from probity.probes.linear_probe import (
    BaseProbe,
    LogisticProbe,
    MultiClassLogisticProbe,
    LinearProbe,
    DirectionalProbe,
    LogisticProbeConfig,
)
from probity.training.trainer import (
    BaseTrainerConfig,
    SupervisedTrainerConfig,
    DirectionalTrainerConfig,
    BaseProbeTrainer,
    SupervisedProbeTrainer,
    DirectionalProbeTrainer,
)


# --- Fixtures ---


@pytest.fixture
def base_config():
    return BaseTrainerConfig(device="cpu", num_epochs=2, show_progress=False)


@pytest.fixture
def supervised_config(base_config):
    # Inherit from base_config and add specific fields
    return SupervisedTrainerConfig(
        **base_config.__dict__,
        train_ratio=0.8,
        patience=2,
        min_delta=1e-4,
    )


@pytest.fixture
def directional_config(base_config):
    # Inherit from base_config
    return DirectionalTrainerConfig(**base_config.__dict__)


@pytest.fixture
def mock_activation_store():
    store = MagicMock(spec=ActivationStore)
    # Simulate data: 10 samples, 5 features
    X = torch.randn(10, 5)
    y = torch.randint(0, 2, (10, 1)).float()  # Binary labels
    store.get_probe_data.return_value = (X, y)
    return store


@pytest.fixture
def mock_activation_store_multiclass():
    store = MagicMock(spec=ActivationStore)
    # Simulate data: 10 samples, 5 features, 3 classes
    X = torch.randn(10, 5)
    y = torch.randint(0, 3, (10, 1)).long()  # Multi-class integer labels
    store.get_probe_data.return_value = (X, y)
    return store


# --- Mock Models ---


class MockProbe(BaseProbe):
    def __init__(self, config, input_size=5, output_size=1):
        super().__init__(config)
        self.linear = nn.Linear(input_size, output_size)
        self.config = config  # Store config

    def forward(self, x):
        return self.linear(x)

    def get_loss_fn(self, **kwargs):
        return nn.BCEWithLogitsLoss(**kwargs)

    def _get_raw_direction_representation(self):
        return self.linear.weight.data.clone()

    def _set_raw_direction_representation(self, direction):
        self.linear.weight.data = direction.clone()


class MockLogisticProbe(MockProbe, LogisticProbe):
    def __init__(self, config, input_size=5, output_size=1):
        # Need to call __init__ of both MockProbe and BaseProbe (which LogisticProbe inherits)
        # But MockProbe already calls BaseProbe.__init__
        MockProbe.__init__(self, config, input_size, output_size)
        # LogisticProbe specific init if any (currently none needed for tests)


class MockMultiClassProbe(MockProbe, MultiClassLogisticProbe):
    def __init__(self, config, input_size=5, output_size=3):
        MockProbe.__init__(self, config, input_size, output_size)
        self.config.output_size = output_size  # Make sure output_size is set

    def get_loss_fn(self, **kwargs):
        return nn.CrossEntropyLoss(**kwargs)


class MockLinearProbe(MockProbe, LinearProbe):
    def __init__(self, config, input_size=5, output_size=1):
        MockProbe.__init__(self, config, input_size, output_size)
        self.config.loss_type = "mse"  # Example loss type

    def get_loss_fn(self, **kwargs):
        # Return instance, not class
        return nn.MSELoss(**kwargs)


class MockDirectionalProbe(MockProbe, DirectionalProbe):
    def __init__(self, config, input_size=5):
        # Directional probes don't have a linear layer in the same way initially
        BaseProbe.__init__(self, config)  # Call BaseProbe init directly
        self.config = config
        self.direction = nn.Parameter(torch.randn(1, input_size), requires_grad=False)

    def forward(self, x):
        # Simulate projection onto the direction
        return torch.matmul(x, self.direction.T)

    def fit(self, x, y):
        # Simple mock: calculate mean difference as direction
        y_squeeze = y.squeeze()
        class0_x = x[y_squeeze == 0]
        class1_x = x[y_squeeze == 1]

        # Handle cases where one class might be missing in a small batch/dataset
        if class0_x.nelement() > 0:  # Check if tensor is not empty
            class0_mean = class0_x.mean(dim=0, keepdim=True)
        else:
            class0_mean = torch.zeros(
                1, x.shape[1], device=x.device, dtype=x.dtype
            )  # Use x's device/dtype

        if class1_x.nelement() > 0:
            class1_mean = class1_x.mean(dim=0, keepdim=True)
        else:
            class1_mean = torch.zeros(1, x.shape[1], device=x.device, dtype=x.dtype)

        fitted_direction = class1_mean - class0_mean
        # Return the fitted direction (potentially scaled if standardization happened)
        return fitted_direction

    def _get_raw_direction_representation(self):
        return self.direction.data.clone()

    def _set_raw_direction_representation(self, direction):
        # Ensure direction is the correct shape [1, dim] or [dim] and reshape if needed
        if direction.dim() == 1:
            direction = direction.unsqueeze(0)
        if direction.shape != self.direction.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.direction.shape}, got {direction.shape}"
            )
        self.direction.data = direction.clone()


# --- Test Classes ---


class TestBaseProbeTrainer:

    def test_init(self, base_config):
        trainer = BaseProbeTrainer(base_config)
        assert trainer.config == base_config
        assert trainer.feature_mean is None
        assert trainer.feature_std is None

    @pytest.mark.parametrize(
        "start_lr, end_lr, num_steps, expected_gamma",
        [
            (
                1e-3,
                1e-5,
                10,
                pytest.approx(math.exp(math.log(1e-2) / 10)),
            ),  # Standard decay
            (1e-3, 1e-3, 10, 1.0),  # No decay
            (1e-4, 1e-6, 5, pytest.approx(math.exp(math.log(1e-2) / 5))),
            (1e-3, 1e-6, 0, None),  # Edge case: 0 steps -> ConstantLR expected
            (0, 1e-5, 10, None),  # Edge case: 0 start_lr -> ConstantLR expected
            (1e-3, 0, 10, None),  # Edge case: 0 end_lr -> ConstantLR expected
        ],
    )
    def test_get_lr_scheduler(
        self, base_config, start_lr, end_lr, num_steps, expected_gamma
    ):
        trainer = BaseProbeTrainer(base_config)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

        # Mock print to avoid console output during tests
        with patch("builtins.print") as mock_print:
            scheduler = trainer._get_lr_scheduler(
                optimizer, start_lr, end_lr, num_steps
            )

            if expected_gamma is None:  # Indicates ConstantLR should be used
                assert isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR)
                mock_print.assert_called()  # Check that warning was printed
            else:
                assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
                assert scheduler.gamma == expected_gamma

    @pytest.mark.parametrize(
        "y_data, expected_weights",
        [
            (
                torch.tensor([[0.0], [0.0], [1.0], [0.0]]),
                torch.tensor([3.0 / (1.0 + 1e-8)]),
            ),  # Binary single output
            (
                torch.tensor([[0.0], [0.0], [0.0], [0.0]]),
                torch.tensor([4.0 / (0.0 + 1e-8)]),
            ),  # All negative
            (
                torch.tensor([[1.0], [1.0], [1.0], [1.0]]),
                torch.tensor([0.0 / (4.0 + 1e-8)]),
            ),  # All positive
            (
                torch.tensor([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]),
                torch.tensor([2.0 / (2.0 + 1e-8), 2.0 / (2.0 + 1e-8)]),
            ),  # Multi-output
            (
                torch.tensor([0, 0, 1, 0]),
                torch.tensor([3.0 / (1.0 + 1e-8)]),
            ),  # Binary 1D input
        ],
    )
    def test_calculate_pos_weights(self, base_config, y_data, expected_weights):
        trainer = BaseProbeTrainer(base_config)
        weights = trainer._calculate_pos_weights(y_data.float())
        assert torch.allclose(weights, expected_weights.float(), atol=1e-7)

    @pytest.mark.parametrize(
        "y_data, num_classes, expected_weights",
        [
            (
                torch.tensor([0, 1, 0, 2, 1, 0]),
                3,
                torch.tensor(
                    [
                        4.0 / (3 * 3.0 + 1e-8),
                        4.0 / (3 * 2.0 + 1e-8),
                        4.0 / (3 * 1.0 + 1e-8),
                    ]
                )
                * 6
                / 4,
            ),  # Multi-class standard
            (
                torch.tensor([0, 0, 0]),
                3,
                torch.tensor(
                    [
                        3.0 / (3 * 3.0 + 1e-8),
                        3.0 / (3 * 0.0 + 1e-8),
                        3.0 / (3 * 0.0 + 1e-8),
                    ]
                )
                * 3
                / 3,
            ),  # One class only
            (torch.tensor([]), 3, None),  # Empty input
            (
                torch.tensor([[0], [1], [0], [2]]),
                3,
                torch.tensor(
                    [
                        4.0 / (3 * 2.0 + 1e-8),
                        4.0 / (3 * 1.0 + 1e-8),
                        4.0 / (3 * 1.0 + 1e-8),
                    ]
                )
                * 4
                / 4,
            ),  # Multi-class [N, 1] input
            (torch.tensor([0, 1]).float(), 2, None),  # Incorrect dtype
        ],
    )
    def test_calculate_class_weights(
        self, base_config, y_data, num_classes, expected_weights
    ):
        trainer = BaseProbeTrainer(base_config)
        # Mock print for unsupported dtype warning
        with patch("builtins.print") as mock_print:
            weights = trainer._calculate_class_weights(y_data, num_classes)
            if expected_weights is None:
                assert weights is None
                if y_data.numel() > 0:  # Only expect print if data is not empty
                    mock_print.assert_called()
            else:
                assert weights is not None
                # Need to adjust the expected calculation slightly: it's total_samples / (num_classes * count)
                total_samples = y_data.numel()
                if y_data.dim() == 2 and y_data.shape[1] == 1:
                    y_data = y_data.squeeze(1)
                counts = torch.bincount(y_data.long(), minlength=num_classes)
                expected_tensor = total_samples / (num_classes * (counts + 1e-8))
                assert torch.allclose(weights, expected_tensor.float(), atol=1e-7)

    @pytest.mark.parametrize("optimizer_type", ["Adam", "SGD", "AdamW"])
    def test_create_optimizer(self, base_config, optimizer_type):
        config = BaseTrainerConfig(
            optimizer_type=optimizer_type, learning_rate=0.1, weight_decay=0.01
        )
        trainer = BaseProbeTrainer(config)
        model = nn.Linear(5, 1)  # Simple model with parameters
        optimizer = trainer._create_optimizer(model)

        assert isinstance(optimizer, getattr(torch.optim, optimizer_type))
        assert optimizer.defaults["lr"] == 0.1
        # Weight decay might be handled differently (e.g., AdamW), check param groups
        for group in optimizer.param_groups:
            assert group["weight_decay"] == 0.01
            # Check only trainable parameters are included
            model_params_ids = {id(p) for p in model.parameters() if p.requires_grad}
            optimizer_params_ids = {id(p) for p in group["params"]}
            assert optimizer_params_ids == model_params_ids

    def test_create_optimizer_invalid(self, base_config):
        config = BaseTrainerConfig(optimizer_type="InvalidOptim")
        trainer = BaseProbeTrainer(config)
        model = nn.Linear(5, 1)
        with pytest.raises(ValueError, match="Unknown optimizer type: InvalidOptim"):
            trainer._create_optimizer(model)

    def test_prepare_data_no_standardization(self, base_config, mock_activation_store):
        trainer = BaseProbeTrainer(base_config)
        X_expected, y_expected = mock_activation_store.get_probe_data("pos")
        X_train, y, X_orig = trainer.prepare_data(mock_activation_store, "pos")

        assert torch.equal(X_train, X_expected)
        assert torch.equal(y, y_expected)
        assert torch.equal(X_orig, X_expected)
        assert trainer.feature_mean is None
        assert trainer.feature_std is None

    def test_prepare_data_with_standardization(
        self, base_config, mock_activation_store
    ):
        config = base_config
        config.standardize_activations = True
        config.device = "cpu"
        trainer = BaseProbeTrainer(config)
        X_expected, y_expected = mock_activation_store.get_probe_data("pos")
        X_train, y, X_orig = trainer.prepare_data(mock_activation_store, "pos")

        assert not torch.equal(X_train, X_expected)  # X_train should be standardized
        assert torch.equal(y, y_expected)
        assert torch.equal(X_orig, X_expected)  # X_orig remains original
        assert trainer.feature_mean is not None
        assert trainer.feature_std is not None
        assert trainer.feature_mean.shape == (1, X_expected.shape[1])
        assert trainer.feature_std.shape == (1, X_expected.shape[1])
        # Check if standardization worked (mean approx 0, std approx 1)
        assert torch.allclose(
            X_train.mean(dim=0), torch.zeros(X_expected.shape[1]), atol=1e-6
        )
        assert torch.allclose(
            X_train.std(dim=0), torch.ones(X_expected.shape[1]), atol=1e-6
        )
        # Check stats are on the correct device
        assert trainer.feature_mean.device.type == "cpu"
        assert trainer.feature_std.device.type == "cpu"


class TestSupervisedProbeTrainer:

    def test_init(self, supervised_config):
        trainer = SupervisedProbeTrainer(supervised_config)
        assert trainer.config == supervised_config

    def test_prepare_supervised_data(self, supervised_config, mock_activation_store):
        trainer = SupervisedProbeTrainer(supervised_config)
        train_loader, val_loader = trainer.prepare_supervised_data(
            mock_activation_store, "pos"
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

        # Check dataset sizes (8 train, 2 val based on ratio 0.8 and 10 total samples)
        assert len(train_loader.dataset) == 8
        assert len(val_loader.dataset) == 2

        # Check batch contents (X_train, y, X_orig)
        x_train_batch, y_batch, x_orig_batch = next(iter(train_loader))
        assert (
            x_train_batch.shape[0] <= supervised_config.batch_size
        )  # Check batch size respected
        assert x_train_batch.shape[1] == 5  # Feature dim
        assert y_batch.shape[0] == x_train_batch.shape[0]
        assert y_batch.shape[1] == 1  # Label dim
        assert x_orig_batch.shape == x_train_batch.shape
        assert y_batch.dtype == torch.float32  # Should be float

    def test_prepare_supervised_data_edge_cases(
        self, supervised_config, mock_activation_store
    ):
        # 1. Test train_ratio = 1.0 (should keep 1 for validation)
        config_all_train = supervised_config
        config_all_train.train_ratio = 1.0
        trainer_all_train = SupervisedProbeTrainer(config_all_train)
        with patch("builtins.print") as mock_print:
            train_loader, val_loader = trainer_all_train.prepare_supervised_data(
                mock_activation_store, "pos"
            )
            assert len(train_loader.dataset) == 9  # type: ignore
            assert len(val_loader.dataset) == 1  # type: ignore
            mock_print.assert_called_with(
                "Warning: train_ratio resulted in no validation data. Adjusting to keep one sample for validation."
            )

        # 2. Test train_ratio = 0.0 (should use all for training, warn)
        # Need a fresh config copy for this case
        config_no_train = SupervisedTrainerConfig(
            device=supervised_config.device,
            num_epochs=supervised_config.num_epochs,
            show_progress=supervised_config.show_progress,
            train_ratio=0.0,  # Override ratio
            patience=supervised_config.patience,
            min_delta=supervised_config.min_delta,
            # Add other fields if necessary
        )
        trainer_no_train = SupervisedProbeTrainer(config_no_train)
        with patch("builtins.print") as mock_print:
            train_loader, val_loader = trainer_no_train.prepare_supervised_data(
                mock_activation_store, "pos"
            )
            assert len(train_loader.dataset) == 10  # type: ignore
            assert len(val_loader.dataset) == 10  # type: ignore # Uses training data for validation
            mock_print.assert_any_call(
                "Warning: train_ratio resulted in no training data. Using all data for training."
            )
            mock_print.assert_any_call(
                "Warning: No validation samples after split. Using training data for validation."
            )

    def test_train_epoch(self, supervised_config):
        trainer = SupervisedProbeTrainer(supervised_config)
        model = MockLogisticProbe(LogisticProbeConfig(input_size=5), input_size=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.BCEWithLogitsLoss()
        # Simple dataset/loader for one batch
        X = torch.randn(4, 5)
        y = torch.randint(0, 2, (4, 1)).float()
        loader = DataLoader(TensorDataset(X, y, X), batch_size=4)  # X_train, y, X_orig

        loss = trainer.train_epoch(model, loader, optimizer, loss_fn, 0, 1)
        assert isinstance(loss, float)
        assert loss > 0

    def test_validate(self, supervised_config):
        trainer = SupervisedProbeTrainer(supervised_config)
        model = MockLogisticProbe(LogisticProbeConfig(input_size=5), input_size=5)
        loss_fn = nn.BCEWithLogitsLoss()
        # Simple dataset/loader for one batch
        X = torch.randn(4, 5)  # This will be treated as X_train (ignored)
        y = torch.randint(0, 2, (4, 1)).float()
        X_orig = torch.randn(4, 5)  # This will be used for validation
        loader = DataLoader(TensorDataset(X, y, X_orig), batch_size=4)

        # Mock model's forward call to check if X_orig is used
        model.forward = MagicMock(return_value=torch.randn(4, 1))

        loss = trainer.validate(model, loader, loss_fn)
        assert isinstance(loss, float)
        assert loss > 0
        # Verify model.forward was called with X_orig
        assert model.forward.call_count == 1
        call_args, _ = model.forward.call_args
        assert torch.equal(call_args[0], X_orig)

    @patch("probity.training.trainer.SupervisedProbeTrainer.train_epoch")
    @patch("probity.training.trainer.SupervisedProbeTrainer.validate")
    def test_train_basic_logistic(
        self, mock_validate, mock_train_epoch, supervised_config, mock_activation_store
    ):
        mock_train_epoch.return_value = 0.5  # Mock train loss
        mock_validate.return_value = 0.4  # Mock validation loss
        trainer = SupervisedProbeTrainer(supervised_config)
        probe_config = LogisticProbeConfig(
            input_size=5, model_name="test", hook_point="test"
        )
        model = MockLogisticProbe(probe_config, input_size=5)
        train_loader, val_loader = trainer.prepare_supervised_data(
            mock_activation_store, "pos"
        )

        history = trainer.train(model, train_loader, val_loader)

        assert mock_train_epoch.call_count == supervised_config.num_epochs
        assert mock_validate.call_count == supervised_config.num_epochs
        assert "train_loss" in history
        assert "val_loss" in history
        assert "learning_rate" in history
        assert len(history["train_loss"]) == supervised_config.num_epochs
        assert len(history["val_loss"]) == supervised_config.num_epochs
        assert history["train_loss"][0] == 0.5
        assert history["val_loss"][0] == 0.4
        assert trainer.feature_mean is None  # No standardization by default

    @patch("probity.training.trainer.SupervisedProbeTrainer.train_epoch")
    @patch("probity.training.trainer.SupervisedProbeTrainer.validate")
    def test_train_multiclass(
        self,
        mock_validate,
        mock_train_epoch,
        supervised_config,
        mock_activation_store_multiclass,
    ):
        mock_train_epoch.return_value = 0.8  # Mock train loss
        mock_validate.return_value = 0.7  # Mock validation loss
        trainer = SupervisedProbeTrainer(supervised_config)
        # Adjust config for multiclass
        supervised_config.handle_class_imbalance = True  # Test weight calculation path
        probe_config = LogisticProbeConfig(
            input_size=5, output_size=3, model_name="test", hook_point="test"
        )
        model = MockMultiClassProbe(probe_config, input_size=5, output_size=3)
        train_loader, val_loader = trainer.prepare_supervised_data(
            mock_activation_store_multiclass, "pos"
        )

        # Check that the correct loss is used (mock the get_loss_fn)
        mock_loss_fn = MagicMock(spec=nn.CrossEntropyLoss)
        model.get_loss_fn = MagicMock(return_value=mock_loss_fn)

        history = trainer.train(model, train_loader, val_loader)

        assert mock_train_epoch.call_count == supervised_config.num_epochs
        assert mock_validate.call_count == supervised_config.num_epochs
        assert len(history["train_loss"]) == supervised_config.num_epochs

        # Verify get_loss_fn was called potentially with weights
        model.get_loss_fn.assert_called_once()
        call_kwargs = model.get_loss_fn.call_args.kwargs
        assert "class_weights" in call_kwargs
        assert isinstance(call_kwargs["class_weights"], torch.Tensor)

        # Check that train_epoch and validate were called with is_multi_class=True
        first_train_call_args = mock_train_epoch.call_args_list[0].kwargs
        assert first_train_call_args.get("is_multi_class") is True
        first_val_call_args = mock_validate.call_args_list[0].kwargs
        assert first_val_call_args.get("is_multi_class") is True

    @patch("probity.training.trainer.SupervisedProbeTrainer.train_epoch")
    @patch("probity.training.trainer.SupervisedProbeTrainer.validate")
    def test_train_with_standardization_and_unscaling(
        self, mock_validate, mock_train_epoch, supervised_config, mock_activation_store
    ):
        mock_train_epoch.return_value = 0.5
        mock_validate.return_value = 0.4
        config = supervised_config
        config.standardize_activations = True
        config.device = "cpu"
        trainer = SupervisedProbeTrainer(config)
        probe_config = LogisticProbeConfig(
            input_size=5, model_name="test", hook_point="test"
        )
        model = MockLogisticProbe(probe_config, input_size=5)
        initial_direction = model._get_raw_direction_representation().clone()

        train_loader, val_loader = trainer.prepare_supervised_data(
            mock_activation_store, "pos"
        )

        # Check standardization stats are computed
        assert trainer.feature_mean is not None
        assert trainer.feature_std is not None
        feature_std_copy = trainer.feature_std.clone()  # For checking unscaling

        # Mock the _set_raw_direction_representation to check the final unscaled direction
        model._set_raw_direction_representation = MagicMock()

        # Run training
        history = trainer.train(model, train_loader, val_loader)

        # Check train/validate were called
        assert mock_train_epoch.call_count == config.num_epochs
        assert mock_validate.call_count == config.num_epochs

        # Check that unscaling happened (_set_raw_direction_representation was called)
        model._set_raw_direction_representation.assert_called_once()

        # Check that the passed direction was different from the initial one (scaled)
        # Note: We don't know the exact final scaled direction from train_epoch mock,
        # but we can verify the unscaling logic *would* be applied.
        # Let's simulate a scaled direction and check the unscaling math
        # Get the final learned direction (which was learned on standardized data)
        # In a real scenario, the model's weights would change during train_epoch
        # Here, we assume the initial_direction is the "learned" scaled direction for simplicity of checking the unscaling step
        simulated_learned_scaled_direction = initial_direction.clone()
        expected_unscaled = (
            simulated_learned_scaled_direction / feature_std_copy.squeeze(0)
        )

        # Get the direction passed to _set_raw_direction_representation
        call_args, _ = model._set_raw_direction_representation.call_args
        final_unscaled_direction = call_args[0]

        assert torch.allclose(final_unscaled_direction, expected_unscaled)

    @patch("probity.training.trainer.SupervisedProbeTrainer.train_epoch")
    @patch("probity.training.trainer.SupervisedProbeTrainer.validate")
    def test_train_early_stopping(
        self, mock_validate, mock_train_epoch, supervised_config, mock_activation_store
    ):
        # Simulate validation loss improving then plateauing
        mock_validate.side_effect = [
            0.5,
            0.4,
            0.3,
            0.3,
            0.3,
            0.3,
        ]  # Improves for 2 epochs, then stalls
        mock_train_epoch.return_value = 0.6
        config = supervised_config
        config.num_epochs = 10
        config.patience = 2
        trainer = SupervisedProbeTrainer(config)
        probe_config = LogisticProbeConfig(
            input_size=5, model_name="test", hook_point="test"
        )
        model = MockLogisticProbe(probe_config, input_size=5)
        train_loader, val_loader = trainer.prepare_supervised_data(
            mock_activation_store, "pos"
        )

        with patch("builtins.print") as mock_print:
            history = trainer.train(model, train_loader, val_loader)

        # Stops after epoch 4 (0-indexed): losses 0.5, 0.4, 0.3, 0.3 (counter=1), 0.3 (counter=2 -> stop)
        # Train/validate called for epochs 0, 1, 2, 3, 4
        assert mock_train_epoch.call_count == 5  # Changed from 4 to 5
        assert mock_validate.call_count == 5  # Changed from 4 to 5
        assert len(history["train_loss"]) == 5  # Changed from 4 to 5
        assert len(history["val_loss"]) == 5  # Changed from 4 to 5
        mock_print.assert_called_with(
            "\nEarly stopping triggered after 5 epochs"
        )  # Changed from 4 to 5


class TestDirectionalProbeTrainer:

    def test_init(self, directional_config):
        trainer = DirectionalProbeTrainer(directional_config)
        assert trainer.config == directional_config

    def test_prepare_supervised_data(self, directional_config, mock_activation_store):
        trainer = DirectionalProbeTrainer(directional_config)
        loader1, loader2 = trainer.prepare_supervised_data(mock_activation_store, "pos")

        assert isinstance(loader1, DataLoader)
        assert isinstance(loader2, DataLoader)
        assert loader1 is loader2  # Should return the same loader instance

        # Check loader properties
        assert loader1.batch_size == 10  # Batch size should be full dataset size
        assert len(loader1.dataset) == 10  # type: ignore

        # Check batch contents
        x_train_batch, y_batch, x_orig_batch = next(iter(loader1))
        assert x_train_batch.shape == (10, 5)
        assert y_batch.shape == (10, 1)
        assert x_orig_batch.shape == (10, 5)
        assert y_batch.dtype == torch.float32

    def test_train_no_standardization(self, directional_config, mock_activation_store):
        trainer = DirectionalProbeTrainer(directional_config)
        # Use MagicMock for probe config as DirectionalProbeConfig import was problematic
        probe_config = MagicMock()
        probe_config.input_size = 5
        probe_config.model_name = "test"
        probe_config.hook_point = "test"
        model = MockDirectionalProbe(probe_config, input_size=5)
        initial_raw_direction = (
            model._get_raw_direction_representation().clone()
        )  # Get initial random direction

        # Mock the fit method to check args and return a predictable direction
        mock_fitted_direction = torch.ones(1, 5) * 0.5
        model.fit = MagicMock(return_value=mock_fitted_direction)

        # Mock _set_raw_direction_representation to check what's being set
        model._set_raw_direction_representation = MagicMock()

        train_loader, _ = trainer.prepare_supervised_data(mock_activation_store, "pos")
        x_train_expected, y_expected, x_orig_expected = next(iter(train_loader))

        history = trainer.train(model, train_loader)

        # Check fit was called with correct data (X_train)
        model.fit.assert_called_once()
        call_args, _ = model.fit.call_args
        assert torch.equal(call_args[0], x_train_expected)
        assert torch.equal(call_args[1], y_expected)

        # Check _set_raw_direction_representation was called with the fitted direction
        model._set_raw_direction_representation.assert_called_once()
        set_call_args, _ = model._set_raw_direction_representation.call_args
        assert torch.equal(set_call_args[0], mock_fitted_direction)

        # Check history contains loss (should be calculated based on X_orig and final direction)
        assert isinstance(history, dict)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 1
        assert not torch.isnan(torch.tensor(history["train_loss"][0]))
        assert history["train_loss"][0] == history["val_loss"][0]

    def test_train_with_standardization_and_unscaling(
        self, directional_config, mock_activation_store
    ):
        config = directional_config
        config.standardize_activations = True
        config.device = "cpu"
        trainer = DirectionalProbeTrainer(config)
        # Use MagicMock for probe config
        probe_config = MagicMock()
        probe_config.input_size = 5
        probe_config.model_name = "test"
        probe_config.hook_point = "test"
        model = MockDirectionalProbe(probe_config, input_size=5)

        # Prepare data to get standardization stats
        train_loader, _ = trainer.prepare_supervised_data(mock_activation_store, "pos")
        x_train_standardized, y_expected, x_orig_expected = next(iter(train_loader))

        assert trainer.feature_mean is not None
        assert trainer.feature_std is not None
        feature_std_copy = trainer.feature_std.clone().squeeze()  # Shape [dim]

        # Mock fit to return a direction presumably learned on standardized data
        mock_fitted_scaled_direction = torch.ones(1, 5) * 0.5
        model.fit = MagicMock(return_value=mock_fitted_scaled_direction)

        # Mock set raw direction to check the final unscaled version
        model._set_raw_direction_representation = MagicMock()

        # Mock the forward pass for loss calculation (uses original data)
        model.forward = MagicMock(return_value=torch.randn_like(y_expected))

        # Run training
        history = trainer.train(model, train_loader)

        # Check fit was called with standardized data
        model.fit.assert_called_once()
        call_args, _ = model.fit.call_args
        assert torch.equal(call_args[0], x_train_standardized)
        assert torch.equal(call_args[1], y_expected)

        # Check unscaling happened and the correct direction was set
        expected_unscaled_direction = mock_fitted_scaled_direction / feature_std_copy
        model._set_raw_direction_representation.assert_called_once()
        set_call_args, _ = model._set_raw_direction_representation.call_args
        final_unscaled_direction = set_call_args[0]

        assert torch.allclose(final_unscaled_direction, expected_unscaled_direction)

        # Check forward pass (for loss) was called with original data AFTER direction was set
        model.forward.assert_called_once()
        forward_call_args, _ = model.forward.call_args
        assert torch.equal(forward_call_args[0], x_orig_expected)

        # Check loss calculation
        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert not torch.isnan(torch.tensor(history["train_loss"][0]))
