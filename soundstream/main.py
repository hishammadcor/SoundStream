import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

# Main function to run the LightningCLI
def cli_main():
    # Initialize the LightningCLI with specific configurations
    cli = LightningCLI(
        save_config_kwargs={"overwrite": True},  # Allow overwriting the configuration file
        trainer_defaults={
            'accelerator': 'gpu',  # Use GPU for training
            'strategy': 'ddp_find_unused_parameters_true',  # Use DDP strategy with finding unused parameters
            'log_every_n_steps': 100,  # Log metrics every 100 steps
            'callbacks': [
                # Model checkpointing to save the best models based on validation loss
                ModelCheckpoint(
                    monitor='val_loss_g',  # Monitor validation loss for generator
                    mode='min',  # Save models with minimum validation loss
                    save_top_k=3,  # Save top 3 models
                    save_last=True,  # Always save the last model
                    every_n_epochs=1,  # Save checkpoints every epoch
                    filename='{epoch}-{step}-{val_loss}',  # Filename pattern for saved models
                ),
                # Monitor learning rate during training
                LearningRateMonitor(logging_interval='step'),
                # Summarize the model structure
                ModelSummary(max_depth=4)
            ]
        }
    )

# Entry point of the script
if __name__ == "__main__":
    cli_main()
