import hydra
from trainer import Trainer
from linear_trainer import LinearTrainer


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg) -> None:

    """
    we use LinearTrainer for CAP, as we only consider 
    linear probing scenarios for MCI datasets.
    """


    if cfg.train.train_setting=='linear':
        trainer = LinearTrainer(cfg)
        trainer.train()
    else:
        trainer = Trainer(cfg)
        trainer.train() 


if __name__ == "__main__":
    main()
