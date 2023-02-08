
from AImed.segmentation import *
from AImed.model.dim2.unet import *
from AImed.model.dim2.utnetv2 import *
from AImed.training.data import CAMUSDataset
from AImed.training.data import CAMUS_DATA

if __name__ == "__main__":

    # these remain constant
    filepath = "/training/"
    filepath_checkpoint = "AImed/model-checkpoints/"

    print(torch.cuda.is_available())
    
    for model_name in ["UTNetV2", "Unet"]:
        for loss_function in ["CE", "Dice", "Focal", "CE+Dice", "CE+EDGE+Dice"]:
            print(loss_function)
            for batch_size in [1,4, 8, 16]:
                for learning_rate in [0.005]:#, 0.0005, 0.00005, 0.000005]:
                    for affine in [
                        [0.6, 1.4, 40, 0.4]]:#,
                        #[0.7, 1.3, 30, 0.3],
                        #[0.8, 1.2, 20, 0.2],
                        #[0.9, 1.2, 10, 0.1],
                    #]:
                        for SNR in [False]:#, False]:
                          for aff in [False]:

                            datamodule = CAMUS_DATA(
                                data_root="AImed/training/camus-dataset",
                                data_root_test = "AImed/testing/camus-test-dataset",
                                batch_size=batch_size,
                                s1=affine[0],
                                s2=affine[1],
                                rotation=affine[2],
                                t=affine[3],
                                SNR=SNR,
                                aff = aff,
                                only_quality=True
                            )
                            model = Segmentation(
                                model=model_name,
                                loss_function=loss_function,
                                weight=[0.5, 1, 1, 1],
                                lr=learning_rate,
                                batch_size=batch_size,
                                data = datamodule
                            )

                            # for saving best model with lowest validation loss,
                            # a lot of parameters in the name so we dont get confused with if we train
                            # different variants
                            checkpoint_callback = ModelCheckpoint(
                                monitor="validation/loss",
                                dirpath=filepath_checkpoint,
                                filename=str(model.name)
                                + "-{epoch:02d}-{val_loss:.6f}-"
                                + str(model.batch_size)
                                + "-"
                                + str(model.loss_function)
                                + "-"
                                + str(affine)
                                + "-"
                                + str(learning_rate)
                                + "-"
                                + str(SNR)
                                + "-"
                                + str(aff)
                                + "-"
                                + str(loss_function),
                            )

                            # api token
                            with open('AImed/key.txt') as f:
                                key = f.readline()
                            f.close()

                            # plug in your own logger, we used neptune, but we removed our secret api token
                            neptune_logger = NeptuneLogger(
                                api_token=key,
                                project="ace-ch/seg",
                                log_model_checkpoints=False
                            )

                            trainer = Trainer(
                                logger=neptune_logger,
                                accelerator="auto",
                                devices=1 if torch.cuda.is_available() else None,
                                max_epochs=10,
                                callbacks=[checkpoint_callback],
                                log_every_n_steps=1,
                                num_sanity_val_steps=0
                            )

                            # train and validate model
                            trainer.fit(
                                model=model, datamodule=datamodule
                            )  