
from AImed.segmentation import *
from AImed.model.dim2.unet import *
from AImed.model.dim2.utnetv2 import *
from AImed.training.data import CAMUSDataset
from AImed.training.data import CAMUS_DATA

if __name__ == "__main__":

    # these remain constant
    filepath = "/training/"
    filepath_checkpoint = ".AImed/model-checkpoints/"
    
    for model_name in ["UTNetV2", "Unet"]:
        for loss_function in ["CE","Dice", "Focal", "CE+Dice", "CE+EDGE+Dice"]:
            for batch_size in [4, 8, 16]:
                for learning_rate in [0.005, 0.0005, 0.00005, 0.000005]:
                    for affine in [
                        [0.6, 1.4, 40, 0.4],
                        [0.7, 1.3, 30, 0.3],
                        [0.8, 1.2, 20, 0.2],
                        [0.9, 1.2, 10, 0.1],
                    ]:
                        for SNR in [True, False]:

                            datamodule = CAMUS_DATA(
                                batch_size=batch_size,
                                s1=affine[0],
                                s2=affine[1],
                                rotation=affine[2],
                                t=affine[3],
                                SNR=SNR,
                            )
                            model = Segmentation(
                                model=model_name,
                                loss_function=loss_function,
                                weight=[0.5, 1, 1, 1],
                                lr=learning_rate,
                                batch_size=batch_size,
                            )

                            # for saving best model with lowest validation loss,
                            # a lot of parameters in the name so we dont get confused with if we train
                            # different variants
                            checkpoint_callback = ModelCheckpoint(
                                monitor="D_loss_val",
                                dirpath=filepath_checkpoint,
                                filename=str(model.name)
                                + "-{epoch:02d}-{D_loss_val:.6f}-"
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
                                max_epochs=50,
                                callbacks=[checkpoint_callback],
                                log_every_n_steps=5,
                            )

                            # train and validate model
                            trainer.fit(
                                model=model, datamodule=datamodule
                            )  