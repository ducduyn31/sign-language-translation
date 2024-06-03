import lightning as L
from datamodule.wlasl import WLASLDataModule
from models.recognition import SignRecognitionNetwork
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import pickle
import torch

def load_word_embedding(path):
    we_dict = pickle.load(open(path, 'rb'))
    embedding = []
    for word in we_dict:
        embedding.append(torch.from_numpy(we_dict[word]))
    return torch.stack(embedding, dim=0).float()


if __name__ == '__main__':
    torch.set_float32_matmul_precision("medium")
    word_embedding = load_word_embedding("checkpoint/wlasl_word_embeddings.pkl") 

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        dirpath='checkpoint',
        filename='sign_recognition_{epoch:02d}_{val_loss:.2f}',
    )
    
    trainer = L.Trainer(
        enable_checkpointing=True,
        max_epochs=10,
        callbacks=[checkpoint],
        # fast_dev_run=True,
        # devices=[0]
    )
    model = SignRecognitionNetwork(
        word_embedding=word_embedding,
        pose_mode_config=dict(
            config_path="configs/rtmpose.py",
            weights_path="checkpoint/rtmpose.pth",
        ),
    )
    state_dict = torch.load("checkpoint/sign_recognition_network.pth")
    model.load_from_pretrained(state_dict["model_state"])
    
    data = WLASLDataModule(
        data_dir="/teamspace/studios/wlasl/WLASL2000/",
        annotations="/teamspace/studios/wlasl/configs/nslt_2000.json",
        batch_size=3,
        num_workers=8,
    )
    trainer.fit(model, datamodule=data, ckpt_path='last')
    trainer.test(model, datamodule=data, ckpt_path=checkpoint.best_model_path)
    # trainer.test(model, datamodule=data)
    
    