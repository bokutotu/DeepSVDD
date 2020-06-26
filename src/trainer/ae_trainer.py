from .base import Base

class AETrainer(Base):
    """
    autoencoder trianer for pretrain
    """

    def __init__(self, model, optimizer, lr_scheduler, dataloader, writer, epoch, 
                 save_freq, device, criterion):
        super().__init__(model, optimizer, lr_scheduler, dataloader, writer, epoch,
                         save_freq, device, criterion)

        self.model.to(device)

    def _train_epoch(self):
        mean = 0
        epoch_iter = 0
        for idx, (img,_,_) in enumerate(self.dataloader):
            img = img.to(self.device)
            self.model.zero_grad()
            out = self.model(img)
            loss = self.criterion(out,img)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('autoencoder', loss.mean().item(),self.iter)
            self.writer.add_image('autoencoderout', out[0], self.iter)

            self.iter+=1
            epoch_iter+=1

            mean += loss.mean().item()

        return mean / epoch_iter

