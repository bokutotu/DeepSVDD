from .base import Base

import torch


class Trainer(Base):
    """
    base trainer for Deep SVDD
    """

    def __init__(self, model, optimizer, lr_scheduler, dataloader, writer, criterion, epoch,
                 save_freq, device, objective, nu, R=0, c=None, rep_dim=128,eps=0.1):
        super().__init__(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                         dataloader=dataloader, writer=writer, criterion=criterion,
                         epoch=epoch, save_freq=save_freq, device=device)
        self.objective = objective
        self.nu = nu

        self.R = torch.tensor(R, device=device)
        self.c = torch.tensor(c, device=device) if c is not None else None

        self.eps = eps

        self.rep_dim = rep_dim
        if self.c is None:
            self._init_center_c()

        self.warm_up_n_epochs = 10


    def _init_center_c(self):
        n_samples = 0
        c = torch.zeros(self.rep_dim).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for data in self.dataloader:
                input, _, _ = data
                input = input.to(self.device)
                output = self.model(input)
                n_samples += output.size(0)
                c += torch.sum(output,dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < self.eps) & (c < 0)] = -self.eps
        c[(abs(c) < self.eps) & (c > 0)] = self.eps

        self.c = c

    def _train_epoch(self):
        loss_epoch = 0
        n_batches = 0
        for idx,(img,_,_) in enumerate(self.dataloader):
            img = img.to(self.device)
            self.model.zero_grad()
            outputs = self.model(img)
            dist = torch.sum((outputs - self.c) ** 2, dim=1)
            if self.objective == 'soft-boundary':
                scores = dist - self.R ** 2
                loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
            else:
                loss = torch.mean(dist)
            loss.backward()
            self.optimizer.step()

            # Update hypersphere radius R on mini-batch distances
            if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

            loss_epoch += loss.item()
            n_batches += 1

            self.writer.add_scalar('main model loss', loss.mean().item(),self.iter)

            self.iter += 1

        return loss_epoch / n_batches
