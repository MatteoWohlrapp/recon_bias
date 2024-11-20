import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
from src.model.reconstruction.style_ganv2 import Discriminator, Generator, ResNetEncoder
from torch import nn, optim
import torch.nn.functional as F
from math import log2
from torch.utils.data import DataLoader

class Trainer:
    def __init__(
        self,
        model,
        samples,
        train_loader,
        train_dataset,
        val_loader,
        num_epochs,
        device,
        log_dir,
        output_dir,
        output_name="model",
        save_interval=1,
        early_stopping_patience=None,
    ):
        """
        Trainer class for training and validating a model with early stopping.

        Args:
            model (torch.nn.Module): The neural network model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            num_epochs (int): Maximum number of epochs to train.
            device (torch.device): Device to run the training on (CPU or GPU).
            log_dir (str): Directory to save TensorBoard logs.
            output_dir (str): Directory to save model checkpoints.
            output_name (str, optional): Base name for the saved model files. Defaults to "model".
            save_interval (int, optional): Interval (in epochs) to save model checkpoints. Defaults to 1.
            early_stopping_patience (int, optional): Number of epochs with no improvement after which training will be stopped. If None, early stopping is disabled.
        """
        self.samples = samples
        self.model = model
        self.critic = Discriminator(512, 1)
        self.enc = ResNetEncoder(512)
        self.gen = Generator(512, 512, 512, 1)
        self.step = None
        self.opt_gen = optim.Adam([{'params': [param for name, param in self.gen.named_parameters() if 'map' not in name]},
                     {'params': self.gen.map.parameters(), 'lr': 1e-5}], lr=1e-4, weight_decay=1e-5)
        self.opt_critic = optim.Adam(
                self.critic.parameters(), lr= 1e-3, weight_decay=1e-4
        )
        self.train_alpha = 1e-7
        self.val_alpha = 1e-7
        self.train_step = 0
        self.val_step = 0
        self.BATCH_SIZES = [256,256,128,64,32,16, 8]
        self.PROGRESSIVE_EPOCHS = [50] * len(self.BATCH_SIZES)
        self.START_TRAIN_IMG_SIZE = 4
        self.train_dataset = train_dataset
        """
        DATSET = '/kaggle/input/women-clothes'
        START_TRAIN_IMG_SIZE = 4
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        LR = 1e-3
        BATCH_SIZES = [256,256,128,64,32,16]
        CHANNELS_IMG = 3
        Z_DIm = 512
        W_DIM = 512
        IN_CHANNELS = 512
        LAMBDA_GP = 10
        PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
        """
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.output_name = output_name
        self.save_interval = save_interval
        self.early_stopping_patience = early_stopping_patience

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Create output directory for checkpoints
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.snapshot_dir = os.path.join(self.output_dir, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Initialize early stopping variables
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.best_model_state = None  # To store the best model's state_dict
        self.best_epoch = None  # To store the epoch number of the best model


    def gradient_penalty(self, real, fake, alpha, train_step, device="cpu"):
        BATCH_SIZE, C, H, W = real.shape
        beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        interpolated_images = real * beta + fake.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)

        # Calculate critic scores
        mixed_scores = self.critic(interpolated_images, alpha, train_step)
    
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty
    
    def get_loader(self, image_size, step):
        batch_size = self.BATCH_SIZES[step]
        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return loader
    
    def train(self):
        step = int(log2(self.START_TRAIN_IMG_SIZE / 4))

        for num_epochs in self.PROGRESSIVE_EPOCHS[step:]:
            alpha = 1e-7
            image_size = 4*2**step
            self.train_loader = self.get_loader(image_size, step)
            print('Curent image size: '+str(image_size))

            for epoch in range(num_epochs):
                alpha = self.train_fn(epoch, step, image_size, alpha)
                
                if epoch % 5 == 0: 
                    self.save_snapshot(epoch, image_size, step, alpha)
            step +=1

    def train_fn(
        self, epoch, step, image_size, alpha
    ):
        train_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch}/{self.PROGRESSIVE_EPOCHS[step]}, Step {step}/{len(self.PROGRESSIVE_EPOCHS) - 1} [Training]"
        )
        for inputs, labels in train_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            inputs = F.interpolate(inputs, size=(image_size, image_size), mode='bilinear', align_corners=False)
            labels = F.interpolate(labels, size=(image_size, image_size), mode='bilinear', align_corners=False)


            cur_batch_size = inputs.shape[0]
            noise = torch.randn(cur_batch_size, 512).to(self.device)
            fake  = self.gen(noise, alpha, step)
            critic_real = self.critic(labels, alpha, step)
            critic_fake = self.critic(fake.detach(), alpha, step)
            gp = self.gradient_penalty(labels, fake, alpha, step, self.device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + 10 * gp
                + (0.001) * torch.mean(critic_real ** 2)
            )

            self.critic.zero_grad()
            loss_critic.backward()
            self.opt_critic.step()

            gen_fake = self.critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

            self.gen.zero_grad()
            loss_gen.backward()
            self.opt_gen.step()

            alpha += cur_batch_size / (
                self.PROGRESSIVE_EPOCHS[step] * 0.5 * self.samples
            )
            alpha = min(alpha,1)


            train_bar.set_postfix(
                gp = gp.item(),
                loss_critic = loss_critic.item(), 
                loss_gen = loss_gen.item(),
            )
        return alpha  
          
        """
        train_loss_d, train_loss_g = self.train_epoch(epoch)

        # Validation phase
        val_loss_d, val_loss_g = self.validate_epoch(epoch)

        # Log metrics
        self.writer.add_scalars(
            "Discriminator loss", {"Train Discriminator": train_loss_d, "Validation Discriminator": val_loss_d}, epoch
        )
        self.writer.add_scalars(
            f"Generator loss", {"Train Generator": train_loss_g, "Validation Generator": val_loss_g}, epoch
        )

        # Save checkpoint at specified intervals
        if epoch % self.save_interval == 0:
            self.save_checkpoint(epoch)
            self.save_snapshot(epoch)

        # Early stopping check
        if val_loss_g < self.best_val_loss:
            self.best_val_loss = val_loss_g
            self.best_model_state = (
                self.model.state_dict()
            )  # Store the model's state_dict
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if (
                self.early_stopping_patience is not None
                and self.epochs_without_improvement >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping triggered after {self.early_stopping_patience} epochs with no improvement."
                )
                break

    # Save the final model
    self.save_checkpoint(epoch, final=True)

    # Save the best model
    if self.best_model_state is not None:
        self.save_best_model()

    self.writer.close()
    """

    def train_epoch(self, epoch):
        self.model.train()
        self.enc.train()
        self.gen.train()
        running_loss_d = 0.0
        running_loss_g = 0.0
        total = 0

        train_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Training]"
        )
        for inputs, labels in train_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            """
            _, fake_discriminator, real_discriminator = self.model(inputs)
            real_labels = torch.ones(labels.size(0), 1).to(self.device)
            fake_labels = torch.zeros(labels.size(0), 1).to(self.device)
            real_loss = self.criterion(real_discriminator, real_labels)
            fake_loss = self.criterion(fake_discriminator.detach(), fake_labels)
            loss_d = real_loss + fake_loss
            self.model.network.discriminator_step(loss_d)

            _, fake_discriminator, _ = self.model(inputs)
            fake_labels = torch.ones(labels.size(0), 1).to(self.device)
            loss_g = self.criterion(fake_discriminator, fake_labels)
            self.model.network.generator_step(loss_g)
            """
            l1_loss = torch.nn.L1Loss()
            w = self.enc(w)
            fake = self.gen(w, self.train_alpha, self.train_step)
            recon_loss = l1_loss(fake, labels)
            loss_g = recon_loss
            self.opt_gen.zero_grad()
            loss_g.backward()
            self.opt_gen.step()

            self.train_step += 1

            
            running_loss_g += loss_g.item()
            #running_loss_d += loss_d.item()
            total += labels.size(0)

            # Update progress bar
            train_bar.set_postfix(
                {
                    "Loss Discriminator": running_loss_d / total,
                    "Loss Generator": running_loss_g / total,
                }
            )

        epoch_loss_d = running_loss_d / total
        epoch_loss_g = running_loss_g / total

        return epoch_loss_d, epoch_loss_g

    def validate_epoch(self, epoch):
        self.model.eval()
        self.enc.eval()
        self.gen.eval()
        running_loss_d = 0.0
        running_loss_g = 0.0
        total = 0
        total = 0

        val_bar = tqdm(
            self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Validation]"
        )
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                """
                _, fake_discriminator, real_discriminator = self.model(inputs)
                real_labels = torch.ones(labels.size(0), 1).to(self.device)
                fake_labels = torch.zeros(labels.size(0), 1).to(self.device)
                real_loss = self.criterion(real_discriminator, real_labels)
                fake_loss = self.criterion(fake_discriminator, fake_labels)
                loss_d = real_loss + fake_loss
                
                fake_image, fake_discriminator, _ = self.model(inputs)
                fake_labels = torch.ones(labels.size(0), 1).to(self.device)
                fake_loss = self.criterion(fake_discriminator, fake_labels)
                reconstruction_loss = torch.nn.L1Loss()(fake_image, labels)
                loss_g = reconstruction_loss"""
                l1_loss = torch.nn.L1Loss()
                w = self.enc(w)
                fake = self.gen(w)
                recon_loss = l1_loss(fake, labels)
                loss_g = recon_loss

                running_loss_g += loss_g.item()
                #running_loss_d += loss_d.item()
                total += labels.size(0)

                # Update progress bar
                val_bar.set_postfix(
                    {
                        "Loss Discriminator": running_loss_d / total,
                        "Loss Generator": running_loss_g / total,
                    }
                )

            epoch_loss_d = running_loss_d / total
            epoch_loss_g = running_loss_g / total

        return epoch_loss_d, epoch_loss_g

    def save_snapshot(self, epoch, image_size, step, alpha):
        """
        Saves a snapshot of the model at the current epoch.

        Args:
            epoch (int): Current epoch number.
        """
        train_snapshot_name = f"{self.output_name}_epoch_{epoch}_step_{step}_snapshot_train"
        train_iter = iter(self.train_loader)

        # Save snapshot for training data
        x, y = next(train_iter)
        if len(x.shape) > 2 and x.shape[0] > 1:
            x = x[0].unsqueeze(0)
            y = y[0].unsqueeze(0)
        x = x.to(self.device)
        y = y.to(self.device)

        x = F.interpolate(x, size=(image_size, image_size), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(image_size, image_size), mode='bilinear', align_corners=False)

        self.enc.eval()
        self.gen.eval()
        with torch.no_grad():
            w = self.enc(x)
            y_pred = self.gen(w, alpha, step)
            path = os.path.join(self.snapshot_dir, train_snapshot_name)
            self.model.save_snapshot(x, y, y_pred, path, self.device, epoch)

    def save_checkpoint(self, epoch, final=False):
        """
        Saves the model checkpoint.

        Args:
            epoch (int): Current epoch number.
            final (bool, optional): If True, saves the model as the final model. Defaults to False.
        """
        if final:
            checkpoint_filename = f"{self.output_name}_epoch_{epoch}_final.pth"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            torch.save(self.model.state_dict(), checkpoint_path)
        else:
            checkpoint_filename = f"{self.output_name}_epoch_{epoch}.pth"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            torch.save(self.model.state_dict(), checkpoint_path)

    def save_best_model(self):
        """
        Saves the best model (based on validation loss) to disk.
        """
        best_model_filename = f"{self.output_name}_epoch_{self.best_epoch}_best.pth"
        best_model_path = os.path.join(self.checkpoint_dir, best_model_filename)
        torch.save(self.best_model_state, best_model_path)
