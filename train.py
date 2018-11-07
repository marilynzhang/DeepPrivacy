import tensorboardX
import torch
from torch.autograd import Variable
import torchvision
import utils
from utils import init_weights, load_checkpoint, save_checkpoint, to_cuda
from models import Generator, Discriminator, get_transition_value
import tqdm
from torchsummary import summary
import os
from dataloaders import load_mnist, load_cifar10, load_pokemon, load_celeba
from options import load_options, print_options
import time
torch.backends.cudnn.benchmark=True


def gradient_penalty(real_data, fake_data, discriminator, transition_variable):
    epsilon_shape = [real_data.shape[0]] + [1]*(real_data.dim() -1)
    epsilon = torch.rand(epsilon_shape)
    epsilon = to_cuda(epsilon)
    x_hat = epsilon * real_data + (1-epsilon) * fake_data.detach()
    x_hat = to_cuda(Variable(x_hat, requires_grad=True))

    logits, _ = discriminator(x_hat, transition_variable)
    grad = torch.autograd.grad(
        outputs=logits,
        inputs=x_hat,
        grad_outputs=to_cuda(torch.ones(logits.shape)),
        create_graph=True
    )[0].view(x_hat.shape[0], -1)
    grad_penalty = ((grad.norm(p=2, dim=1) - 1)**2)
    return grad_penalty

class DataParallellWrapper(torch.nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.forward_block = torch.nn.DataParallel(self.model)

    def forward(self, x, transition_variable):
        return self.forward_block(x, transition_variable)
    
    def extend(self, channel_size):
        self.model.extend(channel_size)
        self.forward_block = torch.nn.DataParallel(self.model)
    
    def summary(self):
        self.model.summary()


def init_model(imsize, noise_dim, start_channel_dim, image_channels, label_size):
    discriminator = Discriminator(image_channels, imsize, start_channel_dim, label_size)
    discriminator = DataParallellWrapper(discriminator)
    generator = Generator(noise_dim, start_channel_dim, image_channels)
    generator = DataParallellWrapper(generator)
    to_cuda([discriminator, generator])
    #discriminator.apply(init_weights)
    #generator.apply(init_weights)
    discriminator.summary()
    generator.summary()
    return discriminator, generator





def adjust_dynamic_range(data):
    return data*2-1

def save_images(writer, images, global_step, directory):
    imsize = images.shape[2]
    filename = "fakes{0}_{1}x{1}.jpg".format(global_step, imsize)
    filepath = os.path.join(directory, filename)
    torchvision.utils.save_image(images, filepath, nrow=10)
    image_grid = torchvision.utils.make_grid(images, nrow=10)
    writer.add_image("Image", image_grid, global_step)

def normalize_img(image):
    image = (image +1) / 2
    image = utils.clip(image, 0, 1)
    return image


def load_dataset(dataset, batch_size, imsize):
    if dataset == "mnist":
        return load_mnist(batch_size, imsize)
    if dataset == "cifar10":
        return load_cifar10(batch_size, imsize)
    if dataset == "celeba":
        return load_celeba(batch_size, imsize)



def preprocess_images(images, transition_variable):
    images = Variable(images)
    images = to_cuda(images)
    images = adjust_dynamic_range(images)
    # Compute averaged image
    s = images.shape
    y = images.view([-1, s[1], s[2]//2, 2, s[3]//2, 2])
    y = y.mean(dim=3, keepdim=True).mean(dim=5, keepdim=True)
    y = y.repeat([1, 1, 1, 2, 1, 2])
    y = y.view(-1, s[1], s[2], s[3])
    
    images = get_transition_value(y, images, transition_variable)

    return images



class Trainer:


    def __init__(self, options):

        # Set Hyperparameters
        self.batch_size = options.batch_size
        self.dataset = options.dataset
        self.num_epochs = options.num_epochs
        self.label_size = options.label_size
        self.noise_dim = options.noise_dim
        self.learning_rate = options.learning_rate

        # Image settings
        self.current_imsize = options.imsize
        self.image_channels = options.image_channels
        self.max_imsize = options.max_imsize

        # Logging variables
        self.generated_data_dir = options.generated_data_dir
        self.checkpoint_dir = options.checkpoint_dir
        self.summaries_dir = options.summaries_dir
        
        # Transition settings
        self.transition_variable = 1.
        self.transition_iters = 8e5 // options.batch_size * options.batch_size
        self.is_transitioning = False
        self.transition_step = 0
        self.start_channel_size = options.start_channel_size
        current_channels = options.start_channel_size
        self.transition_channels = [
            current_channels,
            current_channels,
            current_channels,
            current_channels//2,
            current_channels//4,
            current_channels//8,
            current_channels//16,
            current_channels//32,
            ]
        self.start_time = time.time()
        if not self.load_checkpoint():
            self.discriminator, self.generator = init_model(options.imsize, options.noise_dim, options.start_channel_size, options.image_channels, options.label_size)
            self.init_optimizers()
        self.data_loader = load_dataset(self.dataset, self.batch_size, self.current_imsize)

                
        self.writer = tensorboardX.SummaryWriter(options.summaries_dir)

        


        self.log_model_graphs()
        
        self.label_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def save_checkpoint(self, epoch):
        filename = "step_{}.ckpt".format(self.global_step)
        filepath = os.path.join(self.checkpoint_dir, filename)
        state_dict = {
            "epoch": epoch + 1,
            "D": self.discriminator.state_dict(),
            "G": self.generator.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            "batch_size": self.batch_size,
            "dataset": self.dataset,
            "num_epochs": self.num_epochs,
            "label_size": self.label_size,
            "noise_dim": self.noise_dim,
            "learning_rate": self.learning_rate,
            "current_imsize": self.current_imsize,
            "max_imsize": self.max_imsize,
            "transition_variable": self.transition_variable,
            "transition_step": self.transition_step,
            "is_transitioning": self.is_transitioning,
            "start_channel_size": self.start_channel_size,
            "global_step": self.global_step,
            "image_channels": self.image_channels,
            "z_sample": self.z_sample.data.cpu().numpy(),
            "total_time": self.total_time

        }
        save_checkpoint(state_dict,
                        filepath,
                        max_keep=2)

    def load_checkpoint(self):
        try:
            ckpt = load_checkpoint(self.checkpoint_dir)
            self.start_epoch = ckpt['epoch']
            print_options(ckpt)
            # Set Hyperparameters
            
            self.batch_size = ckpt["batch_size"]
            self.dataset = ckpt["dataset"]
            self.num_epochs = ckpt["num_epochs"]
            self.label_size = ckpt["label_size"]
            self.noise_dim = ckpt["noise_dim"]
            self.learning_rate = ckpt["learning_rate"]
            self.z_sample = to_cuda(torch.tensor(ckpt["z_sample"]))

            # Image settings
            self.current_imsize = ckpt["current_imsize"]
            self.image_channels = ckpt["image_channels"]
            self.max_imsize = ckpt["max_imsize"]

            # Logging variables
            # Transition settings
            self.transition_variable = ckpt["transition_variable"]
            self.transition_iters = 8e5 // ckpt["batch_size"] * ckpt["batch_size"]
            self.is_transitioning = ckpt["is_transitioning"]
            self.transition_step = ckpt["transition_step"]
            self.global_step = ckpt["global_step"]
            self.total_time = ckpt["total_time"]
            current_channels = ckpt["start_channel_size"]
            self.transition_channels = [
                current_channels,
                current_channels,
                current_channels,
                current_channels//2,
                current_channels//4,
                current_channels//8,
                current_channels//16,
                current_channels//32,
                ]
            self.discriminator, self.generator = init_model(self.current_imsize //(2**self.transition_step), self.noise_dim, current_channels, self.image_channels, self.label_size)
            for i in range(self.transition_step):
                self.discriminator.extend(self.transition_channels[i])
                self.generator.extend(self.transition_channels[i])
            self.discriminator.load_state_dict(ckpt['D'])
            self.generator.load_state_dict(ckpt['G'])
            self.init_optimizers()
            self.generator.summary()
            self.discriminator.summary()            
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
            
            return True
        except Exception as e:
            print(e)
            print(' [*] No checkpoint!')
            labels = torch.arange(0, options.label_size).repeat(100 // options.label_size)[:100].view(-1, 1) if options.label_size > 0 else None
            self.z_sample = self.generate_noise(100, labels)
            self.start_epoch = 0
            self.global_step = 0
            return False
    
    def generate_noise(self, batch_size, labels):
        z = Variable(torch.randn(batch_size, self.noise_dim))
        if self.label_size > 0:
            assert labels.shape[0] == batch_size, "Label size: {}, batch size: {}".format(labels.shape, batch_size)
            labels_onehot = torch.zeros((batch_size, self.label_size))
            idx = range(batch_size)
            labels_onehot[idx, labels.long().squeeze()] = 1
            assert labels_onehot.sum() == batch_size, "Was : {} expected:{}".format(labels_onehot.sum(), batch_size) 
            
            z[:, :self.label_size] = labels_onehot
        z = to_cuda(z)
        return z


    def log_model_graphs(self):
        with tensorboardX.SummaryWriter(self.summaries_dir + "_generator") as w:
            dummy_input = to_cuda(torch.zeros((1, self.generator.model.noise_dim, 1, 1)))
            w.add_graph(self.generator.model, input_to_model=dummy_input)
        with tensorboardX.SummaryWriter(self.summaries_dir + "_discriminator") as w:
            imsize= self.discriminator.model.current_input_imsize
            image_channels = self.discriminator.model.image_channels
            dummy_input = to_cuda(torch.zeros((1, 
                                               image_channels,
                                               imsize,
                                               imsize)))
            w.add_graph(self.discriminator.model, input_to_model=dummy_input)

    def init_optimizers(self):
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), 
                                            lr=self.learning_rate, betas=(0.0, 0.999))
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),
                                            lr=self.learning_rate, betas=(0.0, 0.999))

    def log_variable(self, name, value):
        self.writer.add_scalar(name, value, global_step=self.global_step)
    
    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            for i, (real_data, labels) in tqdm.tqdm(enumerate(self.data_loader), 
                                            total=len(self.data_loader),
                                            desc="Global step: {:10.0f}".format(self.global_step)):
                batch_start_time = time.time()
                self.generator.train()
                labels = to_cuda(Variable(labels))
                self.global_step += 1 * self.batch_size
                if self.is_transitioning:
                    self.transition_variable = ((self.global_step-1) % self.transition_iters) / self.transition_iters
                real_data = preprocess_images(real_data, self.transition_variable)
                z = self.generate_noise(real_data.shape[0], labels)

                # Forward G
                fake_data = self.generator(z, self.transition_variable)

                # Train Discriminator
                real_scores, real_logits = self.discriminator(real_data, self.transition_variable)
                fake_scores, fake_logits = self.discriminator(fake_data.detach(), self.transition_variable)

                wasserstein_distance = (real_scores - fake_scores).squeeze()  # Wasserstein-1 Distance
                gradient_pen = gradient_penalty(real_data.data, fake_data.data, self.discriminator, self.transition_variable)
                # Epsilon penalty
                epsilon_penalty = (real_scores ** 2).squeeze()

                # Label loss penalty
                label_penalty_discriminator = to_cuda(torch.Tensor([0]))
                if self.label_size > 0:
                    label_penalty_reals = self.label_criterion(real_logits, labels)
                    label_penalty_fakes = self.label_criterion(fake_logits, labels)
                    label_penalty_discriminator = (label_penalty_reals + label_penalty_fakes).squeeze()
                assert wasserstein_distance.shape == gradient_pen.shape
                assert wasserstein_distance.shape == epsilon_penalty.shape
                D_loss = -wasserstein_distance + gradient_pen * 10 + epsilon_penalty * 0.001 + label_penalty_discriminator


                D_loss = D_loss.mean()
                self.d_optimizer.zero_grad()
                D_loss.backward()
                self.d_optimizer.step()


                # Forward G
                fake_scores, fake_logits = self.discriminator(fake_data, self.transition_variable)
                
                # Label loss penalty
                label_penalty_generator = self.label_criterion(fake_logits, labels).squeeze() if self.label_size > 0 else to_cuda(torch.Tensor([0]))
        
                
                G_loss = (-fake_scores + label_penalty_generator).mean()
                #G_loss = (-fake_scores).mean()
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                G_loss.backward()
                self.g_optimizer.step()


                nsec_per_img = (time.time() - batch_start_time) / self.batch_size
                self.total_time = (time.time() - self.start_time) / 60
                # Log data
                self.log_variable('discriminator/wasserstein-distance', wasserstein_distance.mean().item())
                self.log_variable('discriminator/gradient-penalty', gradient_pen.mean().item())
                self.log_variable("discriminator/real-score", real_scores.mean().item())
                self.log_variable("discriminator/fake-score", fake_scores.mean().item())
                self.log_variable("discriminator/epsilon-penalty", epsilon_penalty.mean().item())
                self.log_variable("stats/transition-value", self.transition_variable)
                self.log_variable("stats/nsec_per_img", nsec_per_img)
                self.log_variable("stats/training_time_minutes", self.total_time)
                self.log_variable("discriminator/label-penalty", label_penalty_discriminator.mean())
                self.log_variable("generator/label-penalty", label_penalty_generator.mean())


                if (self.global_step) % (self.batch_size*100) == 0:
                    self.generator.eval()
                    fake_data_sample = normalize_img(self.generator(self.z_sample, self.transition_variable).data)
                    save_images(self.writer, fake_data_sample, self.global_step, self.generated_data_dir)

                    # Save input images
                    imsize = real_data.shape[2]
                    filename = "reals{0}_{1}x{1}.jpg".format(self.global_step, imsize)
                    filepath = os.path.join(self.generated_data_dir, filename)
                    to_save = normalize_img(real_data[:100])
                    torchvision.utils.save_image(to_save, filepath, nrow=10)
                if self.global_step % (self.batch_size * 1000) == 0:
                    self.save_checkpoint(epoch)
                if self.global_step % self.transition_iters == 0:

                    if self.global_step % (self.transition_iters*2) == 0:
                        # Stop transitioning
                        self.is_transitioning = False
                        self.transition_variable = 1.0
                        self.save_checkpoint(epoch)
                    elif self.current_imsize < self.max_imsize:
                        current_channels = self.transition_channels[self.transition_step]
                        self.discriminator.extend(current_channels)
                        self.generator.extend(current_channels)
                        self.current_imsize *= 2

                        
                        self.discriminator.summary(), self.generator.summary()
                        self.data_loader = load_dataset(self.dataset, self.batch_size, self.current_imsize)
                        self.is_transitioning = True

                        self.init_optimizers()
                        self.transition_variable = 0
                        
                        fake_data_sample = normalize_img(self.generator(self.z_sample, self.transition_variable).data)
                        os.makedirs("lol", exist_ok=True)
                        filepath = os.path.join("lol", "test.jpg")
                        torchvision.utils.save_image(fake_data_sample[:100], filepath, nrow=10)
                        self.log_model_graphs()
                        self.transition_step += 1
                        self.save_checkpoint(epoch)

                        break
            self.save_checkpoint(epoch)





if __name__ == '__main__':
    options = load_options()

    trainer = Trainer(options)
    trainer.train()