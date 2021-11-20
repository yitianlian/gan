if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import time
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import utils, datasets, transforms
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    torch.manual_seed(0)

    # Root directory for dataset
    dataroot = "E:/pythonProject/mnist"

    # Number of workers for dataloader
    workers = 1

    # Batch size during training
    batch_size = 32

    # Spatial size of training images. All images will be resized to this size using a transformer.
    image_size = 32

    # Number of channels in the training images. For color images this is 3
    nc = 1

    # Number of classes in the training images. For mnist dataset this is 10
    num_classes = 10

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 10

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    train_data = datasets.MNIST(
        root=dataroot,
        train=True,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        download=True
    )
    test_data = datasets.MNIST(
        root=dataroot,
        train=False,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    dataset = train_data+test_data
    print(f'Total Size of Dataset: {len(dataset)}')

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )

    device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

    imgs = {}
    for x, y in dataset:
        if y not in imgs:
            imgs[y] = []
        elif len(imgs[y]) != 10:
            imgs[y].append(x)
        elif sum(len(imgs[key]) for key in imgs) == 100:
            break
        else:
            continue

    imgs = sorted(imgs.items(), key=lambda x: x[0])
    imgs = [torch.stack(item[1], dim=0) for item in imgs]
    imgs = torch.cat(imgs, dim=0)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.image = nn.Sequential(
                # state size. (nz) x 1 x 1
                nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True)
                # state size. (ngf*4) x 4 x 4
            )
            self.label = nn.Sequential(
                # state size. (num_classes) x 1 x 1
                nn.ConvTranspose2d(num_classes, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True)
                # state size. (ngf*4) x 4 x 4
            )
            self.main = nn.Sequential(
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 32 x 32
            )

        def forward(self, image, label):
            image = self.image(image)
            label = self.label(label)
            incat = torch.cat((image, label), dim=1)
            return self.main(incat)

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if device.type == 'cuda' and ngpu > 1:
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netG.apply(weights_init)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.image = nn.Sequential(
                # input is (nc) x 32 x 32
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                # state size. (ndf) x 16 x 16
            )
            self.label = nn.Sequential(
                # input is (num_classes) x 32 x 32
                nn.Conv2d(num_classes, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                # state size. (ndf) x 16 x 16
            )
            self.main = nn.Sequential(
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                # state size. (1) x 1 x 1
                nn.Sigmoid()
            )

        def forward(self, image, label):
            image = self.image(image)
            label = self.label(label)
            incat = torch.cat((image, label), dim=1)
            return self.main(incat)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if device.type == 'cuda' and ngpu > 1:
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label_num = 1.
    fake_label_num = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Label one-hot for G
    label_1hots = torch.zeros(10,10)
    for i in range(10):
        label_1hots[i,i] = 1
    label_1hots = label_1hots.view(10,10,1,1).to(device)

    # Label one-hot for D
    label_fills = torch.zeros(10, 10, image_size, image_size)
    ones = torch.ones(image_size, image_size)
    for i in range(10):
        label_fills[i][i] = ones
    label_fills = label_fills.to(device)

    # Create batch of latent vectors and laebls that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(100, nz, 1, 1).to(device)
    fixed_label = label_1hots[torch.arange(10).repeat(10).sort().values]

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    D_x_list = []
    D_z_list = []
    loss_tep = 10

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):

        beg_time = time.time()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()

            # Format batch
            real_image = data[0].to(device)
            b_size = real_image.size(0)

            real_label = torch.full((b_size,), real_label_num).to(device)
            fake_label = torch.full((b_size,), fake_label_num).to(device)

            G_label = label_1hots[data[1]]
            D_label = label_fills[data[1]]

            # Forward pass real batch through D
            output = netD(real_image, D_label).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, real_label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1).to(device)
            # Generate fake image batch with G
            fake = netG(noise, G_label)
            # Classify all fake batch with D
            output = netD(fake.detach(), D_label).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, fake_label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, D_label).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, real_label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()


            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Save D(X) and D(G(z)) for plotting later
            D_x_list.append(D_x)
            D_z_list.append(D_G_z2)

            # Save the Best Model
            if errG < loss_tep:
                torch.save(netG.state_dict(), 'model.pt')
                loss_tep = errG

        # Check how the generator is doing by saving G's output on fixed_noise and fixed_label
        with torch.no_grad():
            fake = netG(fixed_noise, fixed_label).detach().cpu()
        img_list.append(utils.make_grid(fake, nrow=10))

        # Next line
        print()
