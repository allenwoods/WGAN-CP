from src.nn import DCGAN_G, DCGAN_D

nc = 1
nz = 100
ngf = 64
ndf = 64
n_extra_layers = 0
λ = 10

img_size = 28
batch_size = 64
lr_d = 1e-4
lr_g = 1e-4

netG = DCGAN_G(img_size, nz, nc, ngf)
netD = DCGAN_D(img_size, nc, ndf, wgan=False)

print(netG.summary())
print(netD.summary())