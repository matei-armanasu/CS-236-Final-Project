import torch
import torch.nn as nn

# TODO: test that this is accurate and gives us the loss that we expect. Want cross-entropy between the
#       target adversarial distribution and the discriminator output, as well as a MSE between the true 
#       image and the generated image.
def old_generator_loss(images, generated_images, discr_output, y, beta):
    sm_output = nn.Softmax(-1)(discr_output)

    cce_loss = torch.mean(torch.sum(-torch.mul(torch.log(sm_output),y), dim=1))

    mse_loss = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(images, generated_images), dim=(1, 2, 3)))
    return cce_loss + beta * mse_loss

def generator_loss(images, generated_images, discr_output, y, beta):
    cce_loss = torch.mean(torch.sum(-torch.mul(discr_output,y), dim=1)) # maximize the pre-softmax score of the target class

    mse_loss = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(images, generated_images), dim=(1, 2, 3)))
    return cce_loss + beta * mse_loss
