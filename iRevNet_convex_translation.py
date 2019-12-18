from utils import renormalize
import matplotlib.pyplot as plt
import numpy as np
import torch

def iRevNet_convex_translation(model, loader, nz=50):
    model.eval()

    is_cuda = next(model.parameters()).is_cuda
    fig, axs = plt.subplots(4,11, figsize=(25,5))

    input1 = loader.dataset[np.random.randint(0, len(loader.dataset))][0][None]
    input2 = loader.dataset[np.random.randint(0, len(loader.dataset))][0][None]

    if is_cuda:
        input1, input2 = input1.cuda(), input2.cuda()

    output1 = model(input1)
    output2 = model(input2)

    # component-wise
    output1_component = output1.clone()
    output2_component = output2.clone()

    output1_component[:,nz:] = 0
    output2_component[:,nz:] = 0

    queue1 = output1.clone()[:,nz:]
    queue2 = output2.clone()[:,nz:]

    # whole
    output1_whole = output1.clone()
    output2_whole = output2.clone()

    for idx, t in enumerate(np.arange(0, 1.1, 0.1)):
        # component-wise
        z = (1-t) * output1_component + t * output2_component
        inverse_component = model.inverse(z)

        x = inverse_component[0].permute(1,2,0).detach().cpu().numpy()
        x = renormalize(x)

        axs[0,idx].imshow(x[:,:,0], cmap='gray')

        # whole
        z = (1-t) * output1_whole + t * output2_whole
        inverse_whole = model.inverse(z)

        x = inverse_whole[0].permute(1,2,0).detach().cpu().numpy()
        x = renormalize(x)

        axs[1,idx].imshow(x[:,:,0], cmap='gray')

        # Difference
        diff = torch.abs(inverse_component - inverse_whole)

        x = diff[0].permute(1,2,0).detach().cpu().numpy()
        x = renormalize(x)

        axs[2,idx].imshow(x[:,:,0])

        # image translation
        axs[3,idx].imshow(renormalize(((1-t) * input1 + t * input2)[0,0].cpu().numpy()), cmap='gray')

    plt.show()