# FGSM adversarial attack.

# Image based attack for medical X-ray models.

import torch



class FGSMAttack:


    def __init__(
            self,
            epsilon
    ):

        self.epsilon = epsilon



    def generate(
            self,
            model,
            images,
            labels
    ):


        images = images.clone().detach()

        images.requires_grad = True



        outputs = model(images)


        logits = outputs["logits"]



        loss = torch.nn.functional.binary_cross_entropy_with_logits(

            logits.squeeze(),

            labels

        )



        model.zero_grad()


        loss.backward()



        gradient = images.grad.data



        perturbed_images = (

            images

            +

            self.epsilon *
            gradient.sign()

        )


        perturbed_images = torch.clamp(

            perturbed_images,

            0,

            1

        )


        return perturbed_images.detach()
