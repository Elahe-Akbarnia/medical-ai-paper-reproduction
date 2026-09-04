# Projected Gradient Descent attack.

import torch



class PGDAttack:


    def __init__(
            self,
            epsilon,
            alpha=0.01,
            steps=10
    ):

        self.epsilon=epsilon

        self.alpha=alpha

        self.steps=steps



    def generate(
            self,
            model,
            images,
            labels
    ):


        original = images.clone().detach()



        adversarial = original.clone()



        for _ in range(self.steps):


            adversarial.requires_grad=True



            outputs=model(
                adversarial
            )


            loss=torch.nn.functional.binary_cross_entropy_with_logits(

                outputs["logits"].squeeze(),

                labels

            )


            model.zero_grad()


            loss.backward()



            gradient=adversarial.grad.sign()



            adversarial = (

                adversarial

                +

                self.alpha *
                gradient

            )



            perturbation = (

                adversarial

                -

                original

            )


            perturbation=torch.clamp(

                perturbation,

                -self.epsilon,

                self.epsilon

            )


            adversarial=(

                original

                +

                perturbation

            )


            adversarial=torch.clamp(

                adversarial,

                0,

                1

            ).detach()



        return adversarial
