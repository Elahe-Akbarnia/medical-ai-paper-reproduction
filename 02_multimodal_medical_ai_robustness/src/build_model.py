
from models.image_model import (
    SEResNet152Classifier,
    DualViewImageModel
)


from models.text_model import (
    BioClinicalBERTClassifier
)


from models.early_fusion import (
    VisionBERTEarlyFusion
)


from models.late_fusion import (
    VisionBERTLateFusion
)


from models.ensemble_fusion import (
    VisionBERTEnsembleFusion
)



def build_model(name):


    if name == "image":

        return DualViewImageModel()



    elif name == "text":

        return BioClinicalBERTClassifier()



    elif name == "early_fusion":

        image_model = DualViewImageModel()

        text_model = BioClinicalBERTClassifier()


        return VisionBERTEarlyFusion(

            image_model,

            text_model

        )



    elif name == "late_fusion":

        image_model = DualViewImageModel()

        text_model = BioClinicalBERTClassifier()


        return VisionBERTLateFusion(

            image_model,

            text_model

        )



    elif name == "ensemble":

        return VisionBERTEnsembleFusion(

            DualViewImageModel(),

            BioClinicalBERTClassifier()

        )


    else:

        raise ValueError(
            f"Unknown model {name}"
        )
