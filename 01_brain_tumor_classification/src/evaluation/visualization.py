# Visualization utilities.

# Includes:
# - confusion matrix
# - confidence plots
# - robustness comparison



import matplotlib.pyplot as plt

import seaborn as sns



def plot_confusion_matrix(

        matrix,

        classes,

        save_path

):


    plt.figure(

        figsize=(8,6)

    )


    sns.heatmap(

        matrix,

        annot=True,

        fmt="d",

        xticklabels=classes,

        yticklabels=classes

    )


    plt.xlabel(

        "Predicted"

    )


    plt.ylabel(

        "Actual"

    )


    plt.title(

        "Confusion Matrix"

    )


    plt.tight_layout()



    plt.savefig(

        save_path,

        dpi=300

    )


    plt.close()





def plot_confidence_distribution(

        probabilities,

        labels,

        save_path

):


    """

    Plot model confidence.

    Used for adversarial confidence analysis.

    """



    confidence = probabilities.max(

        axis=1

    )



    plt.figure(

        figsize=(8,5)

    )



    plt.hist(

        confidence,

        bins=20

    )


    plt.xlabel(

        "Prediction Confidence"

    )


    plt.ylabel(

        "Frequency"

    )


    plt.title(

        "Prediction Confidence Distribution"

    )


    plt.savefig(

        save_path,

        dpi=300

    )


    plt.close()
