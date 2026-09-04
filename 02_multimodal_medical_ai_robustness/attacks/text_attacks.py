# Text adversarial attacks.
# Implemented: synonym replacement and random word deletion


import random

from nltk.corpus import wordnet



class TextAttack:


    def synonym_replace(
            self,
            text,
            ratio=0.2
    ):


        words=text.split()


        number=int(
            len(words)*ratio
        )


        indexes=random.sample(

            range(len(words)),

            min(
                number,
                len(words)
            )

        )


        for idx in indexes:


            synonyms = wordnet.synsets(
                words[idx]
            )


            if synonyms:


                replacement=(

                    synonyms[0]
                    .lemmas()[0]
                    .name()

                )


                words[idx]=replacement



        return " ".join(words)



    def half_sentence_delete(
            self,
            text
    ):


        words=text.split()



        keep=int(
            len(words)*0.5
        )


        return " ".join(
            words[:keep]
        )
