# Indiana University multimodal dataset builder.
# Creates samples: frontal_image, lateral_image, text_caption and label


from pathlib import Path

import pandas as pd



class IndianaDatasetBuilder:


    def __init__(
            self,
            reports_csv,
            projections_csv,
            image_root
    ):

        self.reports_csv = reports_csv

        self.projections_csv = projections_csv

        self.image_root = Path(image_root)



    def load_reports(self):

        return pd.read_csv(
            self.reports_csv
        )



    def load_projections(self):

        return pd.read_csv(
            self.projections_csv
        )



    def build_dataframe(self):


        reports = self.load_reports()

        projections = self.load_projections()



        merged = projections.merge(
            reports,
            on="uid",
            how="inner"
        )


        return self.create_samples(
            merged
        )



    def create_samples(
            self,
            dataframe
    ):


        samples=[]


        grouped=dataframe.groupby(
            "uid"
        )


        for uid,group in grouped:


            frontal=None
            lateral=None


            for _,row in group.iterrows():

                filename=row["filename"]

                projection=row["projection"]


                if projection=="frontal":

                    frontal=filename


                elif projection=="lateral":

                    lateral=filename



            if frontal and lateral:


                row=group.iloc[0]


                text=self.create_caption(
                    row
                )


                label=row["Label"]


                samples.append(

                    {

                    "uid":uid,

                    "frontal_image":
                        str(
                        self.image_root/frontal
                        ),

                    "lateral_image":
                        str(
                        self.image_root/lateral
                        ),

                    "text_caption":
                        text,

                    "label":
                        int(label)

                    }

                )


        return pd.DataFrame(samples)



    def create_caption(
            self,
            row
    ):


        fields=[

            row.get(
                "impression",
                ""
            ),

            row.get(
                "findings",
                ""
            ),

            row.get(
                "indication",
                ""
            )

        ]


        return " ".join(

            [

                str(x)

                for x in fields

                if pd.notna(x)

            ]

        )
