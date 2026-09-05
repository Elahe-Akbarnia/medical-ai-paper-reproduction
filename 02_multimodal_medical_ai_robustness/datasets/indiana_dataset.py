
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


    def create_samples(self, dataframe):

        samples = []

        grouped = dataframe.groupby("uid")


        for uid, group in grouped:


            frontal = group[
                group["projection"].str.lower() == "frontal"
            ]


            lateral = group[
                group["projection"].str.lower() == "lateral"
            ]


            if len(frontal) > 0 and len(lateral) > 0:


                row = group.iloc[0]


                samples.append(
                    {

                        "uid": uid,


                        "frontal_image":
                            str(
                                self.image_root /
                                frontal.iloc[0]["filename"]
                            ),


                        "lateral_image":
                            str(
                                self.image_root /
                                lateral.iloc[0]["filename"]
                            ),


                        "text_caption":
                            self.create_caption(row),


                        "label":
                            self.create_label(row)

                    }
                )


        return pd.DataFrame(samples)



    def create_caption(self, row):

        fields = [

            row.get("findings", ""),

            row.get("impression", ""),

            row.get("indication", "")

        ]


        return " ".join(

            [

                str(x)

                for x in fields

                if str(x) != "nan"

            ]

        )



    def create_label(self, row):

        problems = str(
            row["Problems"]
        ).lower()


        if problems == "normal":

            return 0


        return 1
