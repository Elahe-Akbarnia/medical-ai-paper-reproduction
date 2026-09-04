import yaml


class Config:


    def __init__(
            self,
            path
    ):

        with open(path,"r") as file:

            self.config=yaml.safe_load(
                file
            )



    def get(
            self,
            key,
            default=None
    ):

        keys=key.split(".")

        value=self.config


        for k in keys:

            if k not in value:

                return default


            value=value[k]


        return value
