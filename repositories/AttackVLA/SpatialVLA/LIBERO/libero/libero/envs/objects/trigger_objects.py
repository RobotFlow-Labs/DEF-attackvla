import os
import re
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string

import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from libero.libero.envs.base_object import register_object


class TriggerObject(MujocoXMLObject):
    def __init__(self, name, obj_name):
        super().__init__(
            os.path.join(
                str(absolute_path),
                f"assets/trigger_objects/{obj_name}/{obj_name}.xml",
            ),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = "x"

        self.object_properties = {"vis_site_names": {}}


@register_object
class Mickey(TriggerObject):
    def __init__(self, name="mickey", obj_name="mickey"):
        super().__init__(name, obj_name)
        self.rotation = {
            "x": (np.pi / 2, np.pi / 2),
            "z": (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None