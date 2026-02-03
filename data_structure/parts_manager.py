#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全局部件管理对象与部件数据对象。
用于将 cas_parts_info 转换为统一的 PartData 管理结构。
"""

from copy import deepcopy


_DEFAULT_ELEMENTS = {"vertices": [], "edges": [], "faces": [], "bodies": []}


def _normalize_elements(elements):
    if not isinstance(elements, dict):
        return deepcopy(_DEFAULT_ELEMENTS)
    normalized = {}
    for key in _DEFAULT_ELEMENTS.keys():
        value = elements.get(key, [])
        normalized[key] = list(value) if isinstance(value, (list, tuple)) else []
    return normalized


class PartData(dict):
    """单个部件对象，包含几何元素、网格元素与部件参数。"""

    def __init__(self, part_name, geometry_elements=None, mesh_elements=None, part_params=None, **kwargs):
        super().__init__()
        self["part_name"] = part_name
        self["geometry_elements"] = _normalize_elements(geometry_elements)
        self["mesh_elements"] = _normalize_elements(mesh_elements)
        if part_params is not None:
            self["part_params"] = part_params
        for key, value in kwargs.items():
            if key not in self:
                self[key] = value

    @classmethod
    def from_dict(cls, part_name, data):
        if isinstance(data, PartData):
            data.setdefault("part_name", part_name)
            return data
        if not isinstance(data, dict):
            return cls(part_name)
        geometry_elements = data.get("geometry_elements")
        mesh_elements = data.get("mesh_elements")
        part_params = data.get("part_params")
        extra = {k: v for k, v in data.items() if k not in ("geometry_elements", "mesh_elements", "part_params")}
        extra.pop("part_name", None)
        return cls(part_name, geometry_elements=geometry_elements, mesh_elements=mesh_elements, part_params=part_params, **extra)

    def ensure_defaults(self):
        if "geometry_elements" not in self:
            self["geometry_elements"] = deepcopy(_DEFAULT_ELEMENTS)
        else:
            self["geometry_elements"] = _normalize_elements(self["geometry_elements"])
        if "mesh_elements" not in self:
            self["mesh_elements"] = deepcopy(_DEFAULT_ELEMENTS)
        else:
            self["mesh_elements"] = _normalize_elements(self["mesh_elements"])

    def set_part_params(self, part_params):
        self["part_params"] = part_params

    def to_dict(self):
        return deepcopy(dict(self))


class GlobalPartsManager(dict):
    """全局部件管理对象，统一管理所有 PartData。"""

    def __init__(self, parts_info=None, parts_params=None):
        super().__init__()
        if parts_info:
            self.update_from_cas_parts_info(parts_info)
        if parts_params:
            self.apply_part_params(parts_params)

    @classmethod
    def from_cas_parts_info(cls, cas_parts_info, parts_params=None):
        manager = cls()
        if cas_parts_info:
            manager.update_from_cas_parts_info(cas_parts_info)
        if parts_params:
            manager.apply_part_params(parts_params)
        return manager

    def __setitem__(self, key, value):
        part_data = PartData.from_dict(key, value)
        part_data.ensure_defaults()
        super().__setitem__(key, part_data)

    def update_from_cas_parts_info(self, cas_parts_info):
        if isinstance(cas_parts_info, GlobalPartsManager):
            for part_name, part_data in cas_parts_info.items():
                self[part_name] = part_data
            return
        if isinstance(cas_parts_info, dict) and "parts_info" in cas_parts_info:
            cas_parts_info = cas_parts_info.get("parts_info") or {}
        if isinstance(cas_parts_info, list):
            for idx, part_data in enumerate(cas_parts_info):
                if isinstance(part_data, dict):
                    part_name = part_data.get("part_name", f"部件_{idx}")
                else:
                    part_name = f"部件_{idx}"
                self[part_name] = part_data
            return
        if isinstance(cas_parts_info, dict):
            for part_name, part_data in cas_parts_info.items():
                self[part_name] = part_data

    def apply_part_params(self, parts_params):
        if not parts_params:
            return
        from data_structure.parameters import MeshParameters
        for param in parts_params:
            part_name = None
            if isinstance(param, MeshParameters):
                part_name = param.part_name
                part_param = param
            elif hasattr(param, "part_name"):
                part_name = param.part_name
                part_param = param
            elif isinstance(param, dict):
                part_name = param.get("part_name")
                if part_name:
                    part_param = MeshParameters(
                        part_name=part_name,
                        max_size=param.get("max_size", 1e6),
                        PRISM_SWITCH=param.get("PRISM_SWITCH", "off"),
                        first_height=param.get("first_height", 0.01),
                        growth_rate=param.get("growth_rate", 1.2),
                        growth_method=param.get("growth_method", "geometric"),
                        max_layers=param.get("max_layers", 5),
                        full_layers=param.get("full_layers", 5),
                        multi_direction=param.get("multi_direction", False),
                    )
                else:
                    part_param = None
            else:
                part_param = None
            if not part_name or part_param is None:
                continue
            if part_name not in self:
                self[part_name] = PartData(part_name, part_params=part_param)
            else:
                self[part_name].set_part_params(part_param)

    def to_cas_parts_info(self):
        return {part_name: part_data.to_dict() for part_name, part_data in self.items()}

    @property
    def parts_info(self):
        return self
