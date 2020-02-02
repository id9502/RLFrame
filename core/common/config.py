import os
import bz2
import ast
import pickle
import argparse
import numpy as np
from core.common import List, Tuple, Union, Dict
from core.common.types import StringList


__all__ = ["Config", "ARGConfig", "ARG", "ParamDict"]


class ParamDict(object):

    def __init__(self, other: Union[Dict, Tuple, List] = None):
        self._items = {}
        if other is None:
            pass
        elif isinstance(other, Dict):
            self._items = other
        elif isinstance(other, ParamDict):
            self._items = other._items
        elif isinstance(other, (Tuple, List)):
            self._items.update(other)
        else:
            raise TypeError(f"Error, cannot convert {type(other)} to ParamDict")

    def require(self, *names: str) -> Union[List, object]:
        ret = []
        for s in names:
            if s not in self._items:
                raise KeyError(f"Key '{s}' is required but not existed in dictionary")
            ret.append(self._items[s])
        if len(names) == 1:
            return ret[0]
        else:
            return ret

    # # ------- override dict methods --------------- # #

    def __setitem__(self, key: str, item: object):
        self._items[key] = item

    def __getitem__(self, key: str):
        return self._items[key]

    def __repr__(self):
        return repr(self._items)

    def __len__(self):
        return len(self._items)

    def __delitem__(self, key: str):
        del self._items[key]

    def __eq__(self, other):
        if not isinstance(other, ParamDict):
            return False
        return self._items == other._items

    def __ne__(self, other):
        return not self == other

    def __iter__(self):
        return iter(self._items)

    def __contains__(self, key):
        return key in self._items

    def __add__(self, other):
        return self.copy().update(other)

    def __iadd__(self, other):
        self.update(other)

    def __radd__(self, other):
        ret = self.copy().update(other)
        if isinstance(other, Dict):
            return ret._items
        elif isinstance(other, ParamDict):
            return ret
        else:
            raise TypeError(f"Error, operator + between {type(other)} and ParamDict has not been defined")

    def update(self, other):
        if isinstance(other, (Dict, List, Tuple)):
            self._items.update(other)
        elif isinstance(other, ParamDict):
            self._items.update(other._items)
        else:
            raise TypeError(f"Error, cannot update from {type(other)}")
        return self

    def clear(self):
        self._items.clear()

    def copy(self):
        return ParamDict(self._items.copy())

    def has_key(self, key: str):
        return key in self._items

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def items(self):
        return self._items.items()

    def pop(self, key):
        _item = self._items[key]
        del self._items[key]
        return _item


class Config(ParamDict):

    def __init__(self, item_list: Union[List, Tuple, Dict, ParamDict] = None):
        super(Config, self).__init__()
        self._fields = {}

        if isinstance(item_list, Dict):
            self._items = item_list
        else:
            self.overwrite(item_list)

    def require_field(self, *names: str) -> ParamDict:
        ret = ParamDict()
        for s in names:
            has_key = False
            for key in self._fields:
                if s in self._fields[key]:
                    ret[s] = self._items[s]
                    has_key = True
            if not has_key:
                raise KeyError(f"Field {s} is required but not existed in Config")
        return ret

    def register_item(self, key: str, value: object = None, fields: StringList = None):
        """
        register item into config, if item exist, will do incremental update
        """
        if value is not None or key not in self._items:
            self._items[key] = value
        if key not in self._fields:
            self._fields[key] = []
        if fields is not None:
            self._fields[key] = list(set(fields).union(self._fields[key]))

    def unregister_item(self, key: str, fields: StringList = None) -> StringList:
        """
        unregister item fields from config, return fields item had been removed from
        :return: fields item had been removed from
        """
        ret = []
        if key not in self._items:
            return ret
        if fields is None:
            fields = self._fields[key]
        for f in fields:
            if f in self._fields[key]:
                self._fields[key].remove(f)
                ret.append(f)
        return ret

    def field_cmp(self, config, mode: str = "this") -> bool:
        """
        :param config: other config
        :param mode: "inter", "union", "this", "other", "none"
        :return: bool
        """
        this_field = [key for key in self._fields if "critical" in self._fields[key]]
        other_field = [key for key in config._fields if "critical" in config._fields[key]]
        cmp_keys = []
        if mode == "this":
            cmp_keys = this_field
        if mode == "other":
            cmp_keys = other_field
        elif mode == "inter":
            cmp_keys = list(set(this_field).intersection(other_field))
        elif mode == "union":
            cmp_keys = list(set(this_field).union(other_field))
        elif mode == "none":
            return True

        for key in cmp_keys:
            if key not in self._items or key not in config._items:
                return False
            if np.any(self[key] != config[key]):
                return False
        return True

    def get_field_item(self, field: str) -> ParamDict:
        ret = ParamDict()
        for key in self._fields:
            if field in self._fields[key]:
                ret[key] = self[key]
        return ret

    def get_field(self, key: str) -> StringList:
        return self._fields[key]

    def in_field(self, key: str, field: str):
        return field in self._fields[key]

    def save(self, file_name: str) -> bool:
        with bz2.BZ2File(file_name, "wb") as f:
            cfg_save = self.to_list(["save"])
            pickle.dump(cfg_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True

    def load(self, file_name: str, check_mode="none") -> bool:
        if not os.path.isfile(file_name):
            return False
        with bz2.BZ2File(file_name, "rb") as f:
            cfg_save = Config(pickle.load(f))
            if self.field_cmp(cfg_save, check_mode):
                self.update(cfg_save)
                return True
            else:
                return False

    def to_list(self, fields: StringList):
        ret = []
        fields = set(fields)
        for key in self._fields:
            if not fields.isdisjoint(self._fields[key]):
                ret.append([key, self._items[key], self._fields[key]])
        return ret

    # # ------- override ParamDict methods --------------- # #

    def __setitem__(self, key: str, item: object):
        super(Config, self).__setitem__(key, item)
        if key not in self._fields:
            self._fields[key] = []

    def __delitem__(self, key: str):
        super(Config, self).__delitem__(key)
        del self._fields[key]

    def __eq__(self, other):
        DeprecationWarning("Comparing between Configs with '==' is deprecated, use 'field_cmp' instead")
        return self.field_cmp(other, "union")

    def __add__(self, other):
        return self.copy().update(other)

    def __iadd__(self, other):
        self.update(other)

    def __radd__(self, other):
        ret = self.copy().update(other)
        if isinstance(other, Config):
            return ret
        elif isinstance(other, Dict):
            return ret._items
        elif isinstance(other, ParamDict):
            return ParamDict(ret._items)
        elif isinstance(other, (Tuple, List)):
            return [[k, self._items[k], self._fields[k]] for k in self._items]
        else:
            raise TypeError(f"Error, operator + between {type(other)} and ParamDict has not been defined")

    def clear(self):
        super(Config, self).clear()
        self._fields.clear()

    def copy(self):
        _copy = Config()
        _copy._fields = self._fields.copy()
        _copy._items = self._items.copy()
        return _copy

    def update(self, config):
        if config is None:
            return self
        if isinstance(config, Config):
            for key in config:
                self._items[key] = config._items[key]
                if key not in self._fields:
                    self._fields[key] = config._fields[key]
                else:
                    self._fields[key] = list(set(self._fields[key]).union(config._fields[key]))
        elif isinstance(config, (Dict, ParamDict)):
            for key in config:
                self._items[key] = config[key]
                if key not in self._fields:
                    self._fields[key] = []
        elif isinstance(config, (List, Tuple)):
            for item in config:
                if len(item) < 2:
                    raise ValueError(f"Error: len() should >= 2, but get {len(item)}")
                self._items[item[0]] = item[1]
                self._fields[item[0]] = list(set(self._fields[item[0]]).union(item[2])) if len(item) > 2 else []
        else:
            raise TypeError(f"input type should be Config or List, but get {type(config)}, do nothing")
        return self

    def overwrite(self, config):
        if config is None:
            return self
        if isinstance(config, Config):
            self._fields.update(config._fields)
            self._items.update(config._items)
        elif isinstance(config, (Dict, ParamDict)):
            for key in config:
                self._items[key] = config[key]
                if key not in self._fields:
                    self._fields[key] = []
        elif isinstance(config, (List, Tuple)):
            for item in config:
                if len(item) < 2:
                    raise ValueError(f"Error: len() should >= 2, but get {len(item)}")
                self._items[item[0]] = item[1]
                self._fields[item[0]] = item[2] if len(item) > 2 else []
        else:
            raise TypeError(f"Error: Input type should be Config or List, but get {type(config)}, do nothing")
        return self

    def pop(self, key):
        _item = self._items[key]
        del self[key]
        return _item


def ARG(name: str, value: object, fields: StringList = None, desc: str = "",
        critical=False, save=True, arg=True, key_name: str = None):
    if fields is None:
        fields = []
    if arg and "arg" not in fields:
        fields.append("arg")
    if critical and "critical" not in fields:
        fields.append("critical")
    if save and "save" not in fields:
        fields.append("save")
    if not isinstance(value, (str, tuple, list, int, float, bool)):
        raise TypeError(f"value type should be one in (str, tuple, list, int, float, bool), but get {type(value)}")
    return name, value, fields, desc.format(value), key_name


class ARGConfig(Config):

    def __init__(self, name: str, *ARG_list):
        super(ARGConfig, self).__init__(ARG_list)
        self._arg_list = list(ARG_list)
        self._name = name

    def parser(self):
        # compiling arg-parser
        parser = argparse.ArgumentParser(description=self._name)
        for arg in self._arg_list:
            if "arg" not in arg[2]:
                continue
            arg_name = arg[0].replace(' ', '_').replace('-', '_')
            parser.add_argument(f"--{arg_name}", type=str, default=str(arg[1]), help=arg[3])

        pared_args = parser.parse_args().__dict__

        for arg in self._arg_list:
            arg_name = arg[0].replace(' ', '_').replace('-', '_')
            if "arg" not in arg[2] or arg_name not in pared_args:
                continue
            self.register_item(key=arg[0] if arg[4] is None else arg[4],
                               value=self.from_string(pared_args[arg_name], type(arg[1])),
                               fields=arg[2])

    def copy(self):
        _copy = ARGConfig(self._name, None)
        _copy._arg_list = self._arg_list.copy()
        _copy._items = self._items.copy()
        _copy._fields = self._fields.copy()
        return _copy

    def update(self, config):
        super(ARGConfig, self).update(config)
        if isinstance(config, ARGConfig):
            value_idx = {}
            for i, item in enumerate(self._arg_list):
                value_idx[item[0]] = i
            for item in config._arg_list:
                if item[0] in value_idx:
                    self._arg_list[value_idx[item]][1] = item[1]
                    self._arg_list[value_idx[item]][2].extend(set(item[2]).difference_update(self._arg_list[value_idx[item]][2]))
                    self._arg_list[value_idx[item]][3] = item[3]
                    self._arg_list[value_idx[item]][4] = item[4]
                else:
                    self._arg_list.append(item)
                    value_idx[item[0]] = len(self._arg_list) - 1

    def overwrite(self, config):
        super(ARGConfig, self).overwrite(config)
        if isinstance(config, ARGConfig):
            value_idx = {}
            for i, item in enumerate(self._arg_list):
                value_idx[item[0]] = i
            for item in config._arg_list:
                if item[0] in value_idx:
                    self._arg_list[value_idx[item]] = item
                else:
                    self._arg_list.append(item)
                    value_idx[item[0]] = len(self._arg_list) - 1

    @staticmethod
    def from_string(string: str, typeinst: type):
        if typeinst == str:
            return string
        elif typeinst == int:
            return int(string)
        elif typeinst == float:
            return float(string)
        elif typeinst == bool:
            return string.lower() == "true"
        elif typeinst == tuple or typeinst == list:
            return typeinst(ast.literal_eval(string))
        else:
            raise TypeError(f"unknown type (str, tuple, list, int, float, bool), but get {typeinst}")


if __name__ == "__main__":
    c = ARGConfig(
        "Test argument config class",
        ARG(name="name", value="1", fields=["critical", "save"]),
        ARG(name="path", value="2", fields=["save"], arg=False),
        ARG(name="tau", value=1., fields=["save"], desc="Tau value, default={}"),
        ["test", False, ["critical"]]
    )
    c.parser()
    print(c.get_field("tau"))
    print(c)
