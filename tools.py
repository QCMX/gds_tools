# -*- coding: utf-8 -*-
import numpy as np
from typing import Iterable, Callable
from gdspy import Rectangle, PolygonSet


def remove_polygons_recursive(
        cell,
        layer: int | Iterable | None = None,
        test: Callable[[np.array, int, int], bool] | None = None):
    if layer is not None:
        assert test is None, "Supply either layer or test, not both"
        if hasattr(layer, '__iter__'): layers = list(layer)
        else: layers = [layer]
        test = lambda points, layer, datatype: layer in layers
    elif test is not None:
        assert layer is None, "Supply either layer or test, not both"
    else:
        raise ValueError("Supply either layer or test argument.")

    # this is a set, so no double-counting of cells referenced multiple times
    cells = cell.get_dependencies(recursive=True)
    cells.add(cell)
    for cell in cells:
        cell.remove_polygons(test)

    return cell


def primitives_to_polygonset(objs: Iterable):
    polygons = []
    layers = []
    for obj in objs:
        if hasattr(obj, 'to_polygonset'):
            pset = obj.to_polygonset()
            polygons.extend(pset.polygons)
            layers.extend(pset.layers)
        elif hasattr(obj, 'to_polygonsets'):
            for pset in obj.to_polygonsets():
                polygons.extend(pset.polygons)
                layers.extend(pset.layers)
        elif isinstance(obj, PolygonSet):
            polygons.extend(obj.polygons)
            layers.extend(obj.layers)
        elif isinstance(obj, Rectangle):
            (x1, y1), (x2, y2) = obj.point1, obj.point2
            polygons.append(np.array([
                (x1, y1), (x2, y1), (x2, y2), (x1, y2)]))
            layers.append(obj.layer)
        elif isinstance(obj, list):
            pset = primitives_to_polygonset(obj)
            polygons.extend(pset.polygons)
            layers.extend(pset.layers)
        else:
            raise ValueError(f"Objects {repr(obj)} not supported yet.")
    assert len(polygons) == len(layers)
    pset = PolygonSet(polygons)
    pset.layers = layers # hack
    return pset
