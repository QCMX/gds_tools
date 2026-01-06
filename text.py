# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
import gdspy

# Note: list available font names with
#     sorted(matplotlib.font_manager.get_font_names())

def any_text(text, width=100, **polyprops):
    """Render text and scale to given width in um.

    Returns a PolygonSet
    """
    fp = FontProperties(fname="./gds_tools/Shrikhand-Regular.ttf")
    poly = gdspy.PolygonSet(render_text(text, 10, font_prop=fp), **polyprops)
    bbox = poly.get_bounding_box()
    w, h = bbox[1] - bbox[0]
    return poly.scale(width/w)


def qcmx_logo(width=100, **polyprops):
    """Render QCMX logo and scale to given width in um.

    Returns a PolygonSet
    """
    fp = FontProperties(fname="./gds_tools/Shrikhand-Regular.ttf")
    poly = gdspy.PolygonSet(render_text("QCMX", 10, font_prop=fp), **polyprops)
    bbox = poly.get_bounding_box()
    w, h = bbox[1] - bbox[0]
    return poly.scale(width/w)


def diffraction_box(rectangle, linewidth1=2, linewidth2=2, **polyprops):
    """Make a horizontal lines inside given rectangle.
    Small linewidths produce a diffraction grating.

    Returns a PolygonSet
    """
    if isinstance(rectangle, gdspy.PolygonSet):
        bbox = rectangle.get_bounding_box()
    else:
        bbox = rectangle
    x0, x1 = bbox[0,0], bbox[1,0]
    points = [
        [[x0, y], [x1, y], [x1, y+linewidth1], [x0, y+linewidth1], [x0, y]]
        for y in np.arange(bbox[0,1], bbox[1,1], linewidth1+linewidth2)]
    return gdspy.PolygonSet(points, **polyprops)


# From the gdspy library examples
# Copyright 2009 Copyright 2009 Lucas Heitzmann Gabrielli.
# Boost Software License - Version 1.0
def render_text(text, size=None, position=(0, 0), font_prop=None, tolerance=0.1):
    path = TextPath(position, text, size=size, prop=font_prop)
    polys = []
    xmax = position[0]
    for points, code in path.iter_segments():
        if code == path.MOVETO:
            c = gdspy.Curve(*points, tolerance=tolerance)
        elif code == path.LINETO:
            c.L(*points)
        elif code == path.CURVE3:
            c.Q(*points)
        elif code == path.CURVE4:
            c.C(*points)
        elif code == path.CLOSEPOLY:
            poly = c.get_points()
            if poly.size > 0:
                if poly[:, 0].min() < xmax:
                    i = len(polys) - 1
                    while i >= 0:
                        if gdspy.inside(
                            poly[:1], [polys[i]], precision=0.1 * tolerance
                        )[0]:
                            p = polys.pop(i)
                            poly = gdspy.boolean(
                                [p],
                                [poly],
                                "xor",
                                precision=0.1 * tolerance,
                                max_points=0,
                            ).polygons[0]
                            break
                        elif gdspy.inside(
                            polys[i][:1], [poly], precision=0.1 * tolerance
                        )[0]:
                            p = polys.pop(i)
                            poly = gdspy.boolean(
                                [p],
                                [poly],
                                "xor",
                                precision=0.1 * tolerance,
                                max_points=0,
                            ).polygons[0]
                        i -= 1
                xmax = max(xmax, poly[:, 0].max())
                polys.append(poly)
    return polys
