# -*- coding: utf-8 -*-

from typing import Iterable, Callable
import numpy as np
from numpy import pi
import gdspy
from gdspy import Rectangle, Polygon, PolygonSet


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


# Dosn't work: PolygonSet supports only all points on same layer.
# Otherwise results look weird.
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
        else:
            raise ValueError("Objects {repr(obj)} not supported yet.")
    assert len(polygons) == len(layers)
    pset = PolygonSet(polygons)
    pset.layers = layers # hack
    return pset


def cpw(length: float , width: float, ctr: float, avoidance: float,
        layer_gap: int, layer_avoidance: int):
    c = ctr * width
    g = (1 - ctr) * width / 2
    return primitives_to_polygonset([
        Rectangle((0, -c/2-g), (length, -c/2), layer=layer_gap),
        Rectangle((0, +c/2+g), (length, +c/2), layer=layer_gap),
        Rectangle(
            (-avoidance, -c/2-g-avoidance),
            (length+avoidance, +c/2+g+avoidance),
            layer=layer_avoidance),
    ])


def cpw_straight_taper(
        length: float, width1: float, width2: float, ctr: float, avoidance: float,
        layer_gap: int, layer_avoidance: int):
    """Straight taper from x=0 to x=length"""
    # with center,gap from c1,g1 to c2,g2
    c1, c2 = ctr * width1, ctr * width2
    g1, g2 = (1 - ctr) * width1 / 2, (1 - ctr) * width2 / 2
    return primitives_to_polygonset([
        Polygon([
            (length, -c2/2-g2), (length, -c2/2),
            (0, -c1/2), (0, -c1/2-g1)
        ], layer=layer_gap),
        Polygon([
            (0, +c1/2+g1), (0, +c1/2),
            (length, +c2/2), (length, +c2/2+g2)
        ], layer=layer_gap),
        Polygon([
            (0, -c1/2-g1-avoidance),
            (-avoidance, -c1/2-g1-avoidance),
            (-avoidance, +c1/2+g1+avoidance),
            (0, +c1/2+g1+avoidance),
            (length, +c2/2+g2+avoidance),
            (length+avoidance, +c2/2+g2+avoidance),
            (length+avoidance, -c2/2-g2-avoidance),
            (length, -c2/2-g2-avoidance)
        ][::-1], layer=layer_avoidance)
    ])


def _norm(vec):
    return np.sqrt(np.sum(vec[0]**2 + vec[1]**2, axis=0))


def _normal(vec):
    """Orthonormal vector, 90 degrees counterclockwise."""
    ort = vec.copy()
    ort[0], ort[1] = -vec[1], vec[0]
    return ort / _norm(ort)


def _rot(angle):
    """2D rotation matrix in counterclockwise direction."""
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])


SQUARE2um = Rectangle((-1, -1), (1, 1))

class GridPath:
    def __init__(
            self,
            path: gdspy.FlexPath,
            gridoffset: float | Iterable,
            gridspacing: float | Iterable,
            gridobj=SQUARE2um):
        """Like gdspy.FlexPath but also placing objects regularly spaced along
        additional parallel lines.
        
        The supplied path is encapsulated, but not all of the
        functionality is exposed.
        
        Calling segment(), arc() or turn() also advances the encapsulated
        FlexPath.
        
        Additionally this path tracks total center length.
        """
        self.path = path
        self.gridobj = gridobj

        # Otherwise total_len is off
        assert path.points.shape[0] == 1, "Need to start with a FlexPath of only one point."

        if hasattr(gridoffset, '__iter__'):
            self.gridoffset = np.array(gridoffset)
        else:
            self.gridoffset = np.array([gridoffset])
        self.nlines = len(self.gridoffset)

        if hasattr(gridspacing, '__iter__'):
            self.gridspacing = np.array(gridspacing)
        else:
            self.gridspacing = np.array([gridspacing]*self.nlines)
        assert len(self.gridspacing) == self.nlines

        # current position of grid line along path
        self.gridlines_pos = [0]*self.nlines
        # list of all already placed objects
        self.gridlines = []*self.nlines
        
        self.total_len = 0


    def get_angle(self):
        if self.path.points.shape[0] < 2:
            raise ValueError(
                "Cannot define angle on a "
                "path withouth previous segments.")
        v = self.path.points[-1] - self.path.points[-2]
        return np.arctan2(v[1], v[0])


    def segment(self, end_point, width = None, offset = None,
                gridoffset: float | Iterable = None,
                #gridspacing: Iterable = None,
                relative: bool = False):
        begin_point = self.path.points[-1]
        self.path.segment(end_point, width, offset, relative)
        end_point = self.path.points[-1]

        if gridoffset is None:
            gridoffset = self.gridoffset
        else:
            if hasattr(gridoffset, '__iter__'):
                gridoffset = np.array(gridoffset)
            else:
                gridoffset = np.array([gridoffset]*self.nlines)

        vpath = end_point - begin_point
        npath = _normal(vpath)
        for i in range(self.nlines):
            spacing = self.gridspacing[i]
            start = begin_point + npath * self.gridoffset[i]
            end = end_point + npath * gridoffset[i]
            v = end - start
            angle = np.arctan2(v[1], v[0])
            d = _norm(v)
            t = spacing - self.gridlines_pos[i] % spacing
            while t <= d or np.isclose(t, d):
                obj = gdspy.copy(self.gridobj).rotate(angle)
                obj = obj.translate(*(start + v * t/d))
                self.gridlines.append(obj)
                t += spacing
            self.gridlines_pos[i] += d

        self.total_len += _norm(vpath)
        self.gridoffset = gridoffset
        return self

    # equal spacing along distance or in angle
    # spacingmode='angle' to spacingmode='dist'
    def arc(self, radius, initial_angle, final_angle,
            width=None, offset=None,
            #gridoffset: float | Iterable = None,
            #gridspacing=None,
            spacingmode='dist'):
        begin_point = self.path.points[-1]
        self.path.arc(radius, initial_angle, final_angle, width, offset)

        # if gridoffset is None:
        #     gridoffset = self.gridoffset
        # else:
        #     if hasattr(gridoffset, '__iter__'):
        #         gridoffset = np.array(gridoffset)
        #     else:
        #         gridoffset = np.array([gridoffset]*self.nlines)

        totalangle = final_angle - initial_angle
        for i in range(self.nlines):
            r = radius + (1 if totalangle < 0 else -1) * self.gridoffset[i]
            s = self.gridspacing[i]
            if spacingmode == 'angle':
                minoffsetidx = np.argmin(self.gridoffset)
                minoffsetradius = radius + self.gridoffset[minoffsetidx]
                d = abs(totalangle) * minoffsetradius
            elif spacingmode == 'dist':
                d = abs(totalangle) * r
            else:
                raise ValueError(r'Invalid value spacingmode={spacingmode}')

            t = s - self.gridlines_pos[i] % s
            center = begin_point - radius * np.array([np.cos(initial_angle), np.sin(initial_angle)])
            while t <= d or np.isclose(t, d):
                angle = initial_angle + totalangle * t / d
                obj = gdspy.copy(self.gridobj).rotate(
                    angle + (pi/2 if totalangle >= 0 else -pi/2))
                pos = center + r * np.array([np.cos(angle), np.sin(angle)])
                obj = obj.translate(*pos)
                self.gridlines.append(obj)
                t += s
            self.gridlines_pos[i] += d

        #self.gridoffset = gridoffset
        self.total_len += radius * np.abs(totalangle)
        return self


    def turn(self, radius, angle, width=None, offset=None, spacingmode='dist'):
        """Like gdspy.FlexPath.turn() with additional argument `spacingmode`
        like in GridCPW.arc()."""
        angle = gdspy.path._angle_dict.get(angle, angle)
        initial_angle = self.get_angle() + (pi/2 if angle < 0 else -pi/2)
        return self.arc(radius, initial_angle, initial_angle + angle,
                        width, offset, spacingmode)


    def meander(self,
            length: float, # total length
            Nturn: int, # number of 180 turns
            radius: float, # turn radius
            lstart: float = 0,
            lend: float = 0):
        lm = length - lstart - lend
        assert (2*Nturn-2+(Nturn+1)*np.pi)*radius < lm, \
            "Radius or Nturn too large for given length."
        llong = (lm + 2*radius - (Nturn+1)*np.pi*radius) / Nturn
        lshort = llong/2 - radius

        angle = self.get_angle()
        straight = np.array([np.cos(angle), np.sin(angle)])
        perpendicular = np.array([-straight[1], straight[0]])

        if lstart != 0:
            self.segment(lstart*straight, relative=True)
        self.turn(radius, 'l')
        self.segment(lshort*perpendicular, relative=True)
        for i in range(Nturn-1):
            if i%2 == 0:
                self.turn(radius, 'rr')
                self.segment(-llong*perpendicular, relative=True)
            else:
                self.turn(radius, 'll')
                self.segment(llong*perpendicular, relative=True)
        if Nturn%2 == 1:
            self.turn(radius, 'rr')
            self.segment(-lshort*perpendicular, relative=True)
            self.turn(radius, 'l')
        else:
            self.turn(radius, 'll')
            self.segment(lshort*perpendicular, relative=True)
            self.turn(radius, 'r')
        if lend != 0:
            self.segment(lend*straight, relative=True)
        return self


    def objects(self):
        l = self.gridlines.copy()
        l.append(self.path)
        return l

    def to_polygonsets(self):
        l = [
            obj.to_polygonset() if not isinstance(obj, gdspy.PolygonSet) else obj
            for obj in self.gridlines]
        l.append(self.path.to_polygonset())
        return l


class CPWPath(GridPath):
    def __init__(self, start_points, total_width, ctr, avoidance,
                 antidotsize=2, antidotspacing=4,
                 innerantidotrows=2, outerantidotrows=3,
                 layer_gap=0, layer_avoidance=20,
                 layer_antidots=21, layer_antidotbound=22):
        if antidotsize == 0:
            outerantidotrows = innerantidotrows = 0
        self.ctr = ctr
        self.total_width = total_width
        self.avoidance = avoidance
        self.antidotsize = antidotsize
        self.antidotspacing = antidotspacing
        self.innerantidotrows = innerantidotrows
        self.outerantidotrows = outerantidotrows
        
        fpwidth, fpoffset, gridoffset = CPWPath._calc_sizes(
            total_width, ctr, avoidance, antidotsize, antidotspacing,
            innerantidotrows, outerantidotrows)

        fp = gdspy.FlexPath(
            start_points, width=fpwidth, offset=fpoffset,
            ends=[(0, 0), (0, 0), (avoidance, avoidance), (antidotsize/2, antidotsize/2)],
            layer=[layer_gap, layer_gap, layer_avoidance, layer_antidotbound])

        gridobj = Rectangle((-antidotsize/2, -antidotsize/2), (antidotsize/2, antidotsize/2),
                            layer=layer_antidots)
        super().__init__(fp, gridoffset, antidotspacing, gridobj)

    @staticmethod
    def _calc_sizes(total_width, ctr, avoidance,
                    antidotsize, antidotspacing,
                    innerantidotrows, outerantidotrows):
        gap = (total_width - ctr * total_width) / 2
        gapoffset = (ctr * total_width / 2) + gap/2
        fpwidth = [
            gap, gap, total_width+2*avoidance,
            total_width+2*(avoidance+antidotspacing*max((outerantidotrows-1), 0)+antidotsize*1.5)]
        fpoffset = [-gapoffset, gapoffset, 0, 0]

        gridoffset = np.concatenate(
            [[
                -total_width/2 - avoidance-antidotsize/2 - i*antidotspacing,
                +total_width/2 + avoidance+antidotsize/2 + i*antidotspacing,
            ] for i in range(outerantidotrows)]
            + [[
                -total_width*ctr/2 + avoidance + i*antidotspacing,
                +total_width*ctr/2 - avoidance - i*antidotspacing,
            ]  for i in range(innerantidotrows)]) if antidotsize else []

        return fpwidth, fpoffset, gridoffset

    def segment(self, end_point,
                total_width: float = None,
                ctr: float = None,
                relative: bool = False):
        if total_width is None:
            total_width = self.total_width
        if ctr is None:
            ctr = self.ctr

        fpwidth, fpoffset, gridoffset = CPWPath._calc_sizes(
            total_width, ctr, self.avoidance,
            self.antidotsize, self.antidotspacing,
            self.innerantidotrows, self.outerantidotrows)
        self.total_width = total_width
        self.ctr = ctr

        return super().segment(end_point, fpwidth, fpoffset, gridoffset, relative)
