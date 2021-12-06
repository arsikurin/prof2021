import math

EPS = 2.220446049250313e-16


class Point:
    def __init__(self, _x, _y=None, _polar=False):
        if isinstance(_x, Point):
            self.polar = _x.polar
            self.angle = _x.angle
            self.x = _x.x
            self.y = _x.y
            self.r = _x.r
        else:
            if _polar:
                self.r = _x
                self.angle = _y
                self.y = self.r * math.sin(self.angle)
                self.x = self.r * math.cos(self.angle)
                self.polar = True
            else:
                self.x = _x
                self.y = _y
                self.r = math.hypot(self.x, self.y)
                if abs(self.r) > EPS:
                    self.angle = math.acos(self.x / self.r)
                else:
                    self.angle = 0
                self.polar = False

    def __str__(self):
        return f"{self.x} {self.y}"

    def dist(self, _x=None, _y=None):
        if _x is None:
            return self.r
        elif _y is None:
            return math.hypot(self.x - _x.x, self.y - _x.y)
        return math.hypot(self.x - _x, self.y - _y)

    def __abs__(self):
        return self.r


class Vector(Point):
    def __init__(self, _x, _y=None, _a=None, _b=None):
        if isinstance(_y, Point):
            super().__init__(_y.x - _x.x, _y.y - _x.y)
        elif isinstance(_x, Point):
            super().__init__(_x.x, _x.y)
        elif isinstance(_b, float):
            super().__init__(_a - _x, _b - _y)
        elif isinstance(_y, float):
            super().__init__(_x, _y)
        elif isinstance(_x, Vector):
            super().__init__(_x.x, _y.x)

    def dot_product(self, other):
        return self.x * other.x + self.y * other.y

    def cross_product(self, other):
        return self.x * other.y - self.y * other.x

    def mul(self, other):
        return Vector(self.x * other, self.y * other)
