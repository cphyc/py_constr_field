'''Classes to represent linear constrains.'''
import attr


@attr.s
class Constrain(object):
    position = attr.ib()
    operator = attr.ib()
    R = attr.ib()
    filter = attr.ib()
