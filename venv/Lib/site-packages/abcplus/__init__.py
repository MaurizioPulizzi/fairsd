"""
abcplus is a stand-in for the stdlib `abc` module that adds an @finalmethod decorator for final methods
"""

import abc

abstractproperty = abc.abstractproperty  # pylint: disable=invalid-name
abstractmethod = abc.abstractmethod


def finalmethod(funcobj):
    """A decorator indicating final methods.

    Requires that the metaclass is abcplus.ABCMeta or derived from it.  A
    class that has a metaclass derived from ABCMeta cannot be
    instantiated if any of the final methods are overridden.
    The abstract methods can be called using any of the normal
    'super' call mechanisms.

    Usage:

        class C:
            __metaclass__ = abcplus.ABCMeta
            @abcplus.finalmethod
            def my_final_method(self, ...):
                ...
    """
    funcobj.__isfinalmethod__ = True
    return funcobj


class ABCMeta(abc.ABCMeta):
    """A modified version of ABCMeta metaclass that adds support for checking final methods"""
    def __new__(mcls, name, bases, namespace):
        """Check the inheritance tree for any final methods and then verify we aren't overriding them."""

        def _check_base_cls(name, namespace, base):
            """Recursively check parent classes for final methods that we have overridden and throw TypeError if found.
            """
            if base.__bases__:
                for base_parent in base.__bases__:
                    _check_base_cls(name, namespace, base_parent)

            for method in getattr(base, "__finalmethods__", set()):
                if method in namespace:
                    error = "Class {cls} cannot override final method {method} from parent class {base}".format(
                        cls=name,
                        method=method,
                        base=base.__name__
                    )
                    raise TypeError(error)

        for base in bases:
            _check_base_cls(name, namespace, base)

        cls = super(ABCMeta, mcls).__new__(mcls, name, bases, namespace)

        finals = set(ns_name
                     for ns_name, ns_value in namespace.items()
                     if getattr(ns_value, "__isfinalmethod__", False))
        cls.__finalmethods__ = frozenset(finals)

        return cls
