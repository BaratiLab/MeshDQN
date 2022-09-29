import numpy as np
from dolfin import *

try:
    from iufl import icompile
    from ufl.corealg.traversal import traverse_unique_terminals
    from iufl.operators import eigw

except ImportError:
    print('iUFL can be obtained from https://github.com/MiroK/ufl-interpreter')


class DragProbe(object):
    '''Integral proble of drag over the tagged mesh oriented exterior surface n.ds'''
    def __init__(self, mu, n, ds, tags, flow_dir=Constant((1, 0))):
        self.dim = flow_dir.ufl_shape[0]
        self.mu = mu
        self.n = n
        self.ds = ds
        self.tags = tags
        self.flow_dir = flow_dir

    def sample(self, u, p):
        '''Eval drag given the flow state'''
        # Stress
        sigma = 2*Constant(self.mu)*sym(grad(u)) - p*Identity(self.dim)

        # The drag form
        form = sum(dot(dot(sigma, self.n), self.flow_dir)*self.ds(i) for i in self.tags)

        return assemble(form)

class LiftProbe(object):
    '''Integral proble of lift over the tagged mesh oriented exterior surface n.ds'''
    def __init__(self, mu, n, ds, tags, flow_dir=Constant((0, 1))):
        self.dim = flow_dir.ufl_shape[0]
        self.mu = mu
        self.n = n
        self.ds = ds
        self.tags = tags
        self.flow_dir = flow_dir

    def sample(self, u, p):
        '''Eval drag given the flow state'''
        # Stress
        sigma = 2*Constant(self.mu)*sym(grad(u)) - p*Identity(self.dim)
        # The drag form
        form = sum(dot(dot(sigma, self.n), self.flow_dir)*self.ds(i) for i in self.tags)

        return assemble(form)


class DragProbeANN(DragProbe):
    '''Drag on the cylinder'''
    def __init__(self, flow, flow_dir=Constant((1, 0))):
        DragProbe.__init__(self,
                           mu=flow.viscosity,
                           n=flow.normal,
                           ds=flow.ext_surface_measure,
                           tags=flow.cylinder_surface_tags,
                           flow_dir=flow_dir)


class PenetratedDragProbe(object):
    '''Drag on a penetrated surface
    https://physics.stackexchange.com/questions/21404/strict-general-mathematical-definition-of-drag
    '''
    def __init__(self, rho, mu, n, ds, tags, flow_dir=Constant((1, 0))):
        self.dim = flow_dir.ufl_shape[0]
        self.mu = mu
        self.rho = rho
        self.n = n
        self.ds = ds
        self.tags = tags
        self.flow_dir = flow_dir

    def sample(self, u, p):
        '''Eval drag given the flow state'''
        mu, rho, n = self.mu, self.rho, self.n
        # Stress
        sigma = 2*Constant(mu)*sym(grad(u)) - p*Identity(self.dim)
        # The drag form
        form = sum(dot(-rho*dot(outer(u, u), n) + dot(sigma, n), self.flow_dir)*self.ds(i)
                   for i in self.tags)

        return assemble(form)


class PenetratedDragProbeANN(PenetratedDragProbe):
    '''Drag on a penetrated surface
    https://physics.stackexchange.com/questions/21404/strict-general-mathematical-definition-of-drag
    '''
    def __init__(self, flow, flow_dir=Constant((1, 0))):
        PenetratedDragProbe.__init__(self,
                                     rho=flow.density,
                                     mu=flow.viscosity,
                                     n=flow.normal,
                                     ds=flow.ext_surface_measure,
                                     tags=flow.cylinder_surface_tags,
                                     flow_dir=flow_dir)

