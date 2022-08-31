from dolfin import *
import numpy as np
from probes import DragProbe, LiftProbe, PenetratedDragProbe
from matplotlib import pyplot as plt
from tqdm import tqdm
import time

# Subdomains to mark
class Bndry(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class EdgeBndry(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[1] > 0.5 - 2*DOLFIN_EPS) or (x[1] < -0.5 + 2*DOLFIN_EPS))

class AirfoilBndry(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] < 3.0 - DOLFIN_EPS) and \
                               (x[0] > -0.5 + DOLFIN_EPS) and \
                               (x[1] < 0.5 - DOLFIN_EPS) and \
                               (x[1] > -0.5 + DOLFIN_EPS)

class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < -0.5 + DOLFIN_EPS and on_boundary

class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 3.0 - 2*DOLFIN_EPS and on_boundary

# Profile
def constant_profile(mesh, degree):
    '''
    Time independent inflow profile.
    '''
    bot = mesh.coordinates().min(axis=0)[1]
    top = mesh.coordinates().max(axis=0)[1]

    H = top - bot

    Um = 1.5

    return Expression(('-4*Um*(x[1]-bot)*(x[1]-top)/H/H','0'), bot=bot, top=top, H=H, Um=Um, degree=degree, time=0)


class FlowSolver(object):
    '''IPCS scheme with explicit treatment of nonlinearity.'''
    def __init__(self, flow_params, geometry_params, solver_params):
        # Using very simple IPCS solver
        mu = Constant(flow_params['mu'])              # dynamic viscosity
        rho = Constant(flow_params['rho'])            # density
        
        # Load airfoil mesh
        mesh_file = geometry_params['mesh']
        self.mesh = Mesh()
        f = XDMFFile(mesh_file)
        f.read(self.mesh)
        f.close()

        # Smooth mesh
        self.smooth = solver_params['smooth']
        print("SMOOTH?: {}".format(self.smooth))
        if(self.smooth):
            #for i in range(10):
            #self.mesh.smooth(1)
            self.mesh.smooth(50)

        # Remember inflow profile function in case it is time dependent
        if(flow_params['inflow'] == 'constant'):
            self.inflow_profile = constant_profile(self.mesh, degree=2)
        else:
            self.inflow_profile = flow_params['inflow']

        self.bmesh = BoundaryMesh(self.mesh, 'local')
        self.removable = []
        for coord in self.mesh.coordinates():
            self.removable.append(coord not in self.bmesh.coordinates())
        
        # Set up markers
        surfaces, bnd, airfoil_bnd, inflow, outflow = self.mark_boundaries()
        self.bnd = bnd

        # Define function spaces
        V = VectorFunctionSpace(self.mesh, 'Lagrange', 2)
        Q = FunctionSpace(self.mesh, 'Lagrange', 1)

        # Define trial and test functions
        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)

        u_n, p_n = Function(V), Function(Q)
        u_, p_ = Function(V), Function(Q)  # Solve into these

        dt = Constant(solver_params['dt'])

        # Define expressions used in variational forms
        U  = Constant(0.5)*(u_n + u)
        n  = FacetNormal(self.mesh)
        f  = Constant((0, 0))

        epsilon = lambda u :sym(nabla_grad(u))

        sigma = lambda u, p: 2*mu*epsilon(u) - p*Identity(2)

        F1 = (rho*dot((u - u_n) / dt, v)*dx
              + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx
              + inner(sigma(U, p_n), epsilon(v))*dx
              + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds
              - dot(f, v)*dx)

        a1, L1 = lhs(F1), rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/dt)*div(u_)*q*dx

        # Define variational problem for step 3
        a3 = dot(u, v)*dx
        L3 = dot(u_, v)*dx - dt*dot(nabla_grad(p_ - p_n), v)*dx

        # Inflow boundary condition
        bcu_inlet = DirichletBC(V, self.inflow_profile, surfaces, 2)
        bcp_outflow = DirichletBC(Q, 0, outflow)

        # No slip
        bcu_noslip = DirichletBC(V, (0,0), bnd)
        bcu_airfoil_noslip = DirichletBC(V, (0,0), airfoil_bnd)

        # All bcs objects togets
        bcu = [bcu_inlet, bcu_airfoil_noslip, bcu_noslip]
        bcp = [bcp_outflow]

        As = [Matrix() for i in range(3)]
        bs = [Vector() for i in range(3)]

        # Assemble matrices
        assemblers = [SystemAssembler(a1, L1, bcu),
                      SystemAssembler(a2, L2, bcp),
                      SystemAssembler(a3, L3, bcu)]

        # Apply bcs to matrices (this is done once)
        for a, A in zip(assemblers, As):
            a.assemble(A)

        # Chose between direct and iterative solvers
        self.solver_type = solver_params.get('la_solve', 'lu')
        assert self.solver_type in ('lu', 'la_solve')

        if self.solver_type == 'lu':
            solvers = list(map(lambda x: LUSolver("mumps"), range(3)))
        else:
            solvers = [KrylovSolver('bicgstab', 'hypre_amg'),  # Very questionable preconditioner
                       KrylovSolver('cg', 'hypre_amg'),
                       KrylovSolver('cg', 'hypre_amg')]

        # Set matrices for once, likewise solver don't change in time
        for s, A in zip(solvers, As):
            s.set_operator(A)

        gtime = 0.  # External clock

        # Things to remeber for evolution
        # Keep track of time so that we can query it outside
        self.gtime, self.dt = gtime, dt

        self.solvers = solvers
        self.assemblers = assemblers
        self.bs = bs
        self.u_, self.u_n = u_, u_n
        self.p_, self.p_n = p_, p_n

        # Rename u_, p_ for to standard names (simplifies processing)
        u_.rename('velocity', '0')
        p_.rename('pressure', '0')

        # Also expose measure for assembly of outputs outside
        self.ext_surface_measure = Measure('ds', domain=self.mesh, subdomain_data=surfaces)

        # Things to remember for easier probe configuration
        self.viscosity = mu
        self.density = rho
        self.normal = n

        # Set up probes
        self.drag_probe = DragProbe(self.viscosity, self.normal, self.ext_surface_measure, tags=[1])
        self.lift_probe = LiftProbe(self.viscosity, self.normal, self.ext_surface_measure, tags=[1])
        self.accumulated_drag = []
        self.accumulated_lift = []

        self.num_vertices = len(self.mesh.coordinates())


    def mark_boundaries(self):
        # Get surface
        surfaces = MeshFunction('size_t', self.mesh, self.mesh.topology().dim()-1)
        surfaces_bool = MeshFunction('bool', self.mesh, self.mesh.topology().dim()-1)
        surfaces_double = MeshFunction('double', self.mesh, self.mesh.topology().dim()-1)

        # Set markings
        surfaces.set_all(4)
        surfaces_bool.set_all(False)
        surfaces_double.set_all(0.4)

        # No slip on top/bottom walls
        bnd = EdgeBndry()
        bnd.mark(surfaces, 0)
        bnd.mark(surfaces_double, 0.0)

        # No slip on airfoil
        airfoil_bnd = AirfoilBndry()
        airfoil_bnd.mark(surfaces, 1)
        airfoil_bnd.mark(surfaces_double, 0.1)
        
        # Mark inflow
        inflow = Inflow()
        inflow.mark(surfaces, 2)
        inflow.mark(surfaces_double, 0.2)
        
        # Mark outflow
        outflow = Outflow()
        outflow.mark(surfaces, 3)
        outflow.mark(surfaces_double, 0.3)
        outflow.mark(surfaces_bool, True)

        return surfaces, bnd, airfoil_bnd, inflow, outflow

    def remesh(self, mesh):
        # Set mesh to new mesh
        self.mesh = mesh
        if(self.smooth):
            self.mesh.smooth(50)

        # Set up markers
        surfaces, bnd, airfoil_bnd, inflow, outflow = self.mark_boundaries()
        self.bnd = bnd

        ###
        # Reset boundaries -> shoudln't mess up previous implementation
        # removable is never updated after initialization
        ###
        self.bmesh = BoundaryMesh(self.mesh, 'local')
        self.removable = []
        for coord in self.mesh.coordinates():
            self.removable.append(coord not in self.bmesh.coordinates())

        # Define function spaces
        V = VectorFunctionSpace(self.mesh, 'Lagrange', 2)
        Q = FunctionSpace(self.mesh, 'Lagrange', 1)

        # Define trial and test functions
        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)

        u_n, p_n = Function(V), Function(Q)
        u_, p_ = Function(V), Function(Q)  # Solve into these

        # Define expressions used in variational forms
        U  = Constant(0.5)*(u_n + u)
        n  = FacetNormal(self.mesh)
        f  = Constant((0, 0))

        epsilon = lambda u :sym(nabla_grad(u))
        sigma = lambda u, p: 2*mu*epsilon(u) - p*Identity(2)

        # Reuse previous parameters
        mu = self.viscosity
        rho = self.density
        dt = self.dt

        F1 = (rho*dot((u - u_n) / dt, v)*dx
              + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx
              + inner(sigma(U, p_n), epsilon(v))*dx
              + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds
              - dot(f, v)*dx)

        a1, L1 = lhs(F1), rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/dt)*div(u_)*q*dx

        # Define variational problem for step 3
        a3 = dot(u, v)*dx
        L3 = dot(u_, v)*dx - dt*dot(nabla_grad(p_ - p_n), v)*dx

        # Inflow boundary condition
        bcu_inlet = DirichletBC(V, self.inflow_profile, surfaces, 2)
        bcp_outflow = DirichletBC(Q, 0, outflow)

        # No slip
        bcu_noslip = DirichletBC(V, (0,0), bnd)
        bcu_airfoil_noslip = DirichletBC(V, (0,0), airfoil_bnd)

        # All bcs objects togets
        bcu = [bcu_inlet, bcu_airfoil_noslip, bcu_noslip]
        #bcp = [bcu_inlet, bcp_outflow]
        bcp = [bcp_outflow]

        As = [Matrix() for i in range(3)]
        bs = [Vector() for i in range(3)]

        # Assemble matrices
        assemblers = [SystemAssembler(a1, L1, bcu),
                      SystemAssembler(a2, L2, bcp),
                      SystemAssembler(a3, L3, bcu)]

        # Apply bcs to matrices (this is done once)
        for a, A in zip(assemblers, As):
            a.assemble(A)

        # Chose between direct and iterative solvers
        if self.solver_type == 'lu':
            solvers = list(map(lambda x: LUSolver("mumps"), range(3)))
        else:
            solvers = [KrylovSolver('bicgstab', 'hypre_amg'),  # Very questionable preconditioner
                       KrylovSolver('cg', 'hypre_amg'),
                       KrylovSolver('cg', 'hypre_amg')]

        # Set matrices for once, likewise solver don't change in time
        for s, A in zip(solvers, As):
            s.set_operator(A)

            #if(self.solver_type == "lu"):
            #    s.parameters['reuse_factorization'] = True

        # Things to remeber for evolution
        self.gtime = 0 # Reset external clock

        self.solvers = solvers
        self.assemblers = assemblers
        self.bs = bs
        self.u_, self.u_n = u_, u_n
        self.p_, self.p_n = p_, p_n

        # Also expose measure for assembly of outputs outside
        self.ext_surface_measure = Measure('ds', domain=self.mesh, subdomain_data=surfaces)

        # Things to remember for easier probe configuration
        self.normal = n
        self.drag_probe = DragProbe(self.viscosity, self.normal, self.ext_surface_measure, tags=[1])
        self.lift_probe = LiftProbe(self.viscosity, self.normal, self.ext_surface_measure, tags=[1])

        self.accumulated_drag = []
        self.accumulated_lift = []

        # Rename u_, p_ for to standard names (simplifies processing)
        u_.rename('velocity', '0')
        p_.rename('pressure', '0')

        self.num_vertices = len(self.mesh.coordinates())


    def evolve(self):
        '''Make one time step with the given values of jet boundary conditions'''

        # # Update bc expressions
        # Make a step
        self.gtime += self.dt(0)

        inflow = self.inflow_profile
        if hasattr(inflow, 'time'):
            inflow.time = self.gtime

        assemblers, solvers = self.assemblers, self.solvers
        bs = self.bs
        u_, p_ = self.u_, self.p_
        u_n, p_n = self.u_n, self.p_n

        for (assembler, b, solver, uh) in zip(assemblers, bs, solvers, (u_, p_, u_)):
            assembler.assemble(b)
            solver.solve(uh.vector(), b)

        u_n.assign(u_)
        p_n.assign(p_)

        drag = self.drag_probe.sample(u_n, p_n)
        self.accumulated_drag.append(drag)

        lift = self.lift_probe.sample(u_n, p_n)
        self.accumulated_lift.append(lift)

        if(np.isclose((self.gtime+0.000001)%(100*self.dt(0)), 0, atol=1e-5)):
            print("TIME: {0:.4f}s \t DRAG: {1:.4f}\t LIFT: {2:.4f}".format(self.gtime, drag, lift))


        # Share with the world
        return u_, p_, drag, lift


if __name__ == '__main__':
    #airfoil = 'bacnlf'
    #airfoil = 'ag11_0.0'
    #airfoil = 'ys930_0.08000'
    airfoil = 'square0.2000_0.09000'
    #airfoil = 'goe435'
    flow_params = {
            #'mu': 1E-3,
            'mu': 1E-3,
            'rho': 1.,
            #'inflow': Expression("1.0", t=0.0, degree=2),
            'inflow': "constant",
            
    }

    geometry_params = {
            'mesh': 'mesh_sweep/square_xdmf_files/{}_triangle.xdmf'.format(airfoil)
    }

    solver_params = {
            'dt': 0.001,
            'solver_type': 'lu'
    }

    solver = FlowSolver(
                flow_params=flow_params,
                geometry_params=geometry_params,
                solver_params=solver_params
    )

    total_drag = []
    #for i in tqdm(range(5000)):
    for i in range(5000):
        u, p, drag = solver.evolve()
        total_drag.append(drag)

    print("Average drag: {}".format(np.mean(total_drag[-50:])))
    print("Average lift: {}".format(np.mean(solver.accumulated_lift[-50:])))
    fig, ax = plt.subplots()
    ax.plot(total_drag)
    plt.savefig("./airfoil_results/{}_drag_plot.png".format(airfoil))
    np.save("./airfoil_results/{}_drag.npy".format(airfoil), solver.accumulated_drag)
    np.save("./airfoil_results/{}_lift.npy".format(airfoil), solver.accumulated_lift)
    plt.close()
    
    plot(u)
    plt.savefig("./airfoil_results/{}_velocity.png".format(airfoil))
    plt.close()

    plot(p)
    plt.savefig("./airfoil_results/{}_pressure.png".format(airfoil))
    plt.close()

