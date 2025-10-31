"""
HYBRID BLOCK-FEM COUPLING METHODS
==================================

This module implements advanced coupling strategies for hybrid discrete-continuum 
structural analysis, combining rigid block assemblies with finite element meshes.

THEORETICAL BACKGROUND AND RESEARCH BASIS:
------------------------------------------

The coupling problem between discrete element methods (DEM/blocks) and continuum 
finite elements (FEM) is a well-studied challenge in computational mechanics:

1. **Multi-Point Constraints (MPC) / Lagrange Multipliers**
   - Farhat & Roux (1991): "A method of finite element tearing and interconnecting"
   - Klarbring (1988): "Large displacement contact problem"
   - Enforces kinematic compatibility through constraint equations
   - Gold standard for accuracy but increases system size

2. **Penalty Methods**
   - Wriggers & Simo (1985): "A note on tangent stiffness for fully nonlinear contact"
   - Approximates constraints through springs with high stiffness
   - Simpler to implement, no additional DOFs
   - Requires careful penalty parameter selection

3. **Mortar Methods**
   - Belgacem et al. (1998): "The mortar finite element method for contact problems"
   - Wohlmuth (2001): "Discretization methods and iterative solvers"
   - Optimal for non-matching meshes
   - Weak enforcement of constraints on interface

4. **Nitsche's Method**
   - Nitsche (1971): "Über ein Variationsprinzip"
   - Annavarapu et al. (2012): "A robust Nitsche's formulation for interface problems"
   - Weak constraint enforcement without Lagrange multipliers
   - Consistent and stable

5. **Interface Elements**
   - Goodman et al. (1968): "A model for the mechanics of jointed rock"
   - Desai et al. (1984): "Thin-layer element for interfaces and joints"
   - Zero-thickness elements for cohesive zones
   - Natural for modeling adhesion/debonding

6. **Arlequin Method**
   - Ben Dhia (1998): "Multiscale mechanical problems: the Arlequin method"
   - Overlapping domain decomposition
   - Allows smooth transition between discrete and continuum

IMPLEMENTATION NOTES:
--------------------
Each method is implemented as a separate class that can be added to the Hybrid
structure. The user can choose the most appropriate method based on:
- Accuracy requirements
- Computational cost constraints
- Problem physics (contact, adhesion, etc.)
- Mesh compatibility

Author: Claude (based on extensive FEM/DEM coupling literature)
Date: 2025
"""