
# Single-View Reconstruction of a Ribbed Cylindrical Vault

**Course:** Image Analysis and Computer Vision (IACV)  
**University:** Politecnico di Milano  
**Programme:** Erasmus Mundus Joint Master in Imaging  
**Academic Year:** 2025–2026  
**Type:** Individual Homework Assignment  
**Tools:** MATLAB

## Overview

This assignment explores the reconstruction of a ribbed cylindrical vault from a single uncalibrated photograph using classical computer vision and projective geometry.

Unlike stereo or multi-view reconstruction, a single image does not directly provide depth information. The objective was therefore to exploit geometric constraints already present in the architecture, including parallel and orthogonal directions, structural symmetry, and the repeated arrangement of the vault's ribs.

The work covers vanishing geometry, planar rectification, camera calibration, reconstruction of a diagonal rib in 3D, and estimation of the cylindrical axis.

## Problem Statement

Given one image of a ribbed cylindrical vault, the assignment required:

1. Determining the vanishing line of a plane perpendicular to the vault's cylindrical axis.
2. Identifying the vanishing point corresponding to the axis direction.
3. Performing Euclidean rectification of a vertical cross-sectional plane.
4. Estimating the camera's intrinsic calibration matrix.
5. Reconstructing sampled points along a diagonal rib in 3D.
6. Estimating the position and orientation of the cylindrical axis in the camera coordinate system.

The scene provides a reference spacing of one unit between consecutive ribs of the same family.

## Methodology

### 1. Vanishing Points and Vanishing Lines

The first step was to identify image segments corresponding to known directions in the scene.

These included:
- Vertical architectural elements.
- Horizontal directions perpendicular to the cylindrical axis.
- Structural lines running parallel to the axis.

Parallel lines in 3D generally converge to a common vanishing point under perspective projection.

The vanishing line of a cross-sectional plane can be estimated from the vanishing points of two independent directions within that plane:

$$
\ell'_{\infty} =
\mathbf{v}_{vertical}
\times
\mathbf{v}_{transverse}
$$

This provides a projective constraint for subsequent rectification.

![Vanishing geometry](assets/vanishing_geometry.png)

*Selected architectural lines, vanishing points, and estimated vanishing line.*

### 2. Planar Rectification

The next task was to rectify a vertical cross-sectional plane perpendicular to the vault's axis.

The process involved:
- Using the estimated vanishing line to address projective distortion.
- Applying additional geometric constraints to obtain a metric rectification.
- Using the known structural relationships and reference spacing to reason about measurements and scale.

The purpose was to recover a representation in which geometric relationships could be examined more directly.

### 3. Camera Calibration

The camera's intrinsic matrix was estimated using vanishing points corresponding to mutually perpendicular scene directions.

For orthogonal vanishing directions, the image of the absolute conic provides the constraint:

$$
\mathbf{v}_i^T
K^{-T}K^{-1}
\mathbf{v}_j = 0
$$

where the camera intrinsic matrix is

$$
K =
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

The calibration relied on assumptions including zero camera skew and constraints on the principal point and pixel aspect ratio.

### 4. 3D Reconstruction of a Diagonal Rib

After estimating the camera intrinsics, image points were selected along one of the vault's diagonal ribs.

Each image point defines a ray originating from the camera centre:

$$
\mathbf{X}(\lambda)
=
\lambda K^{-1}\mathbf{p}
$$

where $\mathbf{p}$ is the homogeneous image coordinate and $\lambda$ determines the depth along the ray.

Since a single image cannot uniquely determine these depths, additional geometric assumptions were used.

These included a planar model of the diagonal rib and an approximate circular-arc representation.

The resulting sampled 3D points were visualized in the camera coordinate system.

![Reconstructed rib](assets/rib_reconstruction.png)

*MATLAB visualization of sampled 3D rib points. The plotted coordinates and scale require further verification.*

### 5. Cylindrical Axis Estimation

The final task was to estimate the cylindrical axis in 3D.

Its direction was inferred from the corresponding vanishing point and the estimated camera intrinsics.

The approximate axis position was then obtained using the assumed cylindrical geometry and the reconstructed scene features.

The axis can be represented parametrically as:

$$
\mathbf{L}(t)
=
\mathbf{P}_0 + t\mathbf{d}
$$

where $\mathbf{P}_0$ is a point on the axis and $\mathbf{d}$ is its direction vector.

## Discussion

This assignment was an exercise in recovering 3D information from limited visual evidence.

The main challenge was not extracting features alone, but identifying which geometric assumptions could constrain an otherwise ambiguous reconstruction.

It also highlighted the importance of camera calibration, scene geometry, and coordinate transformations in 3D computer vision.

The approach uses traditional geometric techniques rather than a learned depth estimation or reconstruction model.

### Limitations

- The reconstruction depends on scene-specific geometric assumptions.
- Some features were selected manually or semi-automatically.
- Depth cannot be uniquely recovered from one image without additional constraints.
- The reference spacing defines relative units rather than independently verified physical dimensions.
- Numerical calibration and reconstruction results require consistency checks; the supplied figures should not be interpreted as ground-truth-validated 3D geometry.

## Repository Contents

The project folder contains the MATLAB implementation, the assignment materials, and the homework report.

The report documents the derivations, assumptions, intermediate results, and visualizations.

## Context

Completed individually in January 2026 as part of the **Image Analysis and Computer Vision** course at **Politecnico di Milano**, during the first year of the **Erasmus Mundus Joint Master in Imaging (2025–2026)**.
