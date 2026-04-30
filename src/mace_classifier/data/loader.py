from typing import Tuple

import torch
from torch import Tensor
import numpy as np

from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

BRAVAIS_LABELS = {
    "aP": 0,
    "mP": 1,
    "mS": 2,
    "oP": 3,
    "oS": 4,
    "oI": 5,
    "oF": 6,
    "tP": 7,
    "tI": 8,
    "hP": 9,
    "hR": 10,
    "cP": 11,
    "cI": 12,
    "cF": 13,
}


def load_prototype(cif_path: str) -> Structure:
    """
    Load CIF, expand symmetry if needed, return Structure
    object with explicit positions, species, cell.
    """
    struct = Structure.from_file(cif_path)
    sga = SpacegroupAnalyzer(struct)
    struct = sga.get_conventional_standard_structure()
    # normalize species to Z=1,2 for binary prototypes (and Z=1 for unary)
    unique_species = struct.composition.elements
    if len(unique_species) == 1:
        struct = Structure(
            lattice=struct.lattice,
            species=[1] * len(struct),
            coords=struct.cart_coords,
            coords_are_cartesian=True,
        )
    elif len(unique_species) == 2:
        species_map = {
            unique_species[0]: 1,
            unique_species[1]: 2,
        }
        struct = Structure(
            lattice=struct.lattice,
            species=[species_map[sp] for sp in struct.species],
            coords=struct.cart_coords,
            coords_are_cartesian=True,
        )
        print(
            f"Mapped species {unique_species[0]}->{species_map[unique_species[0]]}, {unique_species[1]}->{species_map[unique_species[1]]}"
        )
    else:
        raise ValueError(
            f"Only unary/binary prototypes are supported, found {len(unique_species)} species: {unique_species}"
        )
    return struct


def make_supercell(
    atoms: Structure,
    min_box_length: float = 10.0,  # Angstrom
    min_atoms: int = 64,
) -> Structure:
    """
    Build smallest supercell satisfying min_box_length and min_atoms.
    Compute replication factors per axis independently. Then increase
    uniformly to hit min_atoms if needed.
    """
    lat = atoms.lattice
    rep_factors = [1, 1, 1]
    for i in range(3):
        while lat.matrix[i][i] * rep_factors[i] < min_box_length:
            rep_factors[i] += 1
    supercell = atoms * rep_factors
    while len(supercell) < min_atoms:
        rep_factors = [f + 1 for f in rep_factors]
        supercell = atoms * rep_factors
    print(
        f"Supercell replication factors: {rep_factors}, total atoms: {len(supercell)}"
    )

    assert all(supercell.lattice.matrix[i][i] >= min_box_length for i in range(3))
    assert len(supercell) >= min_atoms
    # check that coordinates still match TODO

    return supercell


def rescale_to_dnn(atoms: Structure, target_dnn: float) -> Structure:
    """
    Rescale structure to target_dnn (nearest neighbor distance).
    """
    nn_distance = min(
        atoms.distance_matrix.flatten()[atoms.distance_matrix.flatten() > 0]
    )
    scale_factor = target_dnn / nn_distance
    print(
        f"Rescaling structure from d_nn={nn_distance:.3f} to target_dnn={target_dnn:.3f} with scale factor {scale_factor:.3f}"
    )
    old_volume = atoms.lattice.volume
    new_volume = old_volume * scale_factor**3
    scaled_lattice = atoms.lattice.scale(new_volume)
    scaled_structure = Structure(
        lattice=scaled_lattice,
        species=atoms.species,
        coords=atoms.cart_coords * scale_factor,
        coords_are_cartesian=True,
    )
    return scaled_structure


def compute_scaling_bounds(
    atoms: Structure, r_cut: float = 5.0, d_nn_init: float = 1.5
) -> Tuple[float, float]:
    """
    Compute scaling bounds to preserve neighbor shells within r_cut.

    atoms: input structure
    r_cut: distance cutoff for neighbor shells (Angstrom)
    d_nn_init: initial nearest neighbor distance (Angstrom)

    We sort distances and count the number of unique distances (with some tolerance) achieved within
    r_cut using a scaling of d_nn_init. We then compute the minimum and maximum scaling that keeps the number
    of unique distances the same (i.e. the same neighbor shells) within r_cut.
    This algorithm assumes that the structure is perfect.
    """
    atoms = rescale_to_dnn(atoms, target_dnn=d_nn_init)
    dists = atoms.distance_matrix
    n_atoms = len(atoms)
    pair_dists = dists[np.triu_indices(n_atoms, k=1)]
    pair_dists = pair_dists[pair_dists > 0]
    pair_dists = pair_dists[pair_dists < r_cut * 5]  # heuristic

    unique_dists = np.unique(np.round(pair_dists, decimals=2))

    # max shell below r_cut
    dists_below_rcut = unique_dists[unique_dists < r_cut]
    assert (
        len(dists_below_rcut) > 0
    ), "No neighbor shells found below r_cut. Consider increasing r_cut or decreasing initial d_nn scaling."
    max_shell_dist = dists_below_rcut[-1]
    max_dnn = r_cut / max_shell_dist * d_nn_init

    # min shell above r_cut
    dists_above_rcut = unique_dists[unique_dists > r_cut]
    assert (
        len(dists_above_rcut) > 0
    ), "No neighbor shells found above r_cut. Consider decreasing r_cut, increasing initial d_nn scaling, or increasing the supercell size."
    min_shell_dist = unique_dists[unique_dists > r_cut][0]
    min_dnn = r_cut / min_shell_dist * d_nn_init

    return min_dnn, max_dnn


def suggest_scaling_factors(
    atoms: Structure,
    r_cut: float = 5.0,
    d_nn_init: float = 1.5,
    num_factors: int = 10,
    cluster_lower: float = 2.0,
    cluster_upper: float = 3.2,
    cluster_fraction: float = 0.5,
) -> Tensor:
    """
    Suggest scaling factors within bounds to preserve neighbor shells. Cluster
    more around [2.0, 3.2] angstrom, which is typical for many materials.
    """
    min_dnn, max_dnn = compute_scaling_bounds(atoms, r_cut, d_nn_init)
    print(f"Computed scaling bounds: [{min_dnn:.3f}, {max_dnn:.3f}]")
    factors = torch.linspace(min_dnn, max_dnn, num_factors)
    if cluster_fraction > 0:
        cluster_min = max(cluster_lower, min_dnn)
        cluster_max = min(cluster_upper, max_dnn)
        cluster_factors = torch.linspace(
            cluster_min, cluster_max, int(num_factors * cluster_fraction)
        )
        factors = torch.cat([factors, cluster_factors])
        factors = torch.unique(factors)  # remove duplicates
    return factors


def to_tensor(atoms: Structure) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Convert Structure to PyTorch tensors: positions (N, 3), cell (3, 3), species (N,)
    """
    positions = torch.tensor(atoms.cart_coords, dtype=torch.float32)
    cell = torch.tensor(atoms.lattice.matrix, dtype=torch.float32)
    species = torch.tensor([atom.specie.Z for atom in atoms], dtype=torch.long)
    return positions, cell, species
