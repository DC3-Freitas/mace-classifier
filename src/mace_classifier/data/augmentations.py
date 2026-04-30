import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from pymatgen.core import Lattice, Structure
from torch import Tensor


@dataclass
class AugmentedStructure:
    positions: Tensor  # (N, 3)
    cell: Tensor  # (3, 3)
    species: Tensor  # (N,)
    bravais_lattice: str = None
    ordering: str = None
    sigma: float = None  # structural perturbation
    p: float = None  # species shuffle fraction
    family_id: int = None
    augmentation_type: str = None

    def to_structure(self) -> Structure:
        lattice = Lattice(self.cell.numpy())
        species_list = self.species.numpy().tolist()
        positions_list = self.positions.numpy().tolist()
        return Structure(lattice, species_list, positions_list)

    def to_xyz(self, filename: str):
        with open(filename, "w") as f:
            f.write(f"{len(self.species)}\n")
            f.write(
                f"Augmentation type: {self.augmentation_type}, sigma: {self.sigma}, p: {self.p}\n"
            )
            for i in range(len(self.species)):
                f.write(
                    f"{self.species[i].item()} {self.positions[i, 0].item()} {self.positions[i, 1].item()} {self.positions[i, 2].item()}\n"
                )


def apply_gaussian_noise(
    positions: Tensor,  # (N, 3)
    cell: Tensor,  # (3, 3)
    sigma: float = 0.01,  # noise standard deviation in Angstrom
    seed: int = None,
) -> Tensor:
    """
    Apply Gaussian noise to atomic positions, ensuring that the resulting
    positions remain within the unit cell defined by 'cell'.
    """
    if seed is not None:
        torch.manual_seed(seed)
    dnn = torch.min(
        torch.norm(positions[:, None, :] - positions[None, :, :], dim=-1)
        + torch.eye(len(positions)) * 1e6
    )  # compute nearest neighbor distance. ignore self-distance by adding large value to diagonal
    overlap_thresh = 0.5 * dnn.min()  # threshold for considering atoms as overlapping
    successful_generation = False
    while not successful_generation:
        noise = torch.randn_like(positions) * sigma
        noisy_positions = positions + noise
        # Wrap positions back into the unit cell
        for i in range(3):
            noisy_positions[:, i] = noisy_positions[:, i] % cell[i, i]
        # Test for overlaps
        pairwise_distances = torch.norm(
            noisy_positions[:, None, :] - noisy_positions[None, :, :], dim=-1
        )
        if torch.all(
            pairwise_distances + torch.eye(len(positions)) * 1e6 > overlap_thresh
        ):
            successful_generation = True
    return noisy_positions


def apply_cell_strain(
    positions: Tensor,  # (N, 3)
    cell: Tensor,  # (3, 3)
    magnitude: float,  # maximum strain component
) -> Tuple[Tensor, Tensor]:
    # compute strain tensor
    strain = torch.randn(3, 3) * magnitude
    strain = 0.5 * (strain + strain.T)
    # apply strain to cell
    new_cell = cell + strain @ cell
    new_positions = positions + (strain @ positions.T).T
    return new_positions, new_cell


def apply_species_shuffle(
    species: Tensor,  # (N,)
    fraction: float,  # fraction of cross-species swaps to perform
    seed: int = None,
) -> Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    permuted_species = species.clone()
    unique_species = torch.unique(species)
    for i in range(len(unique_species)):
        for j in range(i + 1, len(unique_species)):
            mask_i = species == unique_species[i]
            mask_j = species == unique_species[j]
            num_swaps = int(fraction * min(mask_i.sum(), mask_j.sum()).item())
            if num_swaps > 0:
                idx_i = torch.where(mask_i)[0]
                idx_j = torch.where(mask_j)[0]
                swap_idx_i = idx_i[torch.randperm(len(idx_i))[:num_swaps]]
                swap_idx_j = idx_j[torch.randperm(len(idx_j))[:num_swaps]]
                permuted_species[swap_idx_i] = unique_species[j]
                permuted_species[swap_idx_j] = unique_species[i]
    return permuted_species


def apply_species_permutation(
    species: Tensor,  # (N,)
) -> Tensor:
    species_set = torch.unique(species)
    if len(species_set) == 1:
        return species.clone()
    if len(species_set) > 2:
        raise ValueError("apply_species_permutation is only defined for binary systems")
    permuted_species = species.clone()
    permuted_species[species == species_set[0]] = species_set[1]
    permuted_species[species == species_set[1]] = species_set[0]
    return permuted_species


def apply_mono_species(
    species: Tensor,  # (N,)
) -> Tensor:
    # Set all species to the same label (1)
    return torch.ones_like(species, dtype=torch.long)


def apply_cross_augmentation(
    positions: Tensor,  # (N, 3)
    cell: Tensor,  # (3, 3)
    species: Tensor,  # (N,)
    sigma: float,
    shuffle_fraction: float,
    seed: int = None,
) -> Tuple[Tensor, Tensor]:
    new_positions = apply_gaussian_noise(positions, cell, sigma, seed)
    new_species = apply_species_shuffle(species, shuffle_fraction, seed)
    return new_positions, new_species


def sample_augmentation_family(
    parent: Structure,
    family_id: int,
    batch_size: int | None,
    sigmas: List[float],
    ps: List[float],
    num_cross: int = 2,
) -> List[AugmentedStructure]:
    """
    If batch_size is None, the output will sweep each sigma and species permutation
    across the provided list of values, as well as solely applying each augmentation type
    (chemical or structural). We'll also add a species permuted version (with types swapped),
    a mono-species version (all species set to the same label), and the original parent structure.

    If batch_size is an integer, the output will contain that many augmentations,
    with sigma and p randomly sampled from the provided lists.
    """
    if batch_size is None:
        augmentations = []
        for _ in num_cross:
            sigma = random.choice(sigmas)
            p = random.choice(ps)
            new_positions, new_species = apply_cross_augmentation(
                torch.tensor(parent.cart_coords, dtype=torch.float),
                torch.tensor(parent.lattice.matrix, dtype=torch.float),
                torch.tensor(parent.atomic_numbers, dtype=torch.long),
                sigma,
                p,
            )
            augmentations.append(
                AugmentedStructure(
                    positions=new_positions,
                    cell=torch.tensor(parent.lattice.matrix, dtype=torch.float),
                    species=new_species,
                    sigma=sigma,
                    p=p,
                    family_id=family_id,
                    augmentation_type="cross",
                )
            )
        # Add pure structural augmentations (sigma > 0, p = 0)
        for sigma in sigmas:
            new_positions = apply_gaussian_noise(
                torch.tensor(parent.cart_coords, dtype=torch.float),
                torch.tensor(parent.lattice.matrix, dtype=torch.float),
                sigma,
            )
            augmentations.append(
                AugmentedStructure(
                    positions=new_positions,
                    cell=torch.tensor(parent.lattice.matrix, dtype=torch.float),
                    species=torch.tensor(parent.atomic_numbers, dtype=torch.long),
                    sigma=sigma,
                    p=0.0,
                    family_id=family_id,
                    augmentation_type="structural",
                )
            )
        # Add pure chemical augmentations (sigma = 0, p > 0)
        for p in ps:
            new_species = apply_species_shuffle(
                torch.tensor(parent.atomic_numbers, dtype=torch.long), p
            )
            augmentations.append(
                AugmentedStructure(
                    positions=torch.tensor(parent.cart_coords, dtype=torch.float),
                    cell=torch.tensor(parent.lattice.matrix, dtype=torch.float),
                    species=new_species,
                    sigma=0.0,
                    p=p,
                    family_id=family_id,
                    augmentation_type="chemical",
                )
            )
        # Add species permutation (swap species labels)
        permuted_species = apply_species_permutation(
            torch.tensor(parent.atomic_numbers, dtype=torch.long)
        )
        augmentations.append(
            AugmentedStructure(
                positions=torch.tensor(parent.cart_coords, dtype=torch.float),
                cell=torch.tensor(parent.lattice.matrix, dtype=torch.float),
                species=permuted_species,
                sigma=0.0,
                p=1.0,
                family_id=family_id,
                augmentation_type="permutation",
            )
        )
        # Add mono-species version
        mono_species = apply_mono_species(
            torch.tensor(parent.atomic_numbers, dtype=torch.long)
        )
        augmentations.append(
            AugmentedStructure(
                positions=torch.tensor(parent.cart_coords, dtype=torch.float),
                cell=torch.tensor(parent.lattice.matrix, dtype=torch.float),
                species=mono_species,
                sigma=0.0,
                p=1.0,
                family_id=family_id,
                augmentation_type="mono_species",
            )
        )
        # Add original parent structure
        augmentations.append(
            AugmentedStructure(
                positions=torch.tensor(parent.cart_coords, dtype=torch.float),
                cell=torch.tensor(parent.lattice.matrix, dtype=torch.float),
                species=torch.tensor(parent.atomic_numbers, dtype=torch.long),
                sigma=0.0,
                p=0.0,
                family_id=family_id,
                augmentation_type="original",
            )
        )
        return augmentations
    else:
        augmentations = []
        for _ in range(batch_size):
            sigma = float(torch.choice(torch.tensor(sigmas)))
            p = float(torch.choice(torch.tensor(ps)))
            new_positions, new_species = apply_cross_augmentation(
                torch.tensor(parent.cart_coords, dtype=torch.float),
                torch.tensor(parent.lattice.matrix, dtype=torch.float),
                torch.tensor(parent.atomic_numbers, dtype=torch.long),
                sigma,
                p,
            )
            augmentations.append(
                AugmentedStructure(
                    positions=new_positions,
                    cell=torch.tensor(parent.lattice.matrix, dtype=torch.float),
                    species=new_species,
                    sigma=sigma,
                    p=p,
                    family_id=family_id,
                    augmentation_type="cross",
                )
            )
        return augmentations
