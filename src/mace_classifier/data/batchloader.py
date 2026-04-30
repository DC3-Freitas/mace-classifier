from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch_geometric
from mace.data import AtomicData, Configuration
from mace.tools import AtomicNumberTable
from pymatgen.core import Structure
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torch_geometric.data import Batch

from mace_classifier.data import augmentations, loader
from mace_classifier.data.augmentations import AugmentedStructure

Z_TABLE = AtomicNumberTable([1, 2])

# Outside the ranges of shuffle
PURE_PERMUTATION = +1e9
MONO_SPECIES_SHUFFLE = -1e9


def make_atomic_data(
    augmented: AugmentedStructure,
    metadata: Dict[str, float | int],
    z_table: AtomicNumberTable,
    cutoff: float,
) -> AtomicData:
    """Convert AugmentedStructure into MACE AtomicData and attach metadata."""
    atomic_numbers = (augmented.species.to(torch.long)).cpu().numpy()
    config = Configuration(
        atomic_numbers=atomic_numbers,
        positions=augmented.positions.to(torch.float32).cpu().numpy(),
        cell=augmented.cell.to(torch.float32).cpu().numpy(),
        pbc=[True, True, True],
        properties={},  # we may want to rework this so our metadata is stored here
        property_weights={},
    )
    data = AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)

    data.bravais_label = torch.tensor(
        loader.BRAVAIS_LABELS[metadata["bravais_label"]], dtype=torch.long
    )
    # data.ordering_type_label = torch.tensor(
    #     metadata["ordering_type_label"], dtype=torch.long
    # )
    data.ordering_type_label = metadata[
        "ordering_type_label"
    ]  # should ask what we do this for
    data.sigma = torch.tensor(metadata["sigma"], dtype=torch.float32)
    data.shuffle_fraction = torch.tensor(
        metadata["shuffle_fraction"], dtype=torch.float32
    )
    data.family_id = torch.tensor(metadata["family_id"], dtype=torch.long)
    return data


class PrototypeDataset(Dataset):
    def __init__(
        self,
        cif_dir,
        manifest,
        sigma_levels,  # Alex: these thingsmust be unique
        shuffle_levels,
        d_nn_range,
        n_scales=5,
        r_cut=5.0,
        min_box_length=10.0,
    ):
        """
        Dataset of prototype structures with on-the-fly augmentations.

        Batch size = 64, 4 families per batch
        Each family has 16 augmentations:
        1. Parent
        2. Species permuted
        3. Monospecies projection
        4. 5x Structural augmentation
        5. 5x Chemical augmentation
        4. 3x cross augmentation

        Parameters
        ----------
        cif_dir: str
            Path to directory containing CIF files
        manifest: dict
            Dict mapping prototype_id to metadata. Each entry should have keys:
            - cif_file: str, filename of CIF in cif_dir
            - bravais_label: str, Bravais lattice label
            - ordering_type_label: str, ordering type label
        sigma_levels: list[float]
            List of sigma levels for structural augmentations
        shuffle_levels: list[float]
            List of shuffle levels for chemical augmentations
        d_nn_range: tuple(float, float)
            Range of nearest neighbor distances for scale augmentations
        n_scales: int
            Number of scale levels to generate between d_nn_range[0] and d_nn_range[1]
        r_cut: float
            Cutoff radius for graph construction
        min_box_length: float
            Minimum box length for supercell construction
        """

        self.cif_dir = cif_dir
        self.manifest = manifest
        self.sigma_levels = sigma_levels
        self.shuffle_levels = shuffle_levels
        self.d_nn_range = d_nn_range
        self.n_scales = n_scales
        self.r_cut = r_cut
        self.min_box_length = min_box_length

        self.n_augmentations = 16

        self.batch = 0
        self.epoch = 0

        self.parents = []
        self.prepare_parents()

    def prepare_parents(self):
        for pid, info in self.manifest.items():
            atoms = loader.load_prototype(Path(self.cif_dir) / Path(info["cif_file"]))
            supercell = loader.make_supercell(
                atoms, min_box_length=self.min_box_length, min_atoms=64
            )
            self.parents.append(
                {
                    "family_id": len(self.parents),
                    "prototype_id": pid,
                    "atoms": supercell,
                    "info": info,
                }
            )

    def __len__(self):
        return len(self.parents) * self.n_augmentations

    def set_epoch(self, epoch):
        """
        Set epoch for deterministic augmentations. Should be called at the start of each epoch.
        """
        self.epoch = epoch

    def set_batch(self, batch):
        """
        Set batch index for deterministic augmentations. Should be called at the start of each batch.
        """
        self.batch = batch

    def __getitem__(self, idx):
        """
        Index structure
        [
            0 = parent,
            1 = species permuted,
            2 = monospecies,
            3 = structural aug 1,
            4 = structural aug 2,
            5 = structural aug 3,
            6 = structural aug 4,
            7 = structural aug 5,
            8 = chemical aug 1,
            9 = chemical aug 2,
            10 = chemical aug 3,
            11 = chemical aug 4,
            12 = chemical aug 5,
            13 = cross aug 1,
            14 = cross aug 2,
            15 = cross aug 3
        ]
        """

        seed = self.batch + self.epoch * 1000
        torch.manual_seed(seed)  # within each batch, augmentations should be identical

        family = idx // self.n_augmentations
        aug_idx = idx % self.n_augmentations
        parent_info = self.parents[family]
        parent_atoms = parent_info["atoms"]
        if aug_idx == 0:
            return make_atomic_data(
                AugmentedStructure(
                    positions=torch.tensor(parent_atoms.cart_coords, dtype=torch.float),
                    cell=torch.tensor(parent_atoms.lattice.matrix, dtype=torch.float),
                    species=torch.tensor(parent_atoms.atomic_numbers, dtype=torch.long),
                ),
                metadata={
                    "bravais_label": parent_info["info"]["bravais_label"],
                    "ordering_type_label": parent_info["info"]["ordering_type_label"],
                    "sigma": 0.0,
                    "shuffle_fraction": 0.0,
                    "family_id": parent_info["family_id"],
                },
                z_table=Z_TABLE,
                cutoff=self.r_cut,
            )
        elif aug_idx == 1:
            new_species = augmentations.apply_species_permutation(
                torch.tensor(parent_atoms.atomic_numbers, dtype=torch.long),
            )
            return make_atomic_data(
                AugmentedStructure(
                    positions=torch.tensor(parent_atoms.cart_coords, dtype=torch.float),
                    cell=torch.tensor(parent_atoms.lattice.matrix, dtype=torch.float),
                    species=new_species,
                ),
                metadata={
                    "bravais_label": parent_info["info"]["bravais_label"],
                    "ordering_type_label": parent_info["info"]["ordering_type_label"],
                    "sigma": 0.0,
                    "shuffle_fraction": PURE_PERMUTATION,  # Alex: special case
                    "family_id": parent_info["family_id"],
                },
                z_table=Z_TABLE,
                cutoff=self.r_cut,
            )
        elif aug_idx == 2:
            new_species = augmentations.apply_mono_species(
                torch.tensor(parent_atoms.atomic_numbers, dtype=torch.long),
            )
            return make_atomic_data(
                AugmentedStructure(
                    positions=torch.tensor(parent_atoms.cart_coords, dtype=torch.float),
                    cell=torch.tensor(parent_atoms.lattice.matrix, dtype=torch.float),
                    species=new_species,
                ),
                metadata={
                    "bravais_label": parent_info["info"]["bravais_label"],
                    "ordering_type_label": parent_info["info"]["ordering_type_label"],
                    "sigma": 0.0,
                    "shuffle_fraction": MONO_SPECIES_SHUFFLE,  # Alex: should not count
                    "family_id": parent_info["family_id"],
                },
                z_table=Z_TABLE,
                cutoff=self.r_cut,
            )
        elif 3 <= aug_idx < 8:
            sigma = self.sigma_levels[aug_idx - 3]
            new_positions = augmentations.apply_gaussian_noise(
                torch.tensor(parent_atoms.cart_coords, dtype=torch.float),
                torch.tensor(parent_atoms.lattice.matrix, dtype=torch.float),
                sigma=sigma,
                seed=seed,
            )
            return make_atomic_data(
                AugmentedStructure(
                    positions=new_positions,
                    cell=torch.tensor(parent_atoms.lattice.matrix, dtype=torch.float),
                    species=torch.tensor(parent_atoms.atomic_numbers, dtype=torch.long),
                    sigma=sigma,
                ),
                metadata={
                    "bravais_label": parent_info["info"]["bravais_label"],
                    "ordering_type_label": parent_info["info"]["ordering_type_label"],
                    "sigma": sigma,
                    "shuffle_fraction": 0.0,
                    "family_id": parent_info["family_id"],
                },
                z_table=Z_TABLE,
                cutoff=self.r_cut,
            )
        elif 8 <= aug_idx < 13:
            p = self.shuffle_levels[aug_idx - 8]
            new_species = augmentations.apply_species_shuffle(
                torch.tensor(parent_atoms.atomic_numbers, dtype=torch.long),
                fraction=p,
                seed=seed,
            )
            return make_atomic_data(
                AugmentedStructure(
                    positions=torch.tensor(parent_atoms.cart_coords, dtype=torch.float),
                    cell=torch.tensor(parent_atoms.lattice.matrix, dtype=torch.float),
                    species=new_species,
                    p=p,
                ),
                metadata={
                    "bravais_label": parent_info["info"]["bravais_label"],
                    "ordering_type_label": parent_info["info"]["ordering_type_label"],
                    "sigma": 0.0,
                    "shuffle_fraction": p,
                    "family_id": parent_info["family_id"],
                },
                z_table=Z_TABLE,
                cutoff=self.r_cut,
            )
        else:
            sigma = torch.tensor(
                self.sigma_levels[torch.randint(len(self.sigma_levels), (1,))]
            )
            p = torch.tensor(
                self.shuffle_levels[torch.randint(len(self.shuffle_levels), (1,))]
            )
            new_positions, new_species = augmentations.apply_cross_augmentation(
                torch.tensor(parent_atoms.cart_coords, dtype=torch.float),
                torch.tensor(parent_atoms.lattice.matrix, dtype=torch.float),
                torch.tensor(parent_atoms.atomic_numbers, dtype=torch.long),
                sigma=sigma,
                shuffle_fraction=p,
                seed=seed,
            )
            return make_atomic_data(
                AugmentedStructure(
                    positions=new_positions,
                    cell=torch.tensor(parent_atoms.lattice.matrix, dtype=torch.float),
                    species=new_species,
                    sigma=sigma,
                    p=p,
                ),
                metadata={
                    "bravais_label": parent_info["info"]["bravais_label"],
                    "ordering_type_label": parent_info["info"]["ordering_type_label"],
                    "sigma": sigma,
                    "shuffle_fraction": p,
                    "family_id": parent_info["family_id"],
                },
                z_table=Z_TABLE,
                cutoff=self.r_cut,
            )


class FamilySampler(BatchSampler):
    def __init__(self, dataset, batch_size, family_size=16):
        """
        BatchSampler that samples batches of families. Keeps families together in the same batch, but
        shuffles the order of families and the order of samples within each family.

        Parameters
        ----------
        dataset: PrototypeDataset
            The dataset to sample from
        batch_size: int
            The batch size to sample
        family_size: int
            The number of augmentations per family (default 16)
        """
        self.dataset_size = len(dataset)
        self.batch_size = batch_size
        self.family_size = family_size
        self.n_families = self.dataset_size // self.family_size
        self.families_per_batch = self.batch_size // self.family_size

    def __iter__(self):
        indices = torch.randperm(self.n_families).tolist()
        for i in range(0, self.n_families, self.families_per_batch):
            batch_families = indices[i : i + self.families_per_batch]
            batch_indices = []
            for family in batch_families:
                family_indices = list(
                    range(family * self.family_size, (family + 1) * self.family_size)
                )
                family_indices = torch.tensor(family_indices)[
                    torch.randperm(self.family_size)
                ].tolist()
                batch_indices.extend(family_indices)
            yield batch_indices

    def __len__(self):
        return len(self.dataset) // self.batch_size


def atomic_data_collate_fn(batch):
    return batch  # the model will need to take in list of AtomicData
