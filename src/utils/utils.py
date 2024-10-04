from typing import Iterable

import numpy as np
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdchem import Mol
import ast

def is_mol(obj: object, true_on_none: bool = False) -> bool:
    """
    Returns True if the passed object is a rdkit.Mol object

    Parameters
    ----------
    obj: object
        some object to check
    true_on_none: bool, default = False
        if object is None return True
        this could be useful because rdkit will return `None` for Mols that are invalid (violate valance rules)

    Returns
    -------
    bool
    """
    if obj is None and true_on_none:
        return True
    return isinstance(obj, Mol)


def to_mol(smi: str or object) -> Mol or None:
    """
    Convert a mol (or some object containing a `smiles` attribute) into an RDKit.Mol object
    will return None if a passed object cannot be converted to mol (just like rdkit)

    Parameters
    ----------
    smi: str or object
        something to

    Returns
    -------
    rdkit.Chem.Mol
    """
    if isinstance(smi, Mol):
        return smi
    if isinstance(smi, str):
        return MolFromSmiles(smi)
    if hasattr(smi, 'smiles'):
        return MolFromSmiles(smi.smiles)
    return None


def to_mols(smis: Iterable[str or object]) -> list[Mol]:
    """
    Convert a list (or other iterable) into a list of rdkit Mols

    Parameters
    ----------
    smis: iterable
        a of SMILES (or object with `.smiles` attributes)

    Returns
    -------
    list[Mol]

    """
    return [to_mol(smi) for smi in smis]


def is_smi(obj: object) -> bool:
    """
    Returns `True` if the passed object is a valid SMILES string

    Parameters
    ----------
    obj: object
        object to check if is SMILES

    Returns
    -------
    bool

    """
    if not isinstance(obj, str):
        return False
    if MolFromSmiles(obj) is not None:
        return True
    return False


def to_smi(mol: Mol or object) -> str or None:
    """
    Convert a Mol (or object with a `.smiles` attribute) into a SMILES string
    Returns None when a passed object cannot be converted to SMILES

    Parameters
    ----------
    mol: Mol or object
        object to convert to SMILES

    Returns
    -------
    smi or None

    """
    if isinstance(mol, str):
        return mol
    if isinstance(mol, Mol):
        return MolToSmiles(mol)
    if hasattr(mol, 'smiles'):
        return mol.smiles
    return None


def to_smis(mols: Iterable[Mol or object]) -> list[str or None]:
    """
    Convert an iterable of Mols (or objects with `.smiles` attributes) into a list of rdkit Mols
    Will return None for any objects that cannot be converted into SMILES

    Parameters
    ----------
    mols: iterable
        iterable of objects to convert to SMILES

    Returns
    -------
    list[str]
    """
    return [to_smi(smi) for smi in mols]



def catch_boost_argument_error(e) -> bool:
    """
    This ugly code is to try and catch the Boost.Python.ArgumentError that rdkit throws when you pass an argument of
    the wrong type into the function
    Parameters
    ----------
    e : an Exception
        the Exception raised by the code

    Returns
    -------
    bool
        True if it is the Boost.Python.ArgumentError, False if it is not
    """
    if str(e).startswith("Python argument types"):
        return True
    else:
        return False


def to_list(obj):
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, str):
        return ast.literal_eval(obj)
    elif not hasattr(obj, "__iter__"):
        return [obj]
    else:
        return list(obj)
