from typing import Union, List, Callable, Optional
from functools import partial
import abc

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin

from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, rdMolDescriptors, RDKFingerprint
from rdkit.Chem.AtomPairs import Pairs
from rdkit import Chem

from src.utils.utils import to_mol, catch_boost_argument_error

from rdkit.rdBase import BlockLogs

block = BlockLogs()


def _wrap_handle_none(fp_func: Callable, *args, fail_size: Optional[int] = None, **kwargs) -> List:
    """
    wraps an FP function and handles RDKit chemical exceptions by returning a list of NaN values.

    Parameters
    ----------
    fp_func (Callable):
        The function to be wrapped.
    *args:
        Variable length argument list to be passed to the function.
    fail_size (int or None, optional):
        The size of the list to be returned in case of failure.
        If None, the fail_size is determined by calling the function with a default argument.
        Defaults to None.
    **kwargs:
        Arbitrary keyword arguments to be passed to the function.

    Returns
    -------
    list:
        A list of NaN values with length equal to fail_size.

    Notes:
    -----
    If not `fail_size` is passed, will try and assume it by calling the FP function on "CCC" to get the FP length,
    This can cause major overhead if lost of SMILES fail inside RDKit (what this wrapper is built to catch),
    so if you expect to see high failure rates preseting the fail length will minimize this overhead

    The `FPFunc` Class has a `dimension` attribute that holds the FP length, thus any defined FPFunc should not
    suffer from this overhead. Any newly added FPFunc should follow this convention as well

    Raises
    ------
    Any Exception:
        If the exception thrown is not a boost C++ exception, will still raise that exception.

    Examples
    ________
    >>> _wrap_handle_none(AllChem.GetMorganFingerprintAsBitVect, Chem.MolFromSmiles("CCCZ"), 2)
    [nan, nan]

    >>> _wrap_handle_none(AllChem.GetMorganFingerprintAsBitVect, Chem.MolFromSmiles("CCCZ"), fail_size=3)
    [nan, nan, nan]
    """
    assert isinstance(fp_func, Callable), "fp_func must be a callable"
    try:
        return list(fp_func(*args, **kwargs))
    except Exception as e:  # throws boost C++ exception, which python cannot catch
        if catch_boost_argument_error(e):
            if fail_size is None:
                # attempt to get fail_size from the func if it is not passed
                fail_size = len(list(fp_func(AllChem.MolFromSmiles("CCC"))))
            return [np.nan] * fail_size
        else:
            raise


class BaseFPFunc(TransformerMixin, abc.ABC):
    """
    Base class for all FP functions used in any AIRCHECK pipeline

    Attributes
    ----------
    func : functools.partial object
        The callable FP function instance as a partial with static setting arguments (e.g., 'radius') pre-set
    binary : bool
        Whether the FP function returns binary fingerprints
    dimension : int
        the dimensionality of the fingerprints that will be generated

    Notes
    -----
    When declaring a child of the `BaseFPFunc` class, the `_func`, `_dimension` and `_binary` attributes must be set
    during instantiation of the child.
    FP Funcs operate on rdkit.ROMol objects, not smiles and will fail if SMILES are passed
    """
    def __init__(self, use_tqdm: bool = False):
        self.use_tqdm = use_tqdm
        self.func: Callable = lambda: None
        self.binary: bool = False
        self.dimension: int = -1

    def __call__(self, smis, *args, use_tqdm: bool = False, **kwargs) -> npt.NDArray[np.int32 or np.float64]:
        return np.array(
            [
                _wrap_handle_none(self.func, to_mol(c), fail_size=self.dimension)
                for c in tqdm(np.atleast_1d(smis), disable=not use_tqdm)
            ]
        )

    def generate_fps(
        self, smis: Union[str, Chem.rdchem.Mol, List[Union[str, Chem.rdchem.Mol]]]
    ) -> npt.NDArray[np.int32 or np.float64]:
        """
        Generate Fingerprints for a set of smiles
        Parameters
        ----------
        smis : str, rdkit Mol or list of rdkit Mol or str
            the SMILES or Mol objects (or multiple SMILES/Mol objects) you want to generate a fingerprint(s) for

        Returns
        -------
        ndarray
            an array of size (M, d), where M is number of Mols passes and d is the dimension of fingerprint

        Notes
        -----
        The passed list can be a mix of SMILES and Mol objects.
        If the SMILES are invalid or the Mol object(s) are None, then that molecules row of the output fingerprint
         array will be `np.nan` (e.i., the fingerprint for that molecule will be 1-d array of `np.nan` of dimension d)
        This function just wraps the __call__ method of the class

        """
        return self.__call__(smis)

    def fit(self, X=None, y=None) -> None:
        pass

    def transform(self, X, y=None):
        return self.generate_fps(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)

    def is_binary(self) -> bool:
        """
        Determines if the FP function is binary

        Returns
        -------
        bool:
            True if the FP function is binary, otherwise False

        Notes
        -----
        This function just returns the `binary` attribute set during instantiation
        """
        return self.binary


class ExtendedConnectivityFP(BaseFPFunc, abc.ABC):
    def __init__(self, radius: int, nBits: int, useFeatures: bool):
        self.radius = radius
        self.nBits = nBits
        self.useFeatures = useFeatures
        super().__init__()


class HitGenFP(abc.ABC):
    company = "hitgen"


class HitGenECFP4(ExtendedConnectivityFP, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating ECFP4 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": False})
        self.func = partial(AllChem.GetHashedMorganFingerprint, radius=self.radius,
                            nBits=self.nBits, useFeatures=self.useFeatures)
        self.dimension = 2048


class HitGenECFP6(ExtendedConnectivityFP, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating ECFP6 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": False})
        self.func = partial(AllChem.GetHashedMorganFingerprint, radius=self.radius,
                            nBits=self.nBits, useFeatures=self.useFeatures)
        self.dimension = 2048


class HitGenFCFP4(ExtendedConnectivityFP, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating FCFP4 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": True})
        self.func = partial(AllChem.GetHashedMorganFingerprint, radius=self.radius,
                            nBits=self.nBits, useFeatures=self.useFeatures)
        self.dimension = 2048


class HitGenFCFP6(ExtendedConnectivityFP, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating FCFP6 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": True})
        self.func = partial(AllChem.GetHashedMorganFingerprint, radius=self.radius,
                            nBits=self.nBits, useFeatures=self.useFeatures)
        self.dimension = 2048


class HitGenBinaryECFP4(ExtendedConnectivityFP, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating BinaryECFP4 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore

    This FP is not directly given by HitGen,
    but can be calculated by just 'binarizing' the hashed FP provided on AIRCHECK
    """
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": False})
        self.func = partial(AllChem.GetHashedMorganFingerprint, radius=self.radius,
                            nBits=self.nBits, useFeatures=self.useFeatures)
        self.dimension = 2048
        self.binary = True


class HitGenBinaryECFP6(ExtendedConnectivityFP, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating BinaryECFP6 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore

    This FP is not directly given by HitGen,
    but can be calculated by just 'binarizing' the hashed FP provided on AIRCHECK
    """
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": False})
        self.func = partial(AllChem.GetHashedMorganFingerprint, radius=self.radius,
                            nBits=self.nBits, useFeatures=self.useFeatures)
        self.dimension = 2048
        self.binary = True


class HitGenBinaryFCFP4(ExtendedConnectivityFP, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating BinaryFCFP4 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore

    This FP is not directly given by HitGen,
    but can be calculated by just 'binarizing' the hashed FP provided on AIRCHECK
    """
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": True})
        self.func = partial(AllChem.GetHashedMorganFingerprint, radius=self.radius,
                            nBits=self.nBits, useFeatures=self.useFeatures)
        self.dimension = 2048
        self.binary = True


class HitGenBinaryFCFP6(ExtendedConnectivityFP, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating BinaryFCFP6 fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore

    This FP is not directly given by HitGen,
    but can be calculated by just 'binarizing' the hashed FP provided on AIRCHECK
    """
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": True})
        self.func = partial(AllChem.GetHashedMorganFingerprint, radius=self.radius,
                            nBits=self.nBits, useFeatures=self.useFeatures)
        self.dimension = 2048
        self.binary = True


class HitGenMACCS(BaseFPFunc, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating MACCS fingerprints

    Notes
    -----
    Unlike other HitGen FPs, MACCS is only generated in a binary fashion by HitGen, thus no hashed/count version exists
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__()
        self.func = partial(rdMolDescriptors.GetMACCSKeysFingerprint)
        self.binary = True
        self.dimension = 167


class HitGenRDK(BaseFPFunc, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating RDK fingerprints

    Notes
    -----
    Unlike other HitGen FPs, RDK is only generated in a binary fashion by HitGen, thus no hashed/count version exists
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__()
        self.fpSize = 2048
        self.func = partial(RDKFingerprint, fpSize=self.fpSize)
        self.binary = True
        self.dimension = 2048


class HitGenAvalon(BaseFPFunc, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating Avalon fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__()
        self.nBits = 2048
        self.func = partial(pyAvalonTools.GetAvalonCountFP, nBits=self.nBits)
        self.dimension = 2048


class HitGenBinaryAvalon(BaseFPFunc, HitGenFP, BaseEstimator):
    """
    The FP calculation used to match the binary Avalon fingerprints generated from HitGen's Avalon fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore

    This FP is not directly given by HitGen,
    but can be calculated by just 'binarizing' the hashed FP provided on AIRCHECK
    """
    def __init__(self):
        super().__init__()
        self.nBits = 2048
        self.func = partial(pyAvalonTools.GetAvalonFP, nBits=self.nBits)
        self.binary = True
        self.dimension = 2048


class HitGenAtomPair(BaseFPFunc, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating AtomPair fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__()
        self.nBits = 2048
        self.func = partial(rdMolDescriptors.GetHashedAtomPairFingerprint, nBits=self.nBits)
        self.dimension = 2048


class HitGenBinaryAtomPair(BaseFPFunc, HitGenFP, BaseEstimator):
    """
    The FP calculation used to match the binary AtomPair fingerprints generated from HitGen's AtomPair fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore

    This FP is not directly given by HitGen,
    but can be calculated by just 'binarizing' the hashed FP provided on AIRCHECK
    """
    def __init__(self):
        super().__init__()
        self.nBits = 2048
        self.func = partial(Pairs.GetAtomPairFingerprintAsBitVect, nBits=self.nBits)
        self.binary = True
        self.dimension = 2048


class HitGenTopTor(BaseFPFunc, HitGenFP, BaseEstimator):
    """
    The FP calculation used by HitGen when generating Topological Torsion (TopTor) fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__()
        self.nBits = 2048
        self.func = partial(AllChem.GetHashedTopologicalTorsionFingerprint, nBits=self.nBits)
        self.dimension = 2048


class HitGenBinaryTopTor(BaseFPFunc, HitGenFP, BaseEstimator):
    """
    The FP calculation used to match the binary TopTor fingerprints generated from HitGen's TopTor fingerprints

    Notes
    -----
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore

    This FP is not directly given by HitGen,
    but can be calculated by just 'binarizing' the hashed FP provided on AIRCHECK
    """
    def __init__(self):
        super().__init__()
        self.nBits = 2048
        self.func = partial(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect, nBits=self.nBits)
        self.binary = True
        self.dimension = 2048


class XChemFP(abc.ABC):
    company = "xchem"


class XChemBinaryECFP4(ExtendedConnectivityFP, XChemFP, BaseEstimator):
    """
    The FP calculation used by XChem when generating BinaryECFP4 fingerprints

    Notes
    -----
    XChem only provides binary fingerprints

    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": False})
        self.func = partial(AllChem.GetHashedMorganFingerprint, radius=self.radius,
                            nBits=self.nBits, useFeatures=self.useFeatures)
        self.dimension = 2048
        self.binary = True


class XChemBinaryECFP6(ExtendedConnectivityFP, XChemFP, BaseEstimator):
    """
    The FP calculation used by XChem when generating BinaryECFP6 fingerprints

    Notes
    -----
    XChem only provides binary fingerprints

    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": False})
        self.func = partial(AllChem.GetHashedMorganFingerprint, radius=self.radius,
                            nBits=self.nBits, useFeatures=self.useFeatures)
        self.dimension = 2048
        self.binary = True


class XChemBinaryFCFP4(ExtendedConnectivityFP, XChemFP, BaseEstimator):
    """
    The FP calculation used by XChem when generating BinaryFCFP4 fingerprints

    Notes
    -----
    XChem only provides binary fingerprints

    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__(**{"radius": 2, "nBits": 2048, "useFeatures": True})
        self.func = partial(AllChem.GetHashedMorganFingerprint, radius=self.radius,
                            nBits=self.nBits, useFeatures=self.useFeatures)
        self.dimension = 2048
        self.binary = True


class XChemBinaryFCFP6(ExtendedConnectivityFP, XChemFP, BaseEstimator):
    """
    The FP calculation used by XChem when generating BinaryFCFP6 fingerprints

    Notes
    -----
    XChem only provides binary fingerprints

    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__(**{"radius": 3, "nBits": 2048, "useFeatures": True})
        self.func = partial(AllChem.GetHashedMorganFingerprint, radius=self.radius,
                            nBits=self.nBits, useFeatures=self.useFeatures)
        self.dimension = 2048
        self.binary = True


class XChemMACCS(BaseFPFunc, XChemFP, BaseEstimator):
    """
    The FP calculation used by XChem when generating MACCS fingerprints

    Notes
    -----
    Unlike other FPs, MACCS is only generated in a binary fashion by XChem, thus no hashed/count version exists
    All settings and attributes are preset during the instantiation of the object.
    Tweaks to FP settings should not be made, as the FP function will not match HitGen anymore
    """
    def __init__(self):
        super().__init__()
        self.func = partial(rdMolDescriptors.GetMACCSKeysFingerprint)
        self.binary = True
        self.dimension = 167


def get_fp_func(class_name: str, **kwargs) -> BaseFPFunc:
    _class = globals().get(class_name)
    if _class:
        return _class(**kwargs)
    raise ValueError(f"cannot find FPFunc {class_name}")
