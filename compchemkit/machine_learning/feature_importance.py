from collections import defaultdict
import numpy as np
import io
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import Geometry
from PIL import Image
from typing import *
from compchemkit.machine_learning.fingerprints import AtomEnvironment


def shap2atomweight(mol, fingerprint, shap_mat):
    bit_atom_env_dict = fingerprint.bit2atom_mapping(mol)
    atom_weight_dict = assign_prediction_importance(bit_atom_env_dict, shap_mat)
    atom_weight_list = [atom_weight_dict[a_idx] if a_idx in atom_weight_dict else 0 for a_idx
                        in range(mol.GetNumAtoms())]
    return atom_weight_list


def assign_prediction_importance(bit_dict: Dict[int, List[AtomEnvironment]], weights: np.ndarray):
    atom_contribution = defaultdict(lambda: 0)
    for bit, atom_env_list in bit_dict.items():  # type: int, List[AtomEnvironment]
        n_machtes = len(atom_env_list)
        for atom_set in atom_env_list:
            for atom in atom_set.environment_atoms:
                atom_contribution[atom] += weights[bit] / (len(atom_set.environment_atoms) * n_machtes)
    assert np.isclose(sum(weights), sum([x for x in atom_contribution.values()])), (
        sum(weights), sum([x for x in atom_contribution.values()]))
    return atom_contribution


def GetSimilarityMapFromWeights(mol: Chem.Mol,
                                weights: Union[np.ndarray, List[float], Tuple[float]],
                                draw2d: Draw.MolDraw2DCairo,
                                sigma=None,
                                sigma_f=0.3,
                                contourLines=10,
                                contour_params: Draw.ContourParams = None):
    """ Stolen... uhm... copied from Chem.Draw.SimilarityMaps
    Generates the similarity map for a molecule given the atomic weights.
    Parameters:
      mol -- the molecule of interest
      colorMap -- the matplotlib color map scheme, default is custom PiWG color map
      scale -- the scaling: scale < 0 -> the absolute maximum weight is used as maximum scale
                            scale = double -> this is the maximum scale
      size -- the size of the figure
      sigma -- the sigma for the Gaussians
      coordScale -- scaling factor for the coordinates
      step -- the step for calcAtomGaussian
      colors -- color of the contour lines
      contourLines -- if integer number N: N contour lines are drawn
                      if list(numbers): contour lines at these numbers are drawn
      alpha -- the alpha blending value for the contour lines
      kwargs -- additional arguments for drawing
    """
    if mol.GetNumAtoms() < 2:
        raise ValueError("too few atoms")
    mol = Draw.rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
    if not mol.GetNumConformers():
        Draw.rdDepictor.Compute2DCoords(mol)
    if sigma is None:
        if mol.GetNumBonds() > 0:
            bond = mol.GetBondWithIdx(0)
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            sigma = sigma_f * (
                        mol.GetConformer().GetAtomPosition(idx1) - mol.GetConformer().GetAtomPosition(idx2)).Length()
        else:
            sigma = sigma_f * (mol.GetConformer().GetAtomPosition(0) - mol.GetConformer().GetAtomPosition(1)).Length()
        sigma = round(sigma, 2)
    sigmas = [sigma] * mol.GetNumAtoms()
    locs = []
    for i in range(mol.GetNumAtoms()):
        p = mol.GetConformer().GetAtomPosition(i)
        locs.append(Geometry.Point2D(p.x, p.y))
    draw2d.DrawMolecule(mol)
    draw2d.ClearDrawing()
    if not contour_params:
        contour_params = Draw.ContourParams()
        contour_params.fillGrid = True
        contour_params.gridResolution = 0.1
        contour_params.extraGridPadding = 0.5
    Draw.ContourAndDrawGaussians(draw2d, locs, weights, sigmas, nContours=contourLines, params=contour_params)
    draw2d.drawOptions().clearBackground = False
    draw2d.DrawMolecule(mol)
    return draw2d


def rdkit_gaussplot(mol, weights, n_contourLines=5, color_tuple=None, ):
    d = Draw.MolDraw2DCairo(600, 600)
    # Coloring atoms of element 0 to 100 black
    d.drawOptions().updateAtomPalette({i: (0, 0, 0, 1) for i in range(100)})
    cps = Draw.ContourParams()
    cps.fillGrid = True
    cps.gridResolution = 0.02
    cps.extraGridPadding = 1.2
    coolwarm = ((0.017, 0.50, 0.850, 0.5),
                (1, 1, 1, 0.5),
                (1, 0.25, 0.0, 0.5)
                )

    if color_tuple is None:
        color_tuple = coolwarm

    cps.setColourMap(color_tuple)

    d = GetSimilarityMapFromWeights(mol, weights, contourLines=n_contourLines,
                                    draw2d=d, contour_params=cps, sigma_f=0.4)

    d.FinishDrawing()
    return d


def show_png(data):
    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img


def save_png(data, path, **kwargs):
    bio = io.BytesIO(data)
    img = Image.open(bio)
    img.save(path, **kwargs)
