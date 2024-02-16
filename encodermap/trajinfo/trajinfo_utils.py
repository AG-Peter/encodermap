# -*- coding: utf-8 -*-
# encodermap/trajinfo/trajinfo_utils.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
#
# Authors:
# Kevin Sawade
#
# Encodermap is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
# This package is distributed in the hope that it will be useful to other
# researches. IT DOES NOT COME WITH ANY WARRANTY WHATSOEVER; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# See <http://www.gnu.org/licenses/>.
################################################################################
"""Util functions for the `TrajEnsemble` and `SingleTraj` classes.

"""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import functools
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce
from operator import xor
from pathlib import Path

# Third Party Imports
import numpy as np
import yaml
from optional_imports import _optional_import

# Local Folder Imports
from ..misc.misc import FEATURE_NAMES
from ..misc.xarray import construct_xarray_from_numpy


##############################################################################
# Optional Imports
##############################################################################


xr = _optional_import("xarray")
md = _optional_import("mdtraj")
_atom_sequence = _optional_import("mdtraj", "geometry.dihedral._atom_sequence")
parse_offsets = _optional_import("mdtraj", "geometry.dihedral.parse_offsets")
_construct_atom_dict = _optional_import(
    "mdtraj", "geometry.dihedral._construct_atom_dict"
)
_strip_offsets = _optional_import("mdtraj", "geometry.dihedral._strip_offsets")


################################################################################
# Typing
################################################################################


# Standard Library Imports
from collections.abc import Generator, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

# Local Folder Imports
from .._typing import CustomAAsDict


if TYPE_CHECKING:  # pragma: no cover
    # Third Party Imports
    import mdtraj as md
    import xarray as xr
    from pyemma.coordinates.data.featurization._base import Feature

    # Local Folder Imports
    from .._typing import AnyFeature
    from ..loading.dask_featurizer import DaskFeaturizer
    from ..loading.featurizer import Featurizer
    from .info_single import SingleTraj, TrajEnsemble

    SingleTrajFeatureType = Union[
        str,
        Path,
        np.ndarray,
        Feature,
        xr.Dataset,
        xr.DataArray,
        DaskFeaturizer,
        Literal["all"],
        AnyFeature,
    ]
    TrajEnsembleFeatureType = Union[
        Sequence[str],
        Sequence[Path],
        Sequence[np.ndarray],
        xr.Dataset,
        DaskFeaturizer,
        Literal["all"],
        AnyFeature,
    ]


################################################################################
# Globals
################################################################################


CAN_BE_FEATURE_NAME = list(FEATURE_NAMES.keys()) + list(FEATURE_NAMES.values())


# fmt: off
_AMINO_ACID_CODES =  {'ACE': None, 'NME':  None, '00C': 'C', '01W':  'X', '02K':
'A', '02L':  'N', '03Y': 'C',  '07O': 'C', '08P':  'C', '0A0': 'D',  '0A1': 'Y',
'0A2': 'K', '0A8':  'C', '0AA': 'V', '0AB': 'V', '0AC':  'G', '0AF': 'W', '0AG':
'L', '0AH':  'S', '0AK': 'D',  '0BN': 'F', '0CS':  'A', '0E5': 'T',  '0EA': 'Y',
'0FL': 'A', '0NC':  'A', '0WZ': 'Y', '0Y8': 'P', '143':  'C', '193': 'X', '1OP':
'Y', '1PA':  'F', '1PI': 'A',  '1TQ': 'W', '1TY':  'Y', '1X6': 'S',  '200': 'F',
'23F': 'F', '23S':  'X', '26B': 'T', '2AD': 'X', '2AG':  'A', '2AO': 'X', '2AS':
'X', '2CO':  'C', '2DO': 'X',  '2FM': 'M', '2HF':  'H', '2KK': 'K',  '2KP': 'K',
'2LU': 'L', '2ML':  'L', '2MR': 'R', '2MT': 'P', '2OR':  'R', '2PI': 'X', '2QZ':
'T', '2R3':  'Y', '2SI': 'X',  '2TL': 'T', '2TY':  'Y', '2VA': 'V',  '2XA': 'C',
'32S': 'X', '32T':  'X', '33X': 'A', '3AH': 'H', '3AR':  'X', '3CF': 'F', '3GA':
'A', '3MD':  'D', '3NF': 'Y',  '3QN': 'K', '3TY':  'X', '3XH': 'G',  '4BF': 'Y',
'4CF': 'F', '4CY':  'M', '4DP': 'W', '4FB': 'P', '4FW':  'W', '4HT': 'W', '4IN':
'W', '4MM':  'X', '4PH': 'F',  '4U7': 'A', '56A':  'H', '5AB': 'A',  '5CS': 'C',
'5CW': 'W', '5HP':  'E', '6CL': 'K', '6CW': 'W', '6GL':  'A', '6HN': 'K', '7JA':
'I', '9NE':  'E', '9NF': 'F',  '9NR': 'R', '9NV':  'V', 'A5N': 'N',  'A66': 'X',
'AA3': 'A', 'AA4':  'A', 'AAR': 'R', 'AB7': 'X', 'ABA':  'A', 'ACB': 'D', 'ACL':
'R', 'ADD':  'X', 'AEA': 'X',  'AEI': 'D', 'AFA':  'N', 'AGM': 'R',  'AGT': 'C',
'AHB': 'N', 'AHH':  'X', 'AHO': 'A', 'AHP': 'A', 'AHS':  'X', 'AHT': 'X', 'AIB':
'A', 'AKL':  'D', 'AKZ': 'D',  'ALA': 'A', 'ALC':  'A', 'ALM': 'A',  'ALN': 'A',
'ALO': 'T', 'ALS':  'A', 'ALT': 'A', 'ALV': 'A', 'ALY':  'K', 'AN8': 'A', 'APE':
'X', 'APH':  'A', 'API': 'K',  'APK': 'K', 'APM':  'X', 'APP': 'X',  'AR2': 'R',
'AR4': 'E', 'AR7':  'R', 'ARG': 'R', 'ARM': 'R', 'ARO':  'R', 'ARV': 'X', 'AS2':
'D', 'AS9':  'X', 'ASA': 'D',  'ASB': 'D', 'ASI':  'D', 'ASK': 'D',  'ASL': 'D',
'ASM': 'X', 'ASN':  'N', 'ASP': 'D', 'ASQ': 'D', 'ASX':  'B', 'AVN': 'X', 'AYA':
'A', 'AZK':  'K', 'AZS': 'S',  'AZY': 'Y', 'B1F':  'F', 'B2A': 'A',  'B2F': 'F',
'B2I': 'I', 'B2V':  'V', 'B3A': 'A', 'B3D': 'D', 'B3E':  'E', 'B3K': 'K', 'B3L':
'X', 'B3M':  'X', 'B3Q': 'X',  'B3S': 'S', 'B3T':  'X', 'B3U': 'H',  'B3X': 'N',
'B3Y': 'Y', 'BB6':  'C', 'BB7': 'C', 'BB8': 'F', 'BB9':  'C', 'BBC': 'C', 'BCS':
'C', 'BE2':  'X', 'BFD': 'D',  'BG1': 'S', 'BH2':  'D', 'BHD': 'D',  'BIF': 'F',
'BIL': 'X', 'BIU':  'I', 'BJH': 'X', 'BL2': 'L', 'BLE':  'L', 'BLY': 'K', 'BMT':
'T', 'BNN':  'F', 'BNO': 'X',  'BOR': 'R', 'BPE':  'C', 'BSE': 'S',  'BTA': 'L',
'BTC': 'C', 'BTR':  'W', 'BUC': 'C', 'BUG': 'V', 'C1X':  'K', 'C22': 'A', 'C3Y':
'C', 'C4R':  'C', 'C5C': 'C',  'C66': 'X', 'C6C':  'C', 'CAF': 'C',  'CAL': 'X',
'CAS': 'C', 'CAV':  'X', 'CAY': 'C', 'CCL': 'K', 'CCS':  'C', 'CDE': 'X', 'CDV':
'X', 'CEA':  'C', 'CGA': 'E',  'CGU': 'E', 'CHF':  'X', 'CHG': 'X',  'CHP': 'G',
'CHS': 'X', 'CIR':  'R', 'CLE': 'L', 'CLG': 'K', 'CLH':  'K', 'CME': 'C', 'CMH':
'C', 'CML':  'C', 'CMT': 'C',  'CPC': 'X', 'CPI':  'X', 'CR5': 'G',  'CS0': 'C',
'CS1': 'C', 'CS3':  'C', 'CS4': 'C', 'CSA': 'C', 'CSB':  'C', 'CSD': 'C', 'CSE':
'C', 'CSJ':  'C', 'CSO': 'C',  'CSP': 'C', 'CSR':  'C', 'CSS': 'C',  'CSU': 'C',
'CSW': 'C', 'CSX':  'C', 'CSZ': 'C', 'CTE': 'W', 'CTH':  'T', 'CUC': 'X', 'CWR':
'S', 'CXM':  'M', 'CY0': 'C',  'CY1': 'C', 'CY3':  'C', 'CY4': 'C',  'CYA': 'C',
'CYD': 'C', 'CYF':  'C', 'CYG': 'C', 'CYJ': 'K', 'CYM':  'C', 'CYQ': 'C', 'CYR':
'C', 'CYS':  'C', 'CZ2': 'C',  'CZZ': 'C', 'D11':  'T', 'D3P': 'G',  'D4P': 'X',
'DA2': 'X', 'DAB':  'A', 'DAH': 'F', 'DAL': 'A', 'DAR':  'R', 'DAS': 'D', 'DBB':
'T', 'DBS':  'S', 'DBU': 'T',  'DBY': 'Y', 'DBZ':  'A', 'DC2': 'C',  'DCL': 'X',
'DCY': 'C', 'DDE':  'H', 'DFI': 'X', 'DFO': 'X', 'DGH':  'G', 'DGL': 'E', 'DGN':
'Q', 'DHA':  'S', 'DHI': 'H',  'DHL': 'X', 'DHN':  'V', 'DHP': 'X',  'DHV': 'V',
'DI7': 'Y', 'DIL':  'I', 'DIR': 'R', 'DIV': 'V', 'DLE':  'L', 'DLS': 'K', 'DLY':
'K', 'DM0':  'K', 'DMH': 'N',  'DMK': 'D', 'DMT':  'X', 'DNE': 'L',  'DNL': 'K',
'DNP': 'A', 'DNS':  'K', 'DOA': 'X', 'DOH': 'D', 'DON':  'L', 'DPL': 'P', 'DPN':
'F', 'DPP':  'A', 'DPQ': 'Y',  'DPR': 'P', 'DSE':  'S', 'DSG': 'N',  'DSN': 'S',
'DSP': 'D', 'DTH':  'T', 'DTR': 'W', 'DTY': 'Y', 'DVA':  'V', 'DYS': 'C', 'ECC':
'Q', 'EFC':  'C', 'EHP': 'F',  'ESB': 'Y', 'ESC':  'M', 'EXY': 'L',  'EYS': 'X',
'F2F': 'F', 'FAK':  'K', 'FB5': 'A', 'FB6': 'A', 'FCL':  'F', 'FGA': 'E', 'FGL':
'G', 'FGP':  'S', 'FH7': 'K',  'FHL': 'K', 'FHO':  'K', 'FLA': 'A',  'FLE': 'L',
'FLT': 'Y', 'FME':  'M', 'FOE': 'C', 'FP9': 'P', 'FRD':  'X', 'FT6': 'W', 'FTR':
'W', 'FTY':  'Y', 'FVA': 'V',  'FZN': 'K', 'GAU':  'E', 'GCM': 'X',  'GFT': 'S',
'GGL': 'E', 'GHG':  'Q', 'GHP': 'G', 'GL3': 'G', 'GLH':  'Q', 'GLJ': 'E', 'GLK':
'E', 'GLM':  'X', 'GLN': 'Q',  'GLQ': 'E', 'GLU':  'E', 'GLX': 'Z',  'GLY': 'G',
'GLZ': 'G', 'GMA':  'E', 'GND': 'X', 'GPL': 'K', 'GSC':  'G', 'GSU': 'E', 'GT9':
'C', 'GVL':  'S', 'H14': 'F',  'H5M': 'P', 'HAC':  'A', 'HAR': 'R',  'HBN': 'H',
'HCS': 'X', 'HFA':  'X', 'HGL': 'X', 'HHI': 'H', 'HIA':  'H', 'HIC': 'H', 'HIP':
'H', 'HIQ':  'H', 'HIS': 'H',  'HL2': 'L', 'HLU':  'L', 'HMR': 'R',  'HPC': 'F',
'HPE': 'F', 'HPH':  'F', 'HPQ': 'F', 'HQA': 'A', 'HRG':  'R', 'HRP': 'W', 'HS8':
'H', 'HS9':  'H', 'HSE': 'S',  'HSL': 'S', 'HSO':  'H', 'HTI': 'C',  'HTN': 'N',
'HTR': 'W', 'HV5':  'A', 'HVA': 'V', 'HY3': 'P', 'HYP':  'P', 'HZP': 'P', 'I2M':
'I', 'I58':  'K', 'IAM': 'A',  'IAR': 'R', 'IAS':  'D', 'IEL': 'K',  'IGL': 'G',
'IIL': 'I', 'ILE':  'I', 'ILG': 'E', 'ILX': 'I', 'IML':  'I', 'IOY': 'F', 'IPG':
'G', 'IT1':  'K', 'IYR': 'Y',  'IYT': 'T', 'IZO':  'M', 'JJJ': 'C',  'JJK': 'C',
'JJL': 'C', 'K1R':  'C', 'KCX': 'K', 'KGC': 'K', 'KNB':  'A', 'KOR': 'M', 'KPI':
'K', 'KST':  'K', 'KYN': 'W',  'KYQ': 'K', 'L2A':  'X', 'LA2': 'K',  'LAA': 'D',
'LAL': 'A', 'LBY':  'K', 'LCK': 'K', 'LCX': 'K', 'LCZ':  'X', 'LDH': 'K', 'LED':
'L', 'LEF':  'L', 'LEH': 'L',  'LEI': 'V', 'LEM':  'L', 'LEN': 'L',  'LET': 'K',
'LEU': 'L', 'LEX':  'L', 'LHC': 'X', 'LLP': 'K', 'LLY':  'K', 'LME': 'E', 'LMF':
'K', 'LMQ':  'Q', 'LP6': 'K',  'LPD': 'P', 'LPG':  'G', 'LPL': 'X',  'LPS': 'S',
'LSO': 'K', 'LTA':  'X', 'LTR': 'W', 'LVG': 'G', 'LVN':  'V', 'LYF': 'K', 'LYK':
'K', 'LYM':  'K', 'LYN': 'K',  'LYR': 'K', 'LYS':  'K', 'LYX': 'K',  'LYZ': 'K',
'M0H': 'C',  'M2L': 'K', 'M2S': 'M',  'M30': 'G', 'M3L': 'K',  'MA': 'A', 'MAA':
'A', 'MAI':  'R', 'MBQ': 'Y',  'MC1': 'S', 'MCG':  'X', 'MCL': 'K',  'MCS': 'C',
'MD3': 'C', 'MD6':  'G', 'MDF': 'Y', 'MDH': 'X', 'MEA':  'F', 'MED': 'M', 'MEG':
'E', 'MEN':  'N', 'MEQ': 'Q',  'MET': 'M', 'MEU':  'G', 'MF3': 'X',  'MGG': 'R',
'MGN': 'Q', 'MGY':  'G', 'MHL': 'L', 'MHO': 'M', 'MHS':  'H', 'MIS': 'S', 'MK8':
'L', 'ML3':  'K', 'MLE': 'L',  'MLL': 'L', 'MLY':  'K', 'MLZ': 'K',  'MME': 'M',
'MMO': 'R', 'MND':  'N', 'MNL': 'L', 'MNV': 'V', 'MOD':  'X', 'MP8': 'P', 'MPH':
'X', 'MPJ':  'X', 'MPQ': 'G',  'MSA': 'G', 'MSE':  'M', 'MSL': 'M',  'MSO': 'M',
'MSP': 'X', 'MT2':  'M', 'MTY': 'Y', 'MVA': 'V', 'N10':  'S', 'N2C': 'X', 'N7P':
'P', 'N80':  'P', 'N8P': 'P',  'NA8': 'A', 'NAL':  'A', 'NAM': 'A',  'NB8': 'N',
'NBQ': 'Y', 'NC1':  'S', 'NCB': 'A', 'NCY': 'X', 'NDF':  'F', 'NEM': 'H', 'NEP':
'H', 'NFA':  'F', 'NHL': 'E',  'NIY': 'Y', 'NLE':  'L', 'NLN': 'L',  'NLO': 'L',
'NLP': 'L', 'NLQ':  'Q', 'NMC': 'G', 'NMM': 'R', 'NNH':  'R', 'NPH': 'C', 'NPI':
'A', 'NSK':  'X', 'NTR': 'Y',  'NTY': 'Y', 'NVA':  'V', 'NYS': 'C',  'NZH': 'H',
'O12': 'X', 'OAR':  'R', 'OAS': 'S', 'OBF': 'X', 'OBS':  'K', 'OCS': 'C', 'OCY':
'C', 'OHI':  'H', 'OHS': 'D',  'OIC': 'X', 'OLE':  'X', 'OLT': 'T',  'OLZ': 'S',
'OMT': 'M', 'ONH':  'A', 'ONL': 'X', 'OPR': 'R', 'ORN':  'A', 'ORQ': 'R', 'OSE':
'S', 'OTB':  'X', 'OTH': 'T',  'OXX': 'D', 'P1L':  'C', 'P2Y': 'P',  'PAQ': 'Y',
'PAS': 'D', 'PAT':  'W', 'PAU': 'A', 'PBB': 'C', 'PBF':  'F', 'PCA': 'E', 'PCC':
'P', 'PCE':  'X', 'PCS': 'F',  'PDL': 'X', 'PEC':  'C', 'PF5': 'F',  'PFF': 'F',
'PFX': 'X', 'PG1':  'S', 'PG9': 'G', 'PGL': 'X', 'PGY':  'G', 'PH6': 'P', 'PHA':
'F', 'PHD':  'D', 'PHE': 'F',  'PHI': 'F', 'PHL':  'F', 'PHM': 'F',  'PIV': 'X',
'PLE': 'L', 'PM3':  'F', 'POM': 'P', 'PPN': 'F', 'PR3':  'C', 'PR9': 'P', 'PRO':
'P', 'PRS':  'P', 'PSA': 'F',  'PSH': 'H', 'PTA':  'X', 'PTH': 'Y',  'PTM': 'Y',
'PTR': 'Y', 'PVH':  'H', 'PVL': 'X', 'PYA': 'A', 'PYL':  'O', 'PYX': 'C', 'QCS':
'C', 'QMM':  'Q', 'QPA': 'C',  'QPH': 'F', 'R1A':  'C', 'R4K': 'W',  'RE0': 'W',
'RE3': 'W', 'RON':  'X', 'RVX': 'S', 'RZ4': 'S', 'S1H':  'S', 'S2C': 'C', 'S2D':
'A', 'S2P':  'A', 'SAC': 'S',  'SAH': 'C', 'SAR':  'G', 'SBL': 'S',  'SCH': 'C',
'SCS': 'C', 'SCY':  'C', 'SD2': 'X', 'SDP': 'S', 'SE7':  'A', 'SEB': 'S', 'SEC':
'U', 'SEG':  'A', 'SEL': 'S',  'SEM': 'S', 'SEN':  'S', 'SEP': 'S',  'SER': 'S',
'SET': 'S', 'SGB':  'S', 'SHC': 'C', 'SHP': 'G', 'SHR':  'K', 'SIB': 'C', 'SLR':
'P', 'SLZ':  'K', 'SMC': 'C',  'SME': 'M', 'SMF':  'F', 'SNC': 'C',  'SNN': 'N',
'SOC': 'C', 'SOY':  'S', 'SRZ': 'S', 'STY': 'Y', 'SUB':  'X', 'SUN': 'S', 'SVA':
'S', 'SVV':  'S', 'SVW': 'S',  'SVX': 'S', 'SVY':  'S', 'SVZ': 'S',  'SYS': 'C',
'T11': 'F', 'T66':  'X', 'TA4': 'X', 'TAV': 'D', 'TBG':  'V', 'TBM': 'T', 'TCQ':
'Y', 'TCR':  'W', 'TDD': 'L',  'TFQ': 'F', 'TH6':  'T', 'THC': 'T',  'THO': 'X',
'THR': 'T', 'THZ':  'R', 'TIH': 'A', 'TMB': 'T', 'TMD':  'T', 'TNB': 'C', 'TNR':
'S', 'TOQ':  'W', 'TPH': 'X',  'TPL': 'W', 'TPO':  'T', 'TPQ': 'Y',  'TQI': 'W',
'TQQ': 'W', 'TRF':  'W', 'TRG': 'K', 'TRN': 'W', 'TRO':  'W', 'TRP': 'W', 'TRQ':
'W', 'TRW':  'W', 'TRX': 'W',  'TRY': 'W', 'TST':  'X', 'TTQ': 'W',  'TTS': 'Y',
'TXY': 'Y', 'TY1':  'Y', 'TY2': 'Y', 'TY3': 'Y', 'TY5':  'Y', 'TYB': 'Y', 'TYI':
'Y', 'TYJ':  'Y', 'TYN': 'Y',  'TYO': 'Y', 'TYQ':  'Y', 'TYR': 'Y',  'TYS': 'Y',
'TYT': 'Y', 'TYW':  'Y', 'TYX': 'X', 'TYY': 'Y', 'TZB':  'X', 'TZO': 'X', 'UMA':
'A', 'UN1':  'X', 'UN2': 'X',  'UNK': 'X', 'VAD':  'V', 'VAF': 'V',  'VAL': 'V',
'VB1': 'K', 'VDL':  'X', 'VLL': 'X', 'VLM': 'X', 'VMS':  'X', 'VOL': 'X', 'WLU':
'L', 'WPA':  'F', 'WRP': 'W',  'WVL': 'V', 'X2W':  'E', 'XCN': 'C',  'XCP': 'X',
'XDT': 'T', 'XPL':  'O', 'XPR': 'P', 'XSN': 'N', 'XX1':  'K', 'YCM': 'C', 'YOF':
'Y', 'YTH':  'T', 'Z01': 'A',  'ZAL': 'A', 'ZCL':  'F', 'ZFB': 'X',  'ZU0': 'T',
'ZZJ': 'A'}
# fmt: on


PHI_ATOMS = ["-C", "N", "CA", "C"]
PSI_ATOMS = ["N", "CA", "C", "+N"]
OMEGA_ATOMS = ["CA", "C", "+N", "+CA"]
CHI1_ATOMS = [
    ["N", "CA", "CB", "CG"],
    ["N", "CA", "CB", "CG1"],
    ["N", "CA", "CB", "SG"],
    ["N", "CA", "CB", "OG"],
    ["N", "CA", "CB", "OG1"],
]
CHI2_ATOMS = [
    ["CA", "CB", "CG", "CD"],
    ["CA", "CB", "CG", "CD1"],
    ["CA", "CB", "CG1", "CD1"],
    ["CA", "CB", "CG", "OD1"],
    ["CA", "CB", "CG", "ND1"],
    ["CA", "CB", "CG", "SD"],
]
CHI3_ATOMS = [
    ["CB", "CG", "CD", "NE"],
    ["CB", "CG", "CD", "CE"],
    ["CB", "CG", "CD", "OE1"],
    ["CB", "CG", "SD", "CE"],
]
CHI4_ATOMS = [["CG", "CD", "NE", "CZ"], ["CG", "CD", "CE", "NZ"]]
CHI5_ATOMS = [["CD", "NE", "CZ", "NH1"]]


__all__ = ["load_CVs_singletraj", "load_CVs_ensembletraj", "CustomTopology"]


################################################################################
# Classes
################################################################################


@dataclass
class Bond:
    """Dataclass, that contains information of an atomic bond.

    Attributes:
        resname (str): The name of the residue, this bond belongs to. Although
            bonds belong to residues, they can also have `atom1` or `atom2`
            belonging to a different residue.
        type (Literal["add", "delete", "optional", "optional_delete"]): Defines
            what should be done with this bond. 'add', adds it to the topology and
            raises an Exception if the bond was already present. 'optional' does
            the same as 'add', but without raising an Exception. 'delete' deletes
            this bond from the topology. An Exception is raised, if this bond
            wasn't even in the topology to begin with. 'optional_delete' deletes
            bonds, but doesn't raise an Exception.
        atom1 (Union[str, int]): The name of the first atom. Can be 'CA', 'N', or
            whatever (not limited to proteins). If it is int it can be any other
            atom of the topology (also belonging to a different residue).
        atom2 (Union[str, int]): The name of the second atom. Can be 'CA', 'N', or
            whatever (not limited to proteins). If it is int it can be any other
            atom of the topology (also belonging to a different residue).

    """

    resname: str
    type: Literal["add", "delete", "optional", "optional_delete"]
    atom1: Union[str, int]
    atom2: Union[str, int]

    def __hash__(self) -> int:
        seq = [
            self.resname,
            self.type,
            self.atom1,
            self.atom2,
        ]
        return reduce(xor, map(hash, seq))


@dataclass
class Dihedral:
    resname: str
    type: Literal["OMEGA", "PHI", "PSI", "CHI1", "CHI2", "CHI3", "CHI4", "CHI5"]
    atom1: Union[int, str, None] = None
    atom2: Union[int, str, None] = None
    atom3: Union[int, str, None] = None
    atom4: Union[int, str, None] = None
    delete: bool = False

    def __hash__(self) -> int:
        seq = [
            self.resname,
            self.type,
            self.atom1,
            self.atom2,
            self.atom3,
            self.atom4,
            self.delete,
        ]
        return reduce(xor, map(hash, seq))

    @property
    def new_atoms_def(self) -> list[str]:
        if not self.delete:
            atoms = [self.atom1, self.atom2, self.atom3, self.atom4]
            assert all(
                [isinstance(a, str) for a in atoms]
            ), f"Can only add to dihedral definitions if all atoms are str. {self.delete=}"
            return atoms
        else:
            return []


@dataclass
class NewResidue:
    name: str
    idx: Union[None, int] = None
    resSeq: Union[None, int] = None
    one_letter_code: str = ""
    topology: Optional[md.Topology] = None
    ignore: bool = False
    bonds: list[Bond] = field(default_factory=list)
    dihedrals: list[Dihedral] = field(default_factory=list)
    common_str: Optional[str] = None

    def __hash__(self) -> int:
        seq = [
            self.name,
            self.idx,
            self.resSeq,
            self.one_letter_code,
            self.topology,
            self.ignore,
        ]
        out_hash = reduce(xor, map(hash, seq))
        if self.bonds:
            out_hash ^= reduce(xor, [hash(b) for b in self.bonds])
        else:
            out_hash ^= hash(None)
        if self.dihedrals:
            out_hash ^= reduce(xor, [hash(d) for d in self.dihedrals])
        else:
            out_hash ^= hash(None)
        return out_hash

    def parse_bonds_and_dihedrals(
        self,
        bonds_and_dihedrals: dict[
            str, Union[list[tuple[Union[int, str], Union[int, str]]], list[str]]
        ],
    ) -> None:
        if self.bonds or self.dihedrals:
            raise Exception(
                f"The method `parse_bonds_and_dihedrals` works on empty `NewResidue` "
                f"instances. If you want to add bonds or dihedrals use the "
                f"`add_bond` or `add_dihedral` methods."
            )
        for bond_or_dihe_type, atoms_or_bonds in bonds_and_dihedrals.items():
            if "bonds" in bond_or_dihe_type:
                bond_type = bond_or_dihe_type.rstrip("bonds").rstrip("_")
                if bond_type == "":
                    bond_type = "add"
                for bond in atoms_or_bonds:
                    bond = Bond(self.name, bond_type, *bond)
                    self.bonds.append(bond)
            else:
                if isinstance(atoms_or_bonds, str):
                    assert (
                        atoms_or_bonds == "delete"
                    ), f"Only a list of str or 'delete' allowed here."
                    dihe_name = bond_or_dihe_type
                    atoms_or_bonds = []
                    delete = True
                else:
                    dihe_name = bond_or_dihe_type
                    delete = False

                dihedral = Dihedral(
                    self.name, dihe_name, *atoms_or_bonds, delete=delete
                )
                self.dihedrals.append(dihedral)

    def get_dihedral_by_type(self, type: str) -> Dihedral:
        for d in self.dihedrals:
            if d.type == type:
                return d

    def add_bond(self, bond: Bond) -> None:
        assert isinstance(bond, Bond)
        self.bonds.append(bond)

    def add_dihedral(self, dihedral: Dihedral) -> None:
        assert isinstance(dihedral, Dihedral)
        self.dihedrals.append(dihedral)

    def as_amino_acid_dict_entry(self) -> dict[str, Union[str, None]]:
        one_letter_code = None if self.ignore else self.one_letter_code
        return {self.name: one_letter_code}

    def _str_summary(self):
        name = self.name
        one_letter_code = self.one_letter_code
        topology = self.topology
        ignore = self.ignore
        common_str = self.common_str
        out = f"{name=} {one_letter_code=} {topology=} {ignore=} {common_str=}"
        if self.bonds:
            out += "\nBonds:"
            for b in self.bonds:
                out += f"\n{b}"
        if self.dihedrals:
            out += "\nDihedrals:"
            for bd in self.dihedrals:
                out += f"\n{bd}"
        return out

    def __str__(self):
        return self._str_summary()


def _delete_bond(
    top: md.Topology,
    bond: tuple["Atom", "Atom"],
) -> md.Topology:
    """Deletes a bond from a MDTraj topology.

    MDTraj's topology has an easy implementation of `md.Topology.add_bond`.
    However, it is lacking the same functionality for removing bonds. This function
    adds it by creating a new topology from a dataframe specifying the atoms
    (which does not change) and a new np.array specifying the bonds.

    Args:
        top (md.Topology): The topology.
        bond (tuple[Atom, Atom]): A tuple of two mdtraj.core.topology.Atom objects.

    Returns:
        md.Topology: The new topology.

    """
    atoms, bonds = top.to_dataframe()
    # fmt: off
    match1 = np.where((bonds[:, 0] == bond[0].index) & (bonds[:, 1] == bond[1].index))[0]
    match2 = np.where((bonds[:, 1] == bond[0].index) & (bonds[:, 0] == bond[1].index))[0]
    # fmt: on
    if (
        (match1.size == 0 and match2.size == 0)
        or (match1.size > 1 and match2.size == 0)
        or (match1.size == 0 and match2.size > 1)
    ):
        raise Exception(f"Could not identify bond {bond} in topology.")
    elif match1.size == 1 and match2.size != 1:
        match = match1[0]
    elif match1.size != 1 and match2.size == 1:
        match = match2[0]
    else:
        raise Exception

    index = np.ones(len(bonds)).astype(bool)
    index[match] = False
    new_bonds = bonds[index]
    new_top = md.Topology.from_dataframe(atoms, new_bonds)
    return new_top


class CustomTopology:
    """Adds custom topology elements to a topology parsed by MDTraj."""

    def __init__(
        self, *new_residues: NewResidue, traj: Optional["InfoSingle"] = None
    ) -> None:
        self.residues = set([*new_residues])
        self.traj = traj
        self._parsed = False
        global _AMINO_ACID_CODES
        self.amino_acid_codes = _AMINO_ACID_CODES

    @property
    def top(self):
        if not self._parsed:
            top = self.add_bonds()
            self.add_amino_acid_codes()
            self._parsed = True
            self._top = top
        return self._top

    @property
    def new_residues(self) -> list[NewResidue]:
        return list(self.residues)

    def add_new_residue(self, new_residue: NewResidue) -> None:
        self.residues += new_residue

    def __hash__(self) -> int:
        if self.residues:
            return reduce(xor, [hash(r) for r in list(self.residues)])
        else:
            return hash(None)

    def __add__(self, other: CustomTopology) -> CustomTopology:
        return CustomTopology(*(self.residues | other.residues))

    def __eq__(self, other: CustomTopology) -> bool:
        return self.residues == other.residues

    def add_bonds(self) -> md.Topology:
        # Encodermap imports
        from encodermap.misc.misc import _validate_uri

        top = self.traj._get_raw_top()
        for residue in self.residues:
            # search for this residue in the protein
            if residue.idx is not None:
                top_residue = top.residue(residue.idx)
                assert top_residue.name == residue.name, (
                    f"There is no residue with the name {residue.name} "
                    f"and the index {residue.idx} in the topology."
                    f"Residue at index {residue.idx} has the name {residue}."
                )
                top_residues = [top_residue]
            elif residue.resSeq is not None:
                top_residues = [r for r in top.residues if r.resSeq == residue.resSeq]
            else:
                top_residues = [r for r in top.residues if r.name == residue.name]

            # add the bonds of this residue
            for b in residue.bonds:
                current_bonds = [(a1, a2) for a1, a2 in top.bonds]
                a1 = b.atom1
                a2 = b.atom2
                action = b.type
                assert isinstance(a1, (str, int))
                assert isinstance(a2, (str, int))

                _a1 = deepcopy(a1)
                _a2 = deepcopy(a2)

                # iterate over the found residues
                for r in top_residues:
                    index = r.index

                    # find atom 1 by str
                    if isinstance(_a1, str):
                        if _a1.startswith("-"):
                            a1 = _a1.lstrip("-")
                            a1_r = top.residue(index - 1)
                            a1 = [a for a in a1_r.atoms if a.name == a1]
                        elif _a1.startswith("+"):
                            a1 = _a1.lstrip("+")
                            a1_r = top.residue(index + 1)
                            a1 = [a for a in a1_r.atoms if a.name == a1]
                        else:
                            a1 = [a for a in r.atoms if a.name == _a1]
                        if len(a1) == 0:
                            if "optional" in action:
                                continue
                            raise Exception(
                                f"Atom {_a1} not part of residue {r}: {a1=}, {b=}"
                            )
                        elif len(a1) > 1:
                            raise Exception(
                                f"Multiple atoms with same name in residue {r}: {a1=}"
                            )
                        a1 = a1[0]

                    # find by int
                    elif isinstance(_a1, int):
                        a1 = top.atom(_a1)
                    else:
                        raise Exception(
                            f"Wrong type: {type(_a1)=}. Needs to be str or int."
                        )

                    # find atom 2 by str
                    if isinstance(_a2, str):
                        if _a2.startswith("-"):
                            a2 = _a2.lstrip("-")
                            a2_r = top.residue(index - 1)
                            a2 = [a for a in a2_r.atoms if a.name == a2]
                        elif _a2.startswith("+"):
                            a2 = _a2.lstrip("+")
                            try:
                                a2_r = top.residue(index + 1)
                            except IndexError:
                                continue
                            a2 = [a for a in a2_r.atoms if a.name == a2]
                        else:
                            a2 = [a for a in r.atoms if a.name == _a2]
                        assert isinstance(a2, list)
                        if len(a2) == 0:
                            if "optional" in action:
                                continue
                            raise Exception(
                                f"Atom {_a2} not part of residue {r}: {a2=}, {b=}"
                            )
                        elif len(a2) > 1:
                            raise Exception(
                                f"Multiple atoms with same name in residue {r}: {a2=}"
                            )
                        a2 = a2[0]

                    # find atom 2 by int
                    elif isinstance(_a2, int):
                        a2 = top.atom(_a2)
                    else:
                        raise Exception(
                            f"Wrong type: {type(_a2)=}. Needs to be str or int."
                        )

                    # decide what to do with this bond
                    if action == "add" or action == "optional":
                        if a1.residue.chain.index != a2.residue.chain.index:
                            self.combine_chains(
                                a1.residue.chain.index, a2.residue.chain.index
                            )
                        if (a1, a2) not in current_bonds and (
                            a2,
                            a1,
                        ) not in current_bonds:
                            top.add_bond(a1, a2)
                            assert (a1, a2) in [(n1, n2) for n1, n2 in top.bonds] or (
                                a2,
                                a1,
                            ) in [(n1, n2) for n1, n2 in top.bonds]
                    elif action == "delete" or action == "optional_delete":
                        if (a1, a2) in current_bonds or (a2, a1) in current_bonds:
                            top = _delete_bond(top, (a1, a2))
                    else:
                        raise Exception(
                            f"Bond action must be 'add', 'optional', 'delete', or "
                            f"'optional_delete'. I got: {action}."
                        )
        return top

    def combine_chains(self, chain_id1: int, chain_id2: int) -> None:
        raise NotImplementedError(
            f"Currently not able to make a new bond across two different chains. "
            f"Should be easy though, just make the atoms to bonds and pandas dataframe "
            f"and manually combine and renumber chains. But don't have the time to do it now."
        )

    # @functools.cache
    def _atom_dict(self) -> dict[Any, Any]:
        return _construct_atom_dict(self.top)

    def get_single_residue_atom_ids(
        self,
        atom_names: list[str],
        r: NewResidue,
    ) -> np.ndarray:
        new_defs = []
        offsets = parse_offsets(atom_names)
        atom_names = _strip_offsets(atom_names)
        for chain in self.top.chains:
            cid = chain.index
            for residue in chain.residues:
                rid = residue.index
                if residue.name == r.name:
                    if r.resSeq is not None:
                        if residue.resSeq != r.resSeq:
                            continue
                    if r.idx is not None:
                        if rid != r.idx:
                            continue
                    new_def = [
                        self._atom_dict()[cid][rid + offset][atom]
                        for atom, offset in zip(atom_names, offsets)
                    ]
                    new_defs.append(new_def)
        return np.asarray(new_defs)

    def backbone_sequence(
        self,
        atom_names: list[str],
        type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        default_indices, default_sequence = _atom_sequence(self.top, atom_names)
        for r in self.residues:
            if r.ignore:
                continue
            d = r.get_dihedral_by_type(type)
            if d is None:
                warnings.warn(
                    f"Your custom topology for residue name={r.name} resSeq={r.resSeq} "
                    f"index={r.idx} does not define atoms for the dihedral {type}. "
                    f"If this dihedral consists of standard atom names, it "
                    f"will be considered for dihedral calculations. If this "
                    f"dihedral should not be present in your custom topology you "
                    f"need to explicitly delete it by adding "
                    f"'{type.upper()}_ATOMS': 'delete' to your custom_topology. "
                    f"If you want this dihedral to be present in your topology, "
                    f"you can ignore this warning."
                )
                continue
            if not d.delete:
                if d.new_atoms_def != atom_names:
                    new_defs = self.get_single_residue_atom_ids(d.new_atoms_def, r)
                    raise NotImplementedError(
                        f"Add this backbone angle with non-standard atoms to the indices"
                    )
                else:
                    new_defs = self.get_single_residue_atom_ids(d.new_atoms_def, r)
                    assert all(
                        [
                            np.any(
                                np.all(
                                    new_def == default_sequence,
                                    axis=1,
                                )
                            )
                            for new_def in new_defs
                        ]
                    )
            if d.delete:
                if type == "PSI":
                    atoms = ["N", "CA", "C", "+N"]
                elif type == "PHI":
                    atoms = ["-C", "N", "CA", "C"]
                elif type == "OMEGA":
                    atoms = ["CA", "C", "+N", "+CA"]
                else:
                    raise Exception(
                        f"The dihedral angle type {type} is not recognized."
                    )
                delete_defs = self.get_single_residue_atom_ids(atoms, r)
                delete_idx = np.all(default_sequence == delete_defs, axis=1)
                if not delete_idx.any():
                    warnings.warn(
                        f"Your custom topology requested the dihedral {d.type} "
                        f"of the residue name={r.name} index={r.idx} resSeq={r.resSeq} "
                        f"to be deleted. This dihedral was not found when searching "
                        f"for standard dihedral names. It is not present in the "
                        f"custom topology and it would also not be present if you "
                        f"haven't specified it to be deleted. If this is unexpected "
                        f"to you, you need to reevaluate your topology."
                    )
                delete_idx = np.where(delete_idx)[0]
                assert len(delete_idx) == 1
                delete_idx = delete_idx[0]
                # msg = f"{delete_idx=}\n\n"
                # for i, row in enumerate(default_sequence):
                #     msg += f"{i:<3} {row}\n"
                default_sequence = np.delete(default_sequence, delete_idx, 0)
                # for delete_def in delete_defs:
                #     idx = np.all(delete_def == default_sequence, axis=1)
                #     assert np.where(idx)[0].size == 1
                #     default_indices = default_indices[~idx]
                #     default_sequence = default_sequence[~idx]
        # return default_indices, default_sequence
        return default_sequence

    def sidechain_sequence(
        self, atom_names: list[str], type: str
    ) -> tuple[np.ndarray, np.ndarray]:
        top = self.top
        for r in self.residues:
            d = r.get_dihedral_by_type(type)
            if d is None:
                continue
            if "OT" in d.new_atoms_def:
                atoms_def = d.new_atoms_def.copy()
                atoms_def[atoms_def.index("OT")] = "OXT"
            else:
                atoms_def = d.new_atoms_def.copy()
            if atoms_def not in atom_names:
                atom_names.append(atoms_def)
            if d.delete:
                raise NotImplementedError(
                    "Currently can not delete custom sidechain dihedrals."
                )
        return self._indices_chi(atom_names)

    def _indices_chi(self, chi_atoms: Sequence[list[str]]) -> np.ndarray:
        rids, indices = zip(*(self._atom_sequence(atoms) for atoms in chi_atoms))
        id_sort = np.argsort(np.concatenate(rids))
        if not any(x.size for x in indices):
            return np.empty(shape=(0, 4), dtype=int)
        indices = np.vstack([x for x in indices if x.size])[id_sort]
        return indices

    def _atom_sequence(self, atom_names, residue_offsets=None):
        if residue_offsets is None:
            residue_offsets = parse_offsets(atom_names)
        atom_names = _strip_offsets(atom_names)
        atom_dict = self._atom_dict()

        atom_indices = []
        found_residue_ids = []
        atoms_and_offsets = list(zip(atom_names, residue_offsets))
        for chain in self.top.chains:
            cid = chain.index
            for residue in chain.residues:
                rid = residue.index
                # Check that desired residue_IDs are in dict
                if all([rid + offset in atom_dict[cid] for offset in residue_offsets]):
                    # Check that we find all atom names in dict
                    if all(
                        [
                            atom in atom_dict[cid][rid + offset]
                            for atom, offset in atoms_and_offsets
                        ]
                    ):
                        # Lookup desired atom indices and add to list
                        atom_indices.append(
                            [
                                atom_dict[cid][rid + offset][atom]
                                for atom, offset in atoms_and_offsets
                            ]
                        )
                        found_residue_ids.append(rid)

        atom_indices = np.array(atom_indices)
        found_residue_ids = np.array(found_residue_ids)

        if len(atom_indices) == 0:
            atom_indices = np.empty(shape=(0, 4), dtype=int)

        return found_residue_ids, atom_indices

    def atom_sequence(
        self,
        type: Literal["PHI", "PSI", "OMEGA", "CHI1", "CHI2", "CHI3", "CHI4", "CHI5"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns either backbone or sidechain indices in a useful order.

        Args:
            type (Literal["OMEGA", "PHI", "PSI", "CHI1", "CHI2", "CHI3", "CHI4", "CHI5"]):
                The angle, that is looked for.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:


        """
        atom_names = globals()[f"{type}_ATOMS"]
        if isinstance(atom_names[0], str):
            return self.backbone_sequence(atom_names, type)
        else:
            return self.sidechain_sequence(atom_names, type)

    def indices_chi1(self):
        return self.atom_sequence(type="CHI1")

    def indices_chi2(self):
        return self.atom_sequence(type="CHI2")

    def indices_chi3(self):
        return self.atom_sequence(type="CHI3")

    def indices_chi4(self):
        return self.atom_sequence(type="CHI4")

    def indices_chi5(self):
        return self.atom_sequence(type="CHI5")

    def indices_psi(self):
        return self.atom_sequence(type="PSI")

    def indices_phi(self):
        return self.atom_sequence(type="PHI")

    def indices_omega(self):
        return self.atom_sequence(type="OMEGA")

    def add_amino_acid_codes(self) -> None:
        self.amino_acid_codes |= {r.name: r.one_letter_code for r in self.new_residues}

    # def get_residues_by_common_str(self, common_str: str) -> Generator[NewResidue]:
    #     for r in self.new_residues:
    #         if r.common_str is None:
    #             yield r
    #         elif r.common_str == common_str:
    #             yield r
    #
    # def get_residues_by_top(self, top: md.Topology) -> Generator[NewResidue]:
    #     for r in self.new_residues:
    #         if r.topology is None:
    #             yield r
    #         elif r.topology == top:
    #             yield r

    def _str_summary(self):
        out = []
        for r in self.new_residues:
            out.append(str(r))
        return "\n\n".join(out)

    def __str__(self):
        return self._str_summary()

    def to_dict(self) -> CustomAAsDict:
        out = {}
        for r in self.new_residues:
            key = r.name
            if r.resSeq is not None:
                assert r.idx is None, f"Can't have resSeq and idx be not None."
                key = f"{key}{r.resSeq}"
            if r.idx is not None:
                assert r.idx is None, f"Can't have resSeq and idx be not None."
                key = f"{key}-{r.index}"
            if r.common_str:
                key = (r.common_str, key)

            if r.ignore:
                out[key] = None
                continue

            one_letter_code = r.one_letter_code
            def_dict = {}
            for b in r.bonds:
                if b.type == "add":
                    btype = "bonds"
                else:
                    btype = f"{b.type}_bonds"
                def_dict.setdefault(btype, []).append((b.atom1, b.atom2))
            for d in r.dihedrals:
                if not d.delete:
                    def_dict[d.type] = [d.atom1, d.atom2, d.atom3, d.atom4]
                else:
                    def_dict["not_" + d.type] = [d.atom1, d.atom2, d.atom3, d.atom4]
            out[key] = (one_letter_code, def_dict)
        return out

    def to_yaml(self, path: Union[str, Path]) -> None:
        data = self.to_dict()
        with open(path, "w") as f:
            yaml.safe_dump(data, f)

    @classmethod
    def from_yaml(cls, path: Union[str, Path], traj: Optional["SingleTraj"] = None):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data, traj)

    @classmethod
    def from_dict(cls, custom_aas: CustomAAsDict, traj: Optional["InfoSingle"] = None):
        """Instantiate the class from a dictionary.

        Args:
            custom_aas (CustomAAsDict):
                Custom AAs defined by a dict with the following properties:
                The keys are the residue names encountered in this traj. The
                values to the keys can be one of three types:
                    * None: if a key: None pair is supplied, this just adds the
                        residue to the recognized residues. Nothing will be done
                        with it.
                    * str: If a key: str pair is supplied, it is expected that the
                        string matches one of the one-letter amino-acid codes.
                        If your new residue is based on Lysine and you named it
                        LYQ, you need to supply: {"LYQ": "K"}
                    * tuple[str, dict]: If your residue has nonstandard side-chain
                        angles (i.e. due to phosphorylation), you can supply a tuple
                        of the one-letter amino-acid code and a dict which defines
                        the sidechain angles like so:
                        {"THR": ("T", {"CHI2": ["CA", "CB", "CG", "P"]})}
                        In this example, the standard amino acid threonine was
                        phosphorylated. The chi2 angle was added.
                        If you want to add custom bonds you can add the "bond" key
                        to the dict and give it either atom names or atom indices of
                        other atoms like so:
                        {"LYQ": ("K", {"bonds": [("N", "CA"), ("N", "H"), ...], "CHI1": ["N", "CA", "CB", "CG"]}).
                    * tuple[str, str, dict]: In this case, the first string should
                        be the name of the amino-acid, the second string should
                        be a common_str, that is in `self.common_str`. That way,
                        the different topologies in this `TrajEnsemble` can dynamically
                        use different custom_aas.

        """
        new_residues = []
        for resname, value in custom_aas.items():
            if isinstance(resname, tuple):
                common_str = resname[0]
                resname = resname[1]
            else:
                common_str = None

            if common_str is not None and traj.common_str != "":
                if common_str != traj.common_str:
                    continue

            if value is not None:
                assert (
                    len(value) == 2
                ), f"The custom_aas dict needs a tuple[str, dict] as its values."

            if "-" in resname:
                idx = int(re.findall("\d+", resname)[-1])
                resSeq = None
                resname = resname.replace(f"-{idx}", "")
            elif any(re.findall("\d+", resname)) and "-" not in resname:
                idx = None
                resSeq = int(re.findall("\d+", resname)[-1])
                resname = resname.replace(resSeq, "")
            else:
                idx = None
                resSeq = None

            if value is None:
                residue = NewResidue(
                    name=resname,
                    common_str=common_str,
                    resSeq=resSeq,
                    idx=idx,
                    ignore=True,
                )
                new_residues.append(residue)
                continue

            one_letter_code = value[0]
            value = value[1]
            residue = NewResidue(
                name=resname,
                one_letter_code=one_letter_code,
                idx=idx,
                resSeq=resSeq,
            )
            residue.parse_bonds_and_dihedrals(value)
            new_residues.append(residue)
        return cls(*new_residues, traj=traj)


################################################################################
# Utils
################################################################################


def trajs_combine_attrs(
    args: Sequence[dict[str, Union[str, np.ndarray]]],
    context: Optional[xr.Context] = None,  # noqa: U100
) -> dict[str, Any]:
    """Used for combining attributes and checking, whether CVs stay in the same unit system.

    Args:
        args (Sequence[dict[str, Any]]): A sequence of dicts to combine.
        context (Optional[xr.Context]): An xarray.Context object.
            Currently not used in the function, but xarray passes it nonetheless

    Returns:
        dict[str, Any]: The combined dict.

    """
    concat = {
        "full_path": "full_paths",
        "topology_file": "topology_files",
        "feature_axis": "feature_axes",
    }
    out = {}
    arrays = []

    if all([not bool(a) for a in args]):
        return {}
    types = []
    for d in args:
        for k, v in d.items():
            types.append(v)

    for d in args:
        for k, v in d.items():
            if k in concat:
                out.setdefault(concat[k], []).append(v)
                continue
            if k in concat.values():
                out.setdefault(k, []).append(v)
                continue
            if isinstance(v, list):
                if all([isinstance(i, (int, float, complex)) for i in v]):
                    inhomogeneous_shapes = False
                elif all([isinstance(i, np.ndarray) for i in v]):
                    inhomogeneous_shapes = (
                        len(np.unique(np.asarray([i.shape for i in v]))) != 1
                    )
                else:
                    dtypes = set([type(i) for i in v])
                    content = v[:2]
                    raise Exception(
                        f"I can't combine `_CVs.attrs` with such "
                        f"inhomogeneous datatypes: {dtypes},"
                        f"{content}"
                    )
                if inhomogeneous_shapes:
                    if k not in out:
                        out.setdefault(k, []).append(v)
                    else:
                        if any(
                            [
                                all([np.array_equal(i, j) for i, j in zip(v, w)])
                                for w in out[k]
                            ]
                        ):
                            continue
                else:
                    v = np.asarray(v)
                    if k not in out:
                        out.setdefault(k, []).append(v)
                    else:
                        if any([np.array_equal(v, a) for a in out[k]]):
                            continue
                arrays.append(k)
                continue
            if isinstance(v, np.ndarray):
                if k not in out:
                    out.setdefault(k, []).append(v)
                else:
                    if any([np.array_equal(v, a) for a in out[k]]):
                        continue
                arrays.append(k)
                continue
            if k not in out:
                out[k] = v
            else:
                if out[k] != v:
                    raise Exception(
                        f"The passed attributes differ in {k}: "
                        f"value1={v}, value2={out[k]}. There is no "
                        f"way to consolidate collective variables in "
                        f"different unit systems."
                    )
    for k, v in concat.items():
        if v in out:
            if isinstance(out[v], list):
                if len(out[v]) == 1:
                    out[k] = out.pop(v)[0]
                    continue
                if all([i == out[v][0] for i in out[v][1:]]):
                    out[v] = out[v][0]
                    continue
    for k, v in concat.items():
        if v in out:
            if not isinstance(out[v], list):
                out[k] = out.pop(v)
    for a in arrays:
        if len(out[a]) == 1:
            out[a] = out[a][0]
    for k, v in out.items():
        if v is None:
            out[k] = np.array([])
    return out


def np_to_xr(
    data: np.ndarray,
    traj: SingleTraj,
    attr_name: Optional[str] = None,
    deg: Optional[bool] = None,
    labels: Optional[list[str]] = None,
    filename: Optional[Union[str, Path]] = None,
) -> xr.DataArray:
    if attr_name is None:
        if filename is None:
            msg = f"Please also provide an `attr_name` under which to save the CV."
            raise Exception(msg)
        attr_name = Path(filename).stem

    if labels is not None:
        if isinstance(labels, str):
            labels = [
                f"{attr_name.upper()} {labels.upper()} {i}"
                for i in range(data.shape[1])
            ]
        elif all([isinstance(l, str) for l in labels]) and len(labels) == data.shape[1]:
            pass
        else:
            raise Exception(
                f"'labels' is either not a list of str or does not have the "
                f"same number of datapoints as {data.shape=}, {labels[:5]=} "
                f"{len(labels)=}."
            )
    data = np.expand_dims(data, axis=0)
    if np.any(np.isnan(data)):
        # if some nans are found along frame remove them
        if data.ndim == 2:
            data = data[:, ~np.isnan(data).any(axis=0)]
        if data.ndim == 3:
            data = data[:, ~np.isnan(data).any(axis=2)[0]]
        if data.ndim == 4:
            data = data[:, ~np.isnan(data).any(axis=2)[0].any(axis=1)]
    da = construct_xarray_from_numpy(
        traj, data, attr_name, deg, labels, check_n_frames=True
    )
    assert len(da.dims) >= 3, f"{da=}"
    return da


def load_CV_from_string_or_path(
    file_or_feature: str,
    traj: SingleTraj,
    attr_name: Optional[str] = None,
    cols: Optional[Union[int, list[int]]] = None,
    deg: Optional[bool] = None,
    labels: Optional[list[str]] = None,
) -> xr.Dataset:
    """Loads CV data from a string. That string can either identify a features,
    or point to a file.

    Args:
        file_or_feature (str): The file or feature to load. If 'all' is
            provided, all "standard" features are loaded. But a feature name
            like 'sidechain_angle' can alsop be provided. If a file with
            the .txt or .npy extension is provided, the data in that file is used.
        traj (SingleTraj): The trajectory, that is used to load the features.
        attr_name (Union[None, str], optional): The name under which the CV should be found in the class.
            Is needed, if a raw numpy array is passed, otherwise the name will be generated from the filename
            (if data == str), the DataArray.name (if data == xarray.DataArray), or the feature name.
        cols (Union[list, None], optional): A list specifying the columns to use for the highD data.
            If your highD data contains (x,y,z,...)-errors or has an enumeration
            column at col=0 this can be used to remove this unwanted data.
        deg (Optional[bool]): Whether the provided data is in radians (False)
                or degree (True). Can also be None for non-angular data.
        labels (Union[list, str, None], optional): If you want to label the data you provided pass a list of str.
            If set to None, the features in this dimension will be labelled as
            [f"{attr_name.upper()} FEATURE {i}" for i in range(self.n_frames)]. If a str is provided, the features
            will be labelled as [f"{attr_name.upper()} {label.upper()} {i}" for i in range(self.n_frames)]. If a list of str
            is provided it needs to have the same length as the traj has frames. Defaults to None.

    Returns:
        xr.Dataset: An xarray dataset.

    """
    if str(file_or_feature) == "all" or str(file_or_feature) in CAN_BE_FEATURE_NAME:
        # feat = Featurizer(traj)
        if file_or_feature == "all":
            traj.featurizer.add_list_of_feats(which="all", deg=deg)
        else:
            traj.featurizer.add_list_of_feats(which=[file_or_feature], deg=deg)
        out = traj.featurizer.get_output()
        if traj.traj_num is not None:
            assert out.coords["traj_num"] == np.array([traj.traj_num]), print(
                traj.traj_num,
                out.coords["traj_num"].values,
                traj.traj_num,
            )
        return out
    elif (f := Path(file_or_feature)).exists():
        if f.suffix == ".txt":
            data = np.loadtxt(f, usecols=cols)
        elif f.suffix == ".npy":
            data = np.load(f)
            if cols is not None:
                data = data[:, cols]
        elif f.suffix in [".nc", ".h5"]:
            data = xr.open_dataset(f)
            if len(data.data_vars.keys()) != 1:
                if attr_name is not None:
                    raise Exception(
                        f"The dataset in {f} has "
                        f"{len(data.data_vars.keys())} dataarrays, "
                        f"but only one `attr_name`: '{attr_name}' "
                        f"was requested. The names of the dataarrays "
                        f"are: {data.data_vars.keys()}. I can't over"
                        f"ride them all with one `attr_name`. Set "
                        f"`attr_name` to None to load the data with "
                        f"their respective names"
                    )
                return data
            else:
                if attr_name is not None:
                    d = list(data.data_vars.values())[0]
                    d.name = attr_name
                return d
        else:
            raise Exception(
                f"Currently only .txt, .npy, .nc, and .h5 files can "
                f"be loaded. Your file {f} does not have the "
                f"correct extension."
            )
    else:
        raise Exception(
            f"If features are loaded via a string, the string needs "
            f"to be 'all', a feature name ('central_dihedrals'), or "
            f'an existing file. Your string "{file_or_feature}"'
            f"is none of those."
        )

    da = np_to_xr(data, traj, attr_name, deg, labels, file_or_feature)
    assert len(da.dims) == 3
    return da


def load_CVs_singletraj(
    data: SingleTrajFeatureType,
    traj: SingleTraj,
    attr_name: Optional[str] = None,
    cols: Optional[list[int]] = None,
    deg: Optional[bool] = None,
    labels: Optional[list[str]] = None,
) -> xr.Dataset:
    # Local Folder Imports
    from ..loading.dask_featurizer import DaskFeaturizer
    from ..loading.features import Feature
    from ..loading.featurizer import EnsembleFeaturizer, SingleTrajFeaturizer

    if isinstance(attr_name, str):
        if not attr_name.isidentifier():
            raise Exception(
                f"Provided string for `attr_name` can not be a "
                f"python identifier. Choose another attribute name."
            )
    # load a string
    if isinstance(data, (str, Path)):
        CVs = load_CV_from_string_or_path(str(data), traj, attr_name, cols, deg, labels)

    # load a list of strings from standard features
    elif isinstance(data, list) and all([isinstance(_, str) for _ in data]):
        # feat = Featurizer(traj)
        traj.featurizer.add_list_of_feats(data)
        out = traj.featurizer.get_output()
        out.coords["traj_num"] = [traj.traj_num]
        return out

    # if the data is a numpy array
    elif isinstance(data, (list, np.ndarray)):
        CVs = np_to_xr(np.asarray(data), traj, attr_name, deg, labels).to_dataset(
            promote_attrs=True
        )

    # xarray objects are simply returned
    elif isinstance(data, xr.Dataset):
        return data

    elif isinstance(data, xr.DataArray):
        return data.to_dataset(promote_attrs=True)

    # if this is a feature
    elif issubclass(data.__class__, Feature):
        # feat = Featurizer(traj)
        traj.featurizer.add_custom_feature(data)
        return traj.featurizer.get_output()

    # if an instance of featurizer is provided
    elif isinstance(data, (DaskFeaturizer, SingleTrajFeaturizer, EnsembleFeaturizer)):
        if isinstance(attr_name, str):
            if len(data) != 1:
                raise TypeError(
                    f"Provided Featurizer contains {len(data)} "
                    f"features and `attr_name` is of type `str`. "
                    f"Please provide a list of str."
                )
            attr_name = [attr_name]
        if isinstance(attr_name, list):
            if len(attr_name) != len(data):
                raise IndexError(
                    f"Provided Featurizer contains {len(data)} "
                    f"features and `attr_name` contains "
                    f"{len(attr_name)} elements. Please make sure "
                    f"they contain the same amount of items."
                )
        out = data.get_output()
        assert out.dims["traj_num"] == 1
        if attr_name is not None:
            if isinstance(attr_name, str):
                attr_name = [attr_name]
            _renaming = {}
            for f, v in zip(data.features, attr_name):
                _feature = False
                if hasattr(f, "name"):
                    if f.name in FEATURE_NAMES:
                        k = FEATURE_NAMES[f.name]
                        _feature = True
                if not _feature:
                    k = f.__class__.__name__
                _renaming[k] = v
            out = out.rename_vars(_renaming)
        return out
    else:
        raise TypeError(
            f"`data` must be str, np.ndarray, list, xr.DataArray, xr.Dataset, "
            f"em.Featurizer or em.features.Feature. You supplied "
            f"{type(data)}."
        )

    return CVs


def load_CVs_ensembletraj(
    trajs: TrajEnsemble,
    data: TrajEnsembleFeatureType,
    attr_name: Optional[list[str]] = None,
    cols: Optional[list[int]] = None,
    deg: Optional[bool] = None,
    labels: Optional[list[str]] = None,
    directory: Optional[Union[Path, str]] = None,
    ensemble: bool = False,
    override: bool = False,
) -> None:
    """Loads CVs for a trajectory ensemble.

    CVs can be loaded from a multitude of sources. The argument `data` can be:
        * np.ndarray: Use a numpy array as a feature.
        * str | Path: You can point to .txt or .npy files and load the features
            from these files. In this case, the `cols` argument can be
            used to only use a subset of columns in these files.
            You can also point to a single directory in which case the basename
            of the trajectories will be used to look for .npy and .txt files.
        * str: Some strings like "central_dihedrals" are recognized out-of-the-box.
            You can also provide "all" to load all dihedrals used in an
            `encodermap.AngleDihecralCartesianEncoderMap`.
        * Feature: You can provide an `encodermap.loading.features` Feature. The
            CVs will be loaded by creating a featurizer, adding this feature, and
            obtaining the output.
        * Featurizer: You can also directly provide a featurizer, wiht multiple
            features.
        * xr.DataArray: You can also provide a xarray.DataArray, which will be
            appended to the existing CVs.
        * xr.Dataset: If you provide a xarray.Dataset, you will overwrite all
            currently loaded CVs.

    Args:
        trajs (TrajEnsemble): The trajectory ensemble to load the data for.
        data (Union[str, list, np.ndarray, 'all', xr.Dataset]): The CV to
            load. When a numpy array is provided, it needs to have a shape
            matching `n_frames`. The data is distributed to the trajs.
            When a list of files is provided, `len(data)` needs to match
            `n_trajs`. The first file will be loaded by the first traj
            (based on the traj's `traj_num`) and so on. If a list of
            `np.ndarray` is provided, the first array will be assigned to
            the first traj (based on the traj's `traj_num`). If None is provided,
            the argument `directory` will be used to construct a str like:
            fname = directory + traj.basename + '_' + attr_name. If there are
            .txt or .npy files matching that string in the `directory`,
            the CVs will be loaded from these files to the corresponding
            trajs. Defaults to None.
        attr_name (Optional[str]): The name under which the CV should
            be found in the class. Choose whatever you like. `highd`, `lowd`,
            `dists`, etc. The CV can then be accessed via dot-notation:
            `trajs.attr_name`. Defaults to None, in which case, the argument
            `data` should point to existing files and the `attr_name` will
            be extracted from these files.
        cols (Optional[list[int]]): A list of integers indexing the columns
            of the data to be loaded. This is useful if a file contains
            columns which are not features (i.e. an indexer or the error of
            the features. eg::

                id   f1    f2    f1_err    f2_err
                0    1.0   2.0   0.1       0.1
                1    2.5   1.2   0.11      0.52

            In that case, you would want to supply `cols=[1, 2]` to the `cols`
            argument. If None all columns are loaded. Defaults to None.
        deg (Optional[bool]): Whether to return angular CVs using degrees.
            If None or False, CVs will be in radian. Defaults to None.
        labels (list): A list containing the labels for the dimensions of
            the data. If you provide a `np.ndarra` with shape (n_trajs,
            n_frames, n_feat), this list needs to be of len(n_feat)
            Defaults to None.
        directory (Optional[str]): The directory to save the data at if data
            is an instance of `em.Featurizer` and this featurizer has
            `in_memory` set to Fase. Defaults to ''.
        ensemble (bool): Whether the trajs in this class belong to an ensemble.
            This implies that they contain either the same topology or are
            very similar (think wt, and mutant). Setting this option True will
            try to match the CVs of the trajs onto the same dataset.
            If a VAL residue has been replaced by LYS in the mutant,
            the number of sidechain dihedrals will increase. The CVs of the
            trajs with VAL will thus contain some NaN values. Defaults to False.
        override (bool): Whether to override CVs with the same name as `attr_name`.

    """
    # Local Folder Imports
    from ..loading.features import CustomFeature, Feature
    from ..loading.featurizer import EnsembleFeaturizer, SingleTrajFeaturizer

    if isinstance(data, (str, Path)) and not ensemble:
        path_data = Path(data)
        if not all([t.basename is None for t in trajs]):
            npy_files = [
                (t._traj_file.parent if directory is None else Path(directory))
                / (t.basename + f"_{data}.npy")
                for t in trajs
            ]
            txt_files = [
                (t._traj_file.parent if directory is None else Path(directory))
                / (t.basename + f"_{data}.txt")
                for t in trajs
            ]
            raw_files = [
                (t._traj_file.parent if directory is None else Path(directory))
                / (t.basename + f"_{data}")
                for t in trajs
            ]
        if str(data) == "all":
            [t.load_CV("all", deg=deg, override=override) for t in trajs]
            return
        if path_data.is_dir():
            return load_CVs_from_dir(
                trajs, data, attr_name=attr_name, deg=deg, cols=cols
            )
        elif data in CAN_BE_FEATURE_NAME:
            [t.load_CV(data, attr_name, cols, deg, labels, override) for t in trajs]
            return
        elif path_data.is_file() and (
            path_data.suffix == ".h5" or path_data.suffix == ".nc"
        ):
            ds = xr.open_dataset(path_data)
            if diff := set([t.traj_num for t in trajs]) - set(ds["traj_num"].values):
                raise Exception(
                    f"The dataset you try to load and the TrajEnsemble "
                    f"have different number of trajectories: {diff}."
                )
            for t, (traj_num, sub_ds) in zip(trajs, ds.groupby("traj_num")):
                assert t.traj_num == traj_num, f"{t.traj_num=}, {traj_num=}"
                sub_ds = sub_ds.assign_coords(traj_num=t.traj_num)
                sub_ds = sub_ds.expand_dims("traj_num")
                assert sub_ds.coords["traj_num"] == np.array([t.traj_num])
                t.load_CV(sub_ds)
            return
        elif all([f.is_file() for f in npy_files]):
            [
                t.load_CV(
                    f,
                    attr_name=data,
                    cols=cols,
                    deg=deg,
                    labels=labels,
                    override=override,
                )
                for t, f in zip(trajs, npy_files)
            ]
            return
        elif all([f.is_file() for f in txt_files]):
            [
                t.load_CV(
                    f,
                    attr_name=data,
                    cols=cols,
                    deg=deg,
                    labels=labels,
                    override=override,
                )
                for t, f in zip(trajs, txt_files)
            ]
            return
        elif all([f.is_file() for f in raw_files]):
            [
                t.load_CV(
                    f,
                    attr_name=data,
                    cols=cols,
                    deg=deg,
                    labels=labels,
                    override=override,
                )
                for t, f in zip(trajs, raw_files)
            ]
            return
        else:
            msg = (
                f"If `data` is provided a single string, the string needs to "
                f"be either a feature ({CAN_BE_FEATURE_NAME}), a .h5/.nc file "
                f"or a npy file. The provided `data`={data} "
                f"fits none of these possibilities."
            )
            raise ValueError(msg)

    elif isinstance(data, list) and not ensemble:
        if all([isinstance(i, (list, np.ndarray)) for i in data]):
            [
                t.load_CV(d, attr_name, cols, deg, labels, override)
                for t, d in zip(trajs, data)
            ]
            return
        elif all([i in CAN_BE_FEATURE_NAME for i in data]):
            [t.load_CV(data, attr_name, cols, deg, labels, override) for t in trajs]
            return
        elif all([Path(f).is_file() for f in data]):
            suffix = set([Path(f).suffix for f in data])
            if len(suffix) != 1:
                raise Exception(
                    "Please provide a list with consistent file "
                    f"extensions and not a mish-mash, like: {suffix}"
                )
            suffix = suffix.pop()
            if suffix == ".npy":
                [
                    t.load_CV(np.load(d), attr_name, cols, deg, labels, override)
                    for t, d in zip(trajs, data)
                ]
            else:
                [
                    t.load_CV(
                        np.genfromtxt(d),
                        attr_name,
                        cols,
                        deg,
                        labels,
                        override,
                    )
                    for t, d in zip(trajs, data)
                ]
            return
        else:
            msg = (
                f"If `data` is provided as a list, the list needs to contain "
                f"strings that can be features ({CAN_BE_FEATURE_NAME}), or "
                f"some combination of lists and numpy arrays."
            )
            raise ValueError(msg)

    elif isinstance(data, np.ndarray):
        if len(data) != trajs.n_trajs and len(data) != trajs.n_frames:
            raise ValueError(
                f"The provided numpy array is misshaped. It needs "
                f"to be of shape (n_trajs={trajs.n_trajs}, "
                f"n_frames={np.unique([t.n_frames for t in trajs])[0]}, "
                f"X, (Y)), but is {data.shape}."
            )
        if len(data) == trajs.n_frames:
            data = [
                data[np.where(trajs.index_arr[:, 0] == t.traj_num)[0]] for t in trajs
            ]
        [
            t.load_CV(d, attr_name, cols, deg, labels, override)
            for t, d in zip(trajs, data)
        ]
        for t in trajs:
            for v in t._CVs.values():
                assert v.shape[0] == 1, f"{t.basename=}, {v=}"
        return

    elif issubclass(data.__class__, Feature):
        [t.load_CV(data, attr_name, cols, deg, labels, override) for t in trajs]
        return

    elif isinstance(data, EnsembleFeaturizer):
        ds = data.get_output()
        for t, (traj_num, sub_ds) in zip(trajs, ds.groupby("traj_num")):
            assert t.traj_num == traj_num, f"{t.traj_num=}, {traj_num=}"
            sub_ds = sub_ds.assign_coords(traj_num=t.traj_num)
            sub_ds = sub_ds.expand_dims("traj_num")
            for name, da in sub_ds.data_vars.items():
                if "feature_axis" in da.attrs:
                    sub_ds = sub_ds.dropna(da.attrs["feature_axis"])
            t.load_CV(sub_ds)
        return

    elif isinstance(data, xr.Dataset):
        for i, (t, (traj_num, sub_ds)) in enumerate(
            zip(trajs, data.groupby("traj_num"))
        ):
            assert t.traj_num == traj_num, f"{t.traj_num=}, {traj_num=}"
            sub_ds = sub_ds.assign_coords(traj_num=t.traj_num)
            sub_ds = sub_ds.expand_dims("traj_num")
            sub_ds = sub_ds.dropna("frame_num", how="all")
            t.load_CV(sub_ds)
        return

    if ensemble:
        return load_CVs_ensemble(trajs, data)

    else:
        raise TypeError(
            f"`data` must be str, np.ndarray, list, xr.Dataset, or "
            f"em.Featurizer. You supplied {type(data)=} {data.__class__.__name__=}."
        )


def load_CVs_ensemble(
    trajs: TrajEnsemble,
    data: Union[str, list[str], Literal["all"]],
) -> None:
    """Loads CVs for a trajectory ensemble. This time with generic feature names
    so different topologies are aligned and can be treated separately. Loading
    CVs with ensemble=True will always delete existing CVs.

    Args:
        trajs (TrajEnsemble): The trajectory ensemble to load the data for.
        data (Union[str, list[str], Literal["all']): The CV to
            load. When a numpy array is provided, it needs to have a shape
            matching `n_frames`. The data is distributed to the trajs.
            When a list of files is provided, `len(data)` needs to match
            `n_trajs`. The first file will be loaded by the first traj
            (based on the traj's `traj_num`) and so on. If a list of
            `np.ndarray` is provided, the first array will be assigned to
            the first traj (based on the traj's `traj_num`). If None is provided,
            the argument `directory` will be used to construct a str like:
            fname = directory + traj.basename + '_' + attr_name. If there are
            .txt or .npy files matching that string in the `directory`,
            the CVs will be loaded from these files to the corresponding
            trajs. Defaults to None.

    """
    if isinstance(data, str):
        if data != "all":
            data = [data]
    trajs.featurizer.add_list_of_feats(data, ensemble=True)
    deg_units = []
    for f in trajs.featurizer.features:
        if hasattr(f, "deg"):
            deg_units.append(f.deg)
    assert all(
        [not d for d in deg_units]
    ), "Loading an ensemble only possible if all degree units are radian."
    output = trajs.featurizer.get_output()
    for t, (traj_num, sub_ds) in zip(trajs, output.groupby("traj_num")):
        assert t.traj_num == traj_num, f"{t.traj_num=}, {traj_num=}"
        try:
            sub_ds = sub_ds.assign_coords(traj_num=t.traj_num)
            sub_ds = sub_ds.expand_dims("traj_num")
        except ValueError as e:
            if "already exists as a scalar" not in str(e):
                raise e
        if t.id.ndim == 2:
            frames = t.id[:, 1]
        else:
            frames = t.id
        sub_ds = sub_ds.sel({"frame_num": frames})
        if t._CVs:
            warnings.warn(
                "Using ensemble=True will drop old CV entries from "
                "trajs, because the feature length increases."
            )
        t._CVs = sub_ds


def load_CVs_from_dir(
    trajs: TrajEnsemble,
    data: Path,
    attr_name: Optional[str] = None,
    cols: Optional[list[int]] = None,
    deg: Optional[bool] = None,
) -> None:
    files = map(str, data.glob("*"))
    files = list(
        filter(
            lambda x: True if any([traj.basename in x for traj in trajs]) else False,
            files,
        )
    )
    key = {"npy": 1, "txt": 2}
    files = sorted(
        files,
        key=lambda x: key[x.split(".")[-1]] if x.split(".")[-1] in key else 3,
    )[: trajs.n_trajs]
    files = sorted(
        files,
        key=lambda x: [traj.basename in x for traj in trajs].index(True),
    )
    assert (
        len(files) == trajs.n_trajs
    ), f"Couldn't find the correct number of files:\n{files=}\nfor trajs:\n{trajs=}"
    for traj, f in zip(trajs, files):
        if traj.basename not in f:
            raise Exception(f"File {f} does not contain substring of traj {traj}.")
        traj.load_CV(f, attr_name=attr_name, cols=cols, deg=deg)
