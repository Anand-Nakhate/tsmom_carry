"""
Configuration for the TSMOM Carry project.
Defines the universe of futures contracts to be downloaded and processed.
"""
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class RootContract:
    dataset: str
    parent: str
    asset_class: str
    region: str

# Global Universe Definition
GLBX_UNIVERSE = [
    # Equity
    RootContract("GLBX.MDP3", "ES.FUT",  "Equity",    "US"),
    RootContract("GLBX.MDP3", "NQ.FUT",  "Equity",    "US"),
    RootContract("GLBX.MDP3", "RTY.FUT", "Equity",    "US"),
    RootContract("GLBX.MDP3", "NKD.FUT", "Equity",    "Japan"),
    # Rates
    RootContract("GLBX.MDP3", "ZT.FUT", "Rates", "US"),
    RootContract("GLBX.MDP3", "ZF.FUT", "Rates", "US"),
    RootContract("GLBX.MDP3", "ZN.FUT", "Rates", "US"),
    RootContract("GLBX.MDP3", "ZB.FUT", "Rates", "US"),
    RootContract("GLBX.MDP3", "UB.FUT", "Rates", "US"),
    # FX
    RootContract("GLBX.MDP3", "6E.FUT", "FX", "Global"),
    RootContract("GLBX.MDP3", "6J.FUT", "FX", "Global"),
    RootContract("GLBX.MDP3", "6B.FUT", "FX", "Global"),
    RootContract("GLBX.MDP3", "6A.FUT", "FX", "Global"),
    RootContract("GLBX.MDP3", "6C.FUT", "FX", "Global"),
    RootContract("GLBX.MDP3", "6S.FUT", "FX", "Global"),
    RootContract("GLBX.MDP3", "6N.FUT", "FX", "Global"),
    # Energy
    RootContract("GLBX.MDP3", "CL.FUT", "Commodity", "Global"),
    RootContract("GLBX.MDP3", "NG.FUT", "Commodity", "US"),
    RootContract("GLBX.MDP3", "HO.FUT", "Commodity", "US"),
    RootContract("GLBX.MDP3", "RB.FUT", "Commodity", "US"),
    # Metals
    RootContract("GLBX.MDP3", "GC.FUT", "Commodity", "Global"),
    RootContract("GLBX.MDP3", "SI.FUT", "Commodity", "Global"),
    RootContract("GLBX.MDP3", "HG.FUT", "Commodity", "Global"),
    # Grains
    RootContract("GLBX.MDP3", "ZC.FUT", "Commodity", "US"),
    RootContract("GLBX.MDP3", "ZW.FUT", "Commodity", "US"),
    RootContract("GLBX.MDP3", "ZS.FUT", "Commodity", "US"),
    RootContract("GLBX.MDP3", "ZL.FUT", "Commodity", "US"),
    RootContract("GLBX.MDP3", "ZM.FUT", "Commodity", "US"),
]
