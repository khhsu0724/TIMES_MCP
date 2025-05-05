from mcp.server.fastmcp import FastMCP
from typing import Literal
import base64
import logging
from mcp.types import TextContent, ImageContent
from pathlib import Path
from sympy import Rational, symbols, sqrt, pi, zeros, simplify, pretty

from lib import enumerate_fn_configurations,get_required_coulomb_terms,coulomb_contribution, orbital_l, get_single_states


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP()


@mcp.tool(
    name = "get_hamiltonian",
    description = "Get the Hamiltonian matrix for a given orbital and number of electrons",
)
async def get_hamiltonian(
    orbital: Literal['s', 'p', 'd', 'f'], 
    n_electrons: int, 
    target_ML: int, 
    target_MS: float) -> TextContent:
    """
    Get the Hamiltonian matrix for a given orbital and number of electrons.

    Args:
        orbital: the orbital to consider, must be one of 's', 'p', 'd', 'f'
        n_electrons: the number of electrons
        target_ML: the target ML
        target_MS: the target MS

    Returns:
        The Hamiltonian matrix as a string
    """
    l = orbital_l[orbital]
    single_electron_states = get_single_states(l)
    
    configs = enumerate_fn_configurations(n_electrons, target_ML, target_MS, single_electron_states)

    response = f"Found {len(configs)} valid antisymmetric configurations."

    if len(configs) == 0:
        print("No valid configurations found.")
        return TextContent(text=response, type="text")

    for idx, cfg in enumerate(configs):
        response += f"\nConfig {idx+1}: {cfg}"

    H = zeros(len(configs))

    for i, cfg1 in enumerate(configs):
        for j, cfg2 in enumerate(configs):
            total_contrib = 0
            terms = get_required_coulomb_terms(cfg1, cfg2)
            for sign, kind, (m1, m2, m3, m4) in terms:
                if kind == "direct":
                    total_contrib += sign * coulomb_contribution(m2, m1, m3, m4, l)
                elif kind == "exchange":
                    total_contrib -= sign * coulomb_contribution(m2, m1, m3, m4, l)
            H[i, j] = simplify(total_contrib)

    response += f"\n\nHamiltonian Matrix (Pretty Print):\n{pretty(H)}"

    if len(configs) <= 6:
        response += f"\nEigenvalues:\n"
        eigvals = H.eigenvals()
        for eig, mult in eigvals.items():
            response += f"{pretty(eig)} (multiplicity {mult})\n"
    else:
        response += "\nMatrix too large to diagonalize symbolically."

    response += f"\n\nHamiltonian Matrix (txt):\n{str(H)}"
    return TextContent(text=response, type="text")



def main():
    # Start server
    logger.info('Starting xray-server')
    mcp.run('stdio')
